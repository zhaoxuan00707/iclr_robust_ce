"""Single total-budget constrained particle WRO with exact OT feasibility.

Only B_geo bounds geometry; D is the fixed distance between the anchors.
Current-validity retention uses the reference CPU model before risk selection.
"""
import time
import copy
import torch
import torch.nn.functional as F
from .geodesic_excess import exact_w2,project_basic


def bank_value_gradient(bank,points,beta_h=50.,beta_m=100.,chunk=256,gradient=True):
    """Stream all models; outer softmax weights follow population reduction."""
    chunk=min(chunk,getattr(bank,"chunk_size",chunk))
    soft=[];true=[]
    with torch.no_grad():
        for start in range(0,bank.count,chunk):
            score=torch.sigmoid(bank.logits(points,start,start+chunk))
            soft.append((F.softplus(beta_h*(.5-score))/beta_h).double().mean(1))
            true.append(torch.relu(.5-score).double().mean(1))
        soft=torch.cat(soft);true=torch.cat(true)
        weights=torch.softmax(beta_m*soft,dim=0)
        value=torch.logsumexp(beta_m*soft,dim=0)/beta_m
    grad=torch.zeros_like(points)
    if gradient:
        for start in range(0,bank.count,chunk):
            score=torch.sigmoid(bank.logits(points,start,start+chunk))
            losses=(F.softplus(beta_h*(.5-score))/beta_h).double().mean(1)
            grad+=torch.autograd.grad((weights[start:start+chunk]*losses).sum(),points)[0]
    return float(value),float(true.max()),grad


def geometry_record(source,w,target,D,B_geo,current,lower,upper):
    p0=exact_w2(source,w);p1=exact_w2(w,target)
    length=p0.distance+p1.distance
    with torch.no_grad():
        probability=torch.sigmoid(current(w.detach().to(next(current.parameters()).device)))
        invalid=float((probability<.5).double().mean())
        box=float(((w<lower-1e-7)|(w>upper+1e-7)).any(1).double().mean())
    record={'d0':p0.distance,'d1':p1.distance,'D':D,'total_length':length,'geometry_excess':length-D,'B_geo':B_geo,'beta_geo':B_geo/D,'slack':B_geo-length,
        'lambda_W':p0.distance/(p0.distance+p1.distance),'current_invalid_mass':invalid,
        'current_valid_mass':1-invalid,'minimum_f0_score':float(probability.min()),'invalid_support_count':int((probability<.5).sum()),'below_0.7_mass':float((probability<.7).double().mean()),'box_violation_mass':box,
        'marginal_residual':max(p0.marginal_residual,p1.marginal_residual)}
    return p0,p1,record


def restore_feasible(source,w,target,D,B_geo,current,lower,upper,tolerance,iterations=12):
    """Numerical restoration towards a feasible target, testing the total two-anchor distance.

    This is an initialization/final-candidate diagnostic, not a displacement
    constraint and not a replacement for primal-dual optimization.
    """
    _,_,initial=geometry_record(source,w,target,D,B_geo,current,lower,upper)
    if initial['total_length']<=B_geo+tolerance and initial['current_invalid_mass']==0 and initial['box_violation_mass']==0:
        return w.detach().clone(),1.
    if B_geo==D:return target.detach().clone(),0.
    lo,hi=0.,1.;best=target.detach().clone()
    for _ in range(iterations):
        scale=(lo+hi)/2
        candidate=target+scale*(w-target)
        _,_,row=geometry_record(source,candidate,target,D,B_geo,current,lower,upper)
        if row['total_length']<=B_geo+tolerance and row['current_invalid_mass']==0 and row['box_violation_mass']==0:
            best=candidate.detach().clone();lo=scale
        else:hi=scale
    return best,lo


def solve_control(bank,current,source,target,initial,D,B_geo,lower,upper,settings):
    """Return feasible candidate checkpoints and complete primal/dual trace."""
    if B_geo < D:raise ValueError("B_geo must be at least the anchor distance D")
    reference_current=copy.deepcopy(current).cpu().eval().requires_grad_(False)
    tolerance=settings['feasibility_relative_tolerance']*max(1.,D)
    w=project_basic(initial,target,current,lower,upper).detach().requires_grad_(True)
    optimizer=torch.optim.Adam([w],lr=settings['learning_rate'])
    nu=0.;rho=settings['rho_initial'];history=[];retained=[];best=float('inf');previous_violation=float('inf')
    def retain(point,tag,step):
        nonlocal best
        _,_,row=geometry_record(source,point,target,D,B_geo,reference_current,lower,upper)
        if row['total_length']<=B_geo+tolerance and row['current_invalid_mass']==0 and row['box_violation_mass']==0:
            smooth,risk,_=bank_value_gradient(bank,point,settings['beta_h'],settings['beta_m'],gradient=False)
            if risk<best:
                best=risk;retained.append({'points':point.detach().cpu().clone(),'tag':tag,'step':step,'true_bank_hinge':risk,'smooth_bank_risk':smooth,**row})
    retain(target,'target_feasible_incumbent',-2)
    restored,scale=restore_feasible(source,w.detach(),target,D,B_geo,reference_current,lower,upper,tolerance)
    retain(restored,f'initial_exact_restoration_scale_{scale:.8f}',-1)
    for step in range(settings['steps']+1):
        start=time.perf_counter()
        p0,p1,row=geometry_record(source,w,target,D,B_geo,reference_current,lower,upper)
        smooth,risk,gradient=bank_value_gradient(bank,w,settings['beta_h'],settings['beta_m'],gradient=step<settings['steps'])
        violation=row['total_length']-B_geo
        if violation<=tolerance and row['current_invalid_mass']==0 and row['box_violation_mass']==0 and risk<best:
            best=risk;retained.append({'points':w.detach().cpu().clone(),'tag':'primal_dual_exact_feasible','step':step,'true_bank_hinge':risk,'smooth_bank_risk':smooth,**row})
        history.append({'step':step,'smooth_bank_risk':smooth,'true_bank_hinge':risk,'nu':nu,'rho_aug':rho,'g':violation,**row,'seconds':time.perf_counter()-start})
        if step%10==0:print('step',step,'risk',risk,'total_length',row['total_length'],'B_geo',B_geo,'nu',nu,'rho',rho,flush=True)
        if step==settings['steps']:break
        g=p0.differentiable_distance(source.double(),w.double())+p1.differentiable_distance(w.double(),target.double())-B_geo
        penalty=nu*g+rho/2*torch.relu(g).square()
        geometry_gradient=torch.autograd.grad(penalty,w)[0]
        if not torch.isfinite(gradient+geometry_gradient).all():raise ValueError('nonfinite primal gradient')
        reference=w.detach().clone()
        optimizer.zero_grad();w.grad=gradient+geometry_gradient;optimizer.step()
        with torch.no_grad():w.copy_(project_basic(w,reference,current,lower,upper))
        if (step+1)%settings['dual_interval']==0:nu=max(0.,nu+rho*violation)
        if (step+1)%settings['stall_interval']==0:
            current_violation=max(0.,violation)
            if current_violation>tolerance and current_violation>=.9*previous_violation:
                rho*=2
            previous_violation=current_violation
    restored,scale=restore_feasible(source,w.detach(),target,D,B_geo,reference_current,lower,upper,tolerance)
    retain(restored,f'final_exact_restoration_scale_{scale:.8f}',settings['steps']+1)
    return retained,history

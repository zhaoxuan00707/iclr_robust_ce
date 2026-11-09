"""Primal-dual particle solver with exact unregularized OT feasibility."""
import time
import torch
import torch.nn.functional as F
from .geodesic_excess import excess,project_basic


def bank_value_gradient(bank,points,beta_h=50.,beta_m=100.,chunk=256,gradient=True):
    """Stream all models; outer softmax weights follow population reduction."""
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


def geometry_record(source,w,target,D,kappa,current,lower,upper):
    p0,p1,E=excess(source,w,target,D)
    with torch.no_grad():
        probability=torch.sigmoid(current(w))
        invalid=float((probability<.5).double().mean())
        box=float(((w<lower-1e-7)|(w>upper+1e-7)).any(1).double().mean())
    record={'d0':p0.distance,'d1':p1.distance,'D':D,'E_geo':E,'kappa':kappa,'slack':kappa-E,
        'lambda_W':p0.distance/(p0.distance+p1.distance),'current_invalid_mass':invalid,
        'below_0.7_mass':float((probability<.7).double().mean()),'box_violation_mass':box,
        'marginal_residual':max(p0.marginal_residual,p1.marginal_residual)}
    return p0,p1,record


def restore_feasible(source,w,target,D,kappa,current,lower,upper,tolerance,iterations=12):
    """Numerical restoration towards a feasible target, testing full excess.

    This is an initialization/final-candidate diagnostic, not a displacement
    constraint and not a replacement for primal-dual optimization.
    """
    _,_,initial=geometry_record(source,w,target,D,kappa,current,lower,upper)
    if initial['E_geo']<=kappa+tolerance and initial['current_invalid_mass']==0 and initial['box_violation_mass']==0:
        return w.detach().clone(),1.
    if kappa==0:return target.detach().clone(),0.
    lo,hi=0.,1.;best=target.detach().clone()
    for _ in range(iterations):
        scale=(lo+hi)/2
        candidate=target+scale*(w-target)
        _,_,row=geometry_record(source,candidate,target,D,kappa,current,lower,upper)
        if row['E_geo']<=kappa+tolerance and row['current_invalid_mass']==0 and row['box_violation_mass']==0:
            best=candidate.detach().clone();lo=scale
        else:hi=scale
    return best,lo


def solve_control(bank,current,source,target,initial,D,kappa,lower,upper,settings):
    """Return feasible candidate checkpoints and complete primal/dual trace."""
    tolerance=settings['feasibility_relative_tolerance']*max(1.,D)
    w=project_basic(initial,target,current,lower,upper).detach().requires_grad_(True)
    optimizer=torch.optim.Adam([w],lr=settings['learning_rate'])
    nu=0.;rho=settings['rho_initial'];history=[];retained=[];best=float('inf');previous_violation=float('inf')
    def retain(point,tag,step):
        nonlocal best
        _,_,row=geometry_record(source,point,target,D,kappa,current,lower,upper)
        if row['E_geo']<=kappa+tolerance and row['current_invalid_mass']==0 and row['box_violation_mass']==0:
            smooth,risk,_=bank_value_gradient(bank,point,settings['beta_h'],settings['beta_m'],gradient=False)
            if risk<best:
                best=risk;retained.append({'points':point.detach().cpu().clone(),'tag':tag,'step':step,'true_bank_hinge':risk,'smooth_bank_risk':smooth,**row})
    retain(target,'target_feasible_incumbent',-2)
    restored,scale=restore_feasible(source,w.detach(),target,D,kappa,current,lower,upper,tolerance)
    retain(restored,f'initial_exact_restoration_scale_{scale:.8f}',-1)
    for step in range(settings['steps']+1):
        start=time.perf_counter()
        p0,p1,row=geometry_record(source,w,target,D,kappa,current,lower,upper)
        smooth,risk,gradient=bank_value_gradient(bank,w,settings['beta_h'],settings['beta_m'],gradient=step<settings['steps'])
        violation=row['E_geo']-kappa
        if violation<=tolerance and row['current_invalid_mass']==0 and row['box_violation_mass']==0 and risk<best:
            best=risk;retained.append({'points':w.detach().cpu().clone(),'tag':'primal_dual_exact_feasible','step':step,'true_bank_hinge':risk,'smooth_bank_risk':smooth,**row})
        history.append({'step':step,'smooth_bank_risk':smooth,'true_bank_hinge':risk,'nu':nu,'rho_aug':rho,'g':violation,**row,'seconds':time.perf_counter()-start})
        if step%10==0:print('step',step,'risk',risk,'excess',row['E_geo'],'kappa',kappa,'nu',nu,'rho',rho,flush=True)
        if step==settings['steps']:break
        g=p0.differentiable_distance(source.double(),w.double())+p1.differentiable_distance(w.double(),target.double())-D-kappa
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
    restored,scale=restore_feasible(source,w.detach(),target,D,kappa,current,lower,upper,tolerance)
    retain(restored,f'final_exact_restoration_scale_{scale:.8f}',settings['steps']+1)
    return retained,history

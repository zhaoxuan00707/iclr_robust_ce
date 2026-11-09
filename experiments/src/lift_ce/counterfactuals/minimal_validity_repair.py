"""Continuous geometry-only WRO and target-directed first-valid restoration.

No current-model information enters the raw support optimizer. Restoration is
performed only after raw controls are frozen, and only on invalid particles.
"""
import copy
import numpy as np
import torch
from .geodesic_excess import exact_w2
from .single_budget_solver import bank_value_gradient


def segment_pieces(model, start, end):
    """Enumerate a Linear/ReLU logit on t in [0,1] using float64 arithmetic."""
    pieces=[(0.,1.,np.asarray(end,dtype=float)-np.asarray(start,dtype=float),np.asarray(start,dtype=float))]
    for layer in list(model.features)+[model.classifier]:
        if isinstance(layer,torch.nn.Linear):
            w=layer.weight.detach().cpu().double().numpy();b=layer.bias.detach().cpu().double().numpy()
            pieces=[(lo,hi,w@a,w@c+b) for lo,hi,a,c in pieces]
        elif isinstance(layer,torch.nn.ReLU):
            new=[]
            for lo,hi,a,c in pieces:
                active=np.abs(a)>1e-15;roots=-c[active]/a[active]
                cuts=np.unique(np.r_[lo,roots[(roots>lo+1e-12)&(roots<hi-1e-12)],hi])
                for left,right in zip(cuts[:-1],cuts[1:]):
                    positive=a*((left+right)/2)+c>0
                    new.append((left,right,a*positive,c*positive))
            pieces=new
        else:raise ValueError('first-crossing enumeration requires a Linear/ReLU classifier')
    return [(lo,hi,float(a.item()),float(c.item())) for lo,hi,a,c in pieces]


def valid_intervals(model,start,end,tau=.5):
    """All numerically favorable intervals; does not assume score monotonicity."""
    threshold=float(np.log(tau/(1-tau)));intervals=[]
    for lo,hi,a,c in segment_pieces(model,start,end):
        c-=threshold
        if abs(a)<1e-15:
            if c>=0:intervals.append((lo,hi))
        elif a>0:
            left=max(lo,-c/a)
            if left<=hi:intervals.append((left,hi))
        else:
            right=min(hi,-c/a)
            if lo<=right:intervals.append((lo,right))
    merged=[]
    for lo,hi in intervals:
        if merged and lo<=merged[-1][1]+1e-12:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
        else:merged.append((lo,hi))
    return merged


def validity(model,points,weights=None,tau=.5):
    with torch.no_grad():scores=torch.sigmoid(model(points.cpu().float())).double()
    weights=torch.full((len(points),),1/len(points),dtype=torch.float64) if weights is None else weights.cpu().double()
    invalid=scores<tau;depth=torch.relu(tau-scores)
    return {'invalid_support_count':int(invalid.sum()),'invalid_mass':float((invalid*weights).sum()),
            'current_validity':float(((~invalid)*weights).sum()),'minimum_f0_score':float(scores.min()),
            'mean_violation_depth':float((depth*weights).sum()),
            'conditional_mean_violation_depth':float((depth*weights).sum()/(invalid*weights).sum()) if invalid.any() else 0.}


def minimal_restore(model,raw,anchors,weights=None,tau=.5,alpha_tolerance=1e-7):
    """Restore each invalid point to the first favorable segment interval.

    Valid raw points remain bitwise unchanged. Invalid anchors cause explicit
    failure. Float64 affine pieces locate the first crossing; reference float32
    predictions refine the crossing without skipping a valid interval. This is
    a numerical root search, not interval certification.
    """
    model=copy.deepcopy(model).cpu().eval().requires_grad_(False)
    raw=raw.detach().cpu().float();anchors=anchors.detach().cpu().float()
    if raw.shape!=anchors.shape:raise ValueError('one corresponding anchor per support required')
    result=raw.clone();rows=[]
    with torch.no_grad():s=torch.sigmoid(model(raw));end=torch.sigmoid(model(anchors))
    bad_anchor=torch.flatnonzero(end<tau) if hasattr(torch,'flatnonzero') else torch.nonzero(end<tau).flatten()
    if len(bad_anchor):raise ValueError(f'invalid corresponding anchors: {bad_anchor.tolist()}; no silent anchor switch')
    mass=torch.full((len(raw),),1/len(raw),dtype=torch.float64) if weights is None else weights.double().cpu()
    for j in range(len(raw)):
        alpha=0.;root=0.;lower=0.;upper=0.;interval_count=0
        if s[j]<tau:
            intervals=valid_intervals(model,raw[j].numpy(),anchors[j].numpy(),tau)
            interval_count=len(intervals);found=False
            def point(a):
                return ((1-a)*raw[j].double()+a*anchors[j].double()).float()
            def passes(a):
                candidate=raw.clone();candidate[j]=point(a)
                with torch.no_grad():return bool(torch.sigmoid(model(candidate))[j]>=tau)
            for left,right in intervals:
                root=float(left)
                # Find a reference-float valid point immediately after this root,
                # then bisect this FIRST feasible interval only.
                candidates=[left]+[min(right,left+h) for h in [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4]]+[(left+right)/2,right]
                hits=[a for a in sorted(set(candidates)) if passes(a)]
                if not hits:continue
                upper=hits[0];lower=max(0.,left-1e-5)
                if passes(lower):lower=0.
                for _ in range(50):
                    if upper-lower<=alpha_tolerance:break
                    middle=(lower+upper)/2
                    if passes(middle):upper=middle
                    else:lower=middle
                alpha=upper;result[j]=point(alpha);found=True;break
            if not found:raise ValueError(f'no representable first-valid segment point for support {j}')
        rows.append({'support_index':j,'raw_score':float(s[j]),'anchor_score':float(end[j]),'raw_invalid':bool(s[j]<tau),
                     'mass':float(mass[j]),'alpha':alpha,'float64_first_crossing':root,'alpha_lower':lower,'alpha_upper':upper,
                     'valid_intervals':interval_count,'displacement':float(torch.linalg.vector_norm(result[j]-raw[j]))})
    with torch.no_grad():after=torch.sigmoid(model(result))
    if (after<tau).any():raise ValueError('reference validity check failed after restoration')
    torch.testing.assert_close(result[s>=tau],raw[s>=tau],atol=0,rtol=0)
    for j,row in enumerate(rows):row['repaired_score']=float(after[j])
    return result,rows


def geometry(source,points,target,D,budget,weights=None):
    p0=exact_w2(source,points,b=weights);p1=exact_w2(points,target,a=weights)
    return p0,p1,{'D':D,'B_geo':budget,'d0':p0.distance,'d1':p1.distance,
                 'total_length':p0.distance+p1.distance,'slack':budget-p0.distance-p1.distance,
                 'marginal_residual':max(p0.marginal_residual,p1.marginal_residual)}


def solve_raw(bank,source,target,initial,D,budget,lower,upper,settings,on_step=None):
    """Historical continuous primal/dual machinery with geometry + box ONLY."""
    tolerance=settings['feasibility_relative_tolerance']*max(1.,D)
    points=initial.clamp(lower,upper).detach().requires_grad_(True)
    optimizer=torch.optim.Adam([points],lr=settings['learning_rate'])
    nu=0.;rho=settings['rho_initial'];previous=float('inf');best=float('inf');retained=[];trace=[]
    def retain(point,tag,step):
        nonlocal best
        _,_,row=geometry(source,point,target,D,budget)
        if row['slack'] < -tolerance:return
        _,risk,_=bank_value_gradient(bank,point,settings['beta_h'],settings['beta_m'],gradient=False)
        if risk<best:
            best=risk;retained.append({'points':point.detach().cpu().clone(),'true_bank_hinge':risk,'tag':tag,'step':step,**row})
    retain(target,'target',-1)
    for step in range(settings['steps']+1):
        p0,p1,row=geometry(source,points,target,D,budget)
        smooth,risk,gradient=bank_value_gradient(bank,points,settings['beta_h'],settings['beta_m'],gradient=step<settings['steps'])
        violation=-row['slack']
        if violation<=tolerance and risk<best:
            best=risk;retained.append({'points':points.detach().cpu().clone(),'true_bank_hinge':risk,'tag':'raw_geometry_feasible','step':step,**row})
        record={'step':step,'true_bank_hinge':risk,'smooth_bank_risk':smooth,'nu':nu,'rho_aug':rho,**row}
        trace.append(record)
        if on_step:on_step(record)
        if step==settings['steps']:break
        constraint=p0.differentiable_distance(source.double(),points.double())+p1.differentiable_distance(points.double(),target.double())-budget
        penalty=nu*constraint+rho/2*torch.relu(constraint).square()
        grad=torch.autograd.grad(penalty,points)[0]
        if not torch.isfinite(gradient+grad).all():raise ValueError('nonfinite raw optimizer gradient')
        optimizer.zero_grad();points.grad=gradient+grad;optimizer.step()
        with torch.no_grad():points.clamp_(lower,upper)
        if (step+1)%settings['dual_interval']==0:nu=max(0.,nu+rho*violation)
        if (step+1)%settings['stall_interval']==0:
            if violation>tolerance and violation>=.9*previous:rho*=2
            previous=max(0.,violation)
    # Same numerical target-directed geometry restoration, without f0 checks.
    # Exact verification, not an assumed monotonic feasibility certificate.
    _,_,last_geometry=geometry(source,points.detach(),target,D,budget)
    if last_geometry['slack']>=-tolerance:
        retain(points.detach(),'final_exact_feasible',settings['steps']+1)
        return retained,trace
    lo,hi=0.,1.;candidate=target.clone()
    for _ in range(settings['restoration_steps']):
        scale=(lo+hi)/2;trial=target+scale*(points.detach()-target)
        _,_,row=geometry(source,trial,target,D,budget)
        if row['slack']>=-tolerance:candidate=trial;lo=scale
        else:hi=scale
    retain(candidate,'final_geometry_restoration',settings['steps']+1)
    return retained,trace


def restore_with_budget(model,source,raw,target,D,budget,weights=None,tolerance=1e-6):
    """First minimal repair, then a bounded joint scalar search if necessary.

    Valid raw supports stay fixed throughout. The fallback moves only initially
    invalid supports along their own anchor segments, checks all constraints,
    and minimizes displacement within its searched feasible bracket. Failure
    means no feasible repair found by this declared numerical search, not a
    proof that every continuous repair is impossible.
    """
    repaired,rows=minimal_restore(model,raw,target,weights)
    _,_,initial=geometry(source,repaired,target,D,budget,weights)
    info={'initial_minimal_repair_geometry':initial,'fallback_used':False,'attempts':[]}
    if initial['slack']>=-tolerance:return repaired,rows,info
    info['fallback_used']=True
    raw=raw.cpu();target=target.cpu();invalid=torch.tensor([r['raw_invalid'] for r in rows])
    alphas=torch.tensor([r['alpha'] for r in rows],dtype=torch.float64)
    def trial(t):
        a=alphas+t*(1-alphas);a[~invalid]=0
        w=((1-a[:,None])*raw.double()+a[:,None]*target.double()).float()
        w[~invalid]=raw[~invalid]
        validity_row=validity(model,w,weights)
        if validity_row['invalid_support_count']:return w,a,None
        _,_,geom=geometry(source,w,target,D,budget,weights)
        return w,a,geom
    previous=0.;chosen=None
    for t in sorted(set(np.r_[np.geomspace(1e-6,1.,16),np.linspace(0.,1.,17)[1:]].tolist())):
        w,a,geom=trial(t)
        info['attempts'].append({'t':t,'geometry':geom})
        if geom is not None and geom['slack']>=-tolerance:
            lo,hi=previous,t;chosen=(w,a,geom)
            for _ in range(20):
                mid=(lo+hi)/2;w,a,geom=trial(mid)
                if geom is not None and geom['slack']>=-tolerance:hi=mid;chosen=(w,a,geom)
                else:lo=mid
            break
        previous=t
    if chosen is None:
        info['status']='infeasible_under_declared_joint_repair_search'
        return None,rows,info
    result,alphas,geom=chosen
    for i,row in enumerate(rows):
        row['minimal_alpha']=row['alpha'];row['alpha']=float(alphas[i]);row['joint_budget_repair']=True
        row['displacement']=float(torch.linalg.vector_norm(result[i]-raw[i]))
        with torch.no_grad():row['repaired_score']=float(torch.sigmoid(model(result[i:i+1]))[0])
    info['status']='joint_repair_feasible';info['final_geometry']=geom
    return result,rows,info

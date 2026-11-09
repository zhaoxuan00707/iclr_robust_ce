# Frozen pre-ablation optimizer function for regression verification.
def grace(model,x,y,base,p1,domain,r,beta,cfg):
    device=x.device; b=base.to(device); p1=p1.to(device); q=b.clone().requires_grad_(True)
    sync(); start=time.perf_counter()
    D=exact_w2(base,p1.cpu()).distance; budget=beta*D; dual=0.; previous=None; history=[]
    for step in range(cfg['solver']['steps']):
        f,nom,sens=mixed_surrogate(model,x,y,q,r,domain,**cfg['update'],temperature=cfg['smooth_temperature'],cat_temperature=cfg['categorical_temperature'],smoothing=cfg['categorical_smoothing'])
        g=differentiable_w2(b,q)[0]+differentiable_w2(q,p1)[0]-budget
        objective=float(f.detach()); change=math.inf if previous is None else abs(objective-previous)/max(1.,abs(previous))
        history.append(dict(step=step,objective=objective,nominal=float(nom.detach()),sensitivity=float(sens.detach()),violation=float(g.detach()),dual=dual,relative_change=change))
        if float(g.detach())<=cfg['solver']['feas_tol']*max(1.,D) and change<=cfg['solver']['obj_tol']: break
        loss=f+dual*g+.5*cfg['solver']['rho']*torch.relu(g).square()
        gradient=torch.autograd.grad(loss,q)[0]
        with torch.no_grad(): qnew=domain.project(q-cfg['solver']['primal_lr']*gradient)
        if not torch.isfinite(qnew).all(): raise RuntimeError('Nonfinite GRACE supports')
        q=qnew.detach().requires_grad_(True)
        newg=exact_w2(base,q.detach().cpu()).distance+exact_w2(q.detach().cpu(),p1.cpu()).distance-budget
        dual=max(0.,dual+cfg['solver']['rho']*newg)
        previous=objective
    z=q.detach().cpu(); plan=exact_w2(base,z); assignment=plan.target_index[plan.source_index.argsort()]; assigned=z[assignment]
    w1=plan.distance; w2=exact_w2(z,p1.cpu()).distance
    sync(); seconds=time.perf_counter()-start
    # Final diagnostics separately from generation runtime.
    f,nom,sens=mixed_surrogate(model,x,y,q,r,domain,**cfg['update'],temperature=cfg['smooth_temperature'],cat_temperature=cfg['categorical_temperature'],smoothing=cfg['categorical_smoothing'])
    return dict(q=z,cf=assigned,assignment=assignment,history=history,seconds=seconds,r=r,beta=beta,D=D,budget=budget,W_base_Q=w1,W_Q_P1=w2,feasible=w1+w2<=budget+cfg['solver']['feas_tol']*max(1.,D),nominal=float(nom.detach()),sensitivity=float(sens.detach()),surrogate=float(f.detach()),iterations=len(history))

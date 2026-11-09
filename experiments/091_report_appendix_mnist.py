"""Verify task100 fixed-MNIST ablations and save requested appendix artifacts."""
import argparse,os,sys,json,math,textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/src'))
os.environ.setdefault('MPLCONFIGDIR','/tmp/appendix100_mpl')
import pandas as pd,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lift_ce.experiments.stylemnist_grace_v1 import assets,load,sha,shifted_images
from lift_ce.counterfactuals.update_aware import surrogate
from lift_ce.counterfactuals.geodesic_excess import exact_w2
RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/appendix_mnist/20260925_v1');PARENT=Path('/home/xu.zhao/robust_ce_results/iclr_runs/mnist_low_margin_base/20260924_v1/p05001')

def main():
 global RUN
 parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,default=RUN);parser.add_argument("--results-root",type=Path,default=ROOT/"results");args=parser.parse_args();RUN=args.run
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 e1=args.results_root/'exp1_mnist_T_mismatch';e2=args.results_root/'exp2_mnist_no_geometry';e1.mkdir(exist_ok=True,parents=True);e2.mkdir(exist_ok=True,parents=True);(e2/'examples').mkdir(exist_ok=True)
 assert not (e1/'summary.md').exists() and not (e2/'summary.md').exists()
 c=json.loads((RUN/'resolved_config.json').read_text());d=load(RUN/'population.pt');m,ae,_=assets(c,'cuda');original={k:v.clone() for k,v in m.state_dict().items()};checks=[]
 for p,h in json.loads((RUN/'input_manifest.json').read_text()).items():
  if str(p)==str(ROOT/'tasks/100_extra_appendix.md') and sha(Path(p))!=h:assert sha(ROOT/'results/100_extra_appendix/task_at_mnist_launch.md')==h
  else:assert sha(Path(p))==h
 raw=pd.concat([pd.read_csv(RUN/f'future_worker{i}.csv') for i in range(8)],ignore_index=True);assert len(raw)==360 and set(raw.gpu)==set(range(8))
 oldmanifest=pd.read_csv(PARENT/'test_model_manifest.csv');imbase=d['test']['base'];con=load(PARENT/'test/r0.05_beta1.02/output.pt');unc=load(RUN/'unconstrained/test/r0.05_beta1.02/output.pt');images={'Base':imbase,'GRACE':con['images'],'Unconstrained':unc['images']};unrows=[]
 # Each trajectory is independently reconstructed with explicit autograd and out-of-place SGD arithmetic, not unroll.
 for scenario,rec in enumerate(oldmanifest.to_dict('records')):
  xx=shifted_images(d['x'],rec['family'],dict(zip(c['future']['families'],c['future']['strengths']))[rec['family']]*rec['severity'],rec['seed']).cuda();yy=d['y'].cuda();m.load_state_dict(original);m.requires_grad_(True);optimizer=torch.optim.SGD(m.parameters(),lr=.01)
  for step in range(1,41):
   optimizer.zero_grad();torch.nn.functional.binary_cross_entropy_with_logits(m(xx),yy).backward();
   with torch.no_grad():
    for parameter in m.parameters():parameter.copy_(parameter-.01*parameter.grad)
   if step not in [3,5,10,20,40]:continue
   saved=load(RUN/f'future_models/scenario{scenario:02d}_T{step}.pt');err=max(float((v.detach().cpu()-saved[k].cpu()).abs().max()) for k,v in m.state_dict().items());assert err<2e-6,err
   with torch.no_grad():
    train_loss=float(torch.nn.functional.binary_cross_entropy_with_logits(m(xx),yy));disp=sum(float((p-original[n]).square().sum()) for n,p in m.named_parameters())**.5
    for method in ['Base','GRACE']:
     row=raw[(raw.scenario==scenario)&(raw.T_eval==step)&(raw.method==method)].iloc[0];prob=m(images[method].cuda()).sigmoid();fv=float((prob>=.5).float().mean());assert abs(fv-row.FV)<1e-7;assert abs(train_loss-row.train_loss)<1e-5;assert abs(disp-row.parameter_displacement)<1e-5;assert sha(Path(row.checkpoint))==row.sha256
    if step==3:
     p=m(images['Unconstrained'].cuda()).sigmoid();unrows.append({'scenario':scenario,'family':rec['family'],'severity':rec['severity'],'FV':float((p>=.5).float().mean())})
   checks.append({'scenario':scenario,'T':step,'parameter_max_error':err,'score_loss_displacement_verified':True})
 raw.sort_values(['scenario','T_eval','method']).to_csv(e1/'raw.csv',index=False)
 summary=raw.groupby(['T_eval','method']).agg(FV=('FV','mean'),FV_scenario_std=('FV','std'),train_loss=('train_loss','mean'),parameter_displacement=('parameter_displacement','mean'),proximity=('proximity','first'),scenarios=('scenario','count')).reset_index();summary.to_csv(e1/'summary.csv',index=False)
 plt.rcParams.update({'pdf.fonttype':42,'font.size':11})
 for value,filename,ylabel in [('FV','fv_vs_Teval.pdf','Future Validity (%)'),('train_loss','train_loss_vs_Teval.pdf','Future training BCE'),('parameter_displacement','parameter_displacement_vs_Teval.pdf',r'$\|\theta_t-\theta_0\|_2$')]:
  fig,ax=plt.subplots(figsize=(5.3,3.5))
  for method in (['Base','GRACE'] if value=='FV' else ['GRACE']):
   v=summary[summary.method==method];ax.plot(v.T_eval,v[value]*(100 if value=='FV' else 1),marker='o',label=method)
  ax.axvline(3,color='gray',ls='--',lw=1);ax.set(xlabel='Evaluation horizon',ylabel=ylabel,xticks=[3,5,10,20,40]);ax.grid(alpha=.2)
  if value=='FV':ax.legend()
  fig.tight_layout();fig.savefig(e1/filename);plt.close(fig)
 pd.DataFrame(checks).to_csv(e1/'independent_checks.csv',index=False)
 fv=summary[summary.method=='GRACE'].set_index('T_eval').FV
 (e1/'summary.md').write_text(f'''# MNIST update-horizon mismatch\n\nFixed task016 near-boundary p05001 population, seed0, saved CNN/64D decoder,64 factuals, T_design3, r.05, beta1.02. No recalibration. One trajectory per each of36 old held-out scenarios; checkpoints3/5/10/20/40, all eight GPUs. Scenario-derived seeds81000/81001/81002 are inherited realizations, not independent experiment-seed runs. Original inputs unchanged. Raw `{RUN}`.\n\n{summary.to_markdown(index=False)}\n\nProximity is mean factual-to-CF pixel L2 and remains fixed. Future training loss is post-update BCE on the scenario training images. Parameter displacement includes all trainable CNN parameters, not buffers. Scenario std is descriptive across shifts, not a multi-seed standard error. Checkpoints atT3 replay prior states exactly;180 checkpoints independently reconstructed using independent explicit autograd/SGD arithmetic and scores verified.\n\nGRACE FV at the five horizons: {', '.join(f'T{t}: {100*v:.2f}%' for t,v in fv.items())}. Largest adjacent change: {100*fv.diff().abs().max():.2f} percentage points. Do not infer arbitrary-retraining robustness from this short-update mismatch diagnostic. Saved CNN retained per user instruction despite PDF MLP wording.\n''')
 m.load_state_dict(original);metrics=[]
 for label,payload,src in [('Constrained',con,PARENT/'test/r0.05_beta1.02'),('Unconstrained',unc,RUN/'unconstrained/test/r0.05_beta1.02')]:
  q=payload['Q'];deployed=payload['latents'];saved=json.loads((src/'metrics.json').read_text());p0=exact_w2(d['test']['base_z'],q);p1=exact_w2(q,d['target_z']);D=exact_w2(d['test']['base_z'],d['target_z']).distance
  torch.testing.assert_close(ae.decode(deployed.cuda()).detach().cpu(),payload['images'],atol=1e-5,rtol=0);assert abs(float((deployed-d['test']['base_z']).square().sum(1).mean().sqrt())-p0.distance)<1e-4
  f,n,s=surrogate(m,d['x'].cuda(),d['y'].cuda(),q.cuda(),.05,steps=3,lr=.01,decode=ae.decode,hinge_temperature=.05)
  for key,v in [('nominal',n),('sensitivity',s),('surrogate',f)]:assert abs(float(v.detach())-saved[key])<1e-5
  trace=pd.read_csv(src/'trace.csv');assert len(trace)==800
  if label=='Unconstrained':assert (trace.dual==0).all()
  scores=raw[(raw.T_eval==3)&(raw.method=='GRACE')].FV if label=='Constrained' else pd.Series([r['FV'] for r in unrows]);row={'method':label,'FV':float(scores.mean()),'proximity':float((payload['images']-d['test']['factual']).flatten(1).norm(dim=1).mean()),'nominal':float(n.detach()),'sensitivity':float(s.detach()),'surrogate':float(f.detach()),'W2_Base_Q':p0.distance,'W2_Q_P1':p1.distance,'total_detour':p0.distance+p1.distance,'original_budget':1.02*D,'within_original_budget':p0.distance+p1.distance<=1.02*D+1e-5*max(1,D),'current_validity':float((m(payload['images'].cuda()).sigmoid()>=.5).float().mean()),'updates':800};metrics.append(row)
 out=pd.DataFrame(metrics);out.to_csv(e2/'metrics.csv',index=False);pd.DataFrame(unrows).to_csv(e2/'unconstrained_per_scenario.csv',index=False)
 fig,axes=plt.subplots(4,4,figsize=(7.8,7.8));ids=[28,30,11,3]
 for row,i in enumerate(ids):
  for j,(label,ims) in enumerate([('Factual',d['test']['factual']),('Base',imbase),('Constrained',con['images']),('Unconstrained',unc['images'])]):axes[row,j].imshow(ims[i].squeeze(),cmap='gray',vmin=0,vmax=1);axes[row,j].axis('off');axes[row,j].set_title(label if row==0 else '')
 fig.tight_layout();fig.savefig(e2/'examples/comparison.pdf');fig.savefig(e2/'examples/comparison.png',dpi=150);plt.close(fig)
 (e2/'summary.md').write_text(f'''# MNIST without the geometry constraint\n\nSame saved CNN, near-boundary Base latent codes,64 supports, T3 eta.01, smooth.05, r.05, primal20 and800 updates; no added regularizer. Reused completed constrained output. Only augmented geometry terms and dual update disabled; geometry remains a diagnostic and one-to-one assignment is retained. Latent domain is R64 with identity projection. Seed0; new unconstrained solve on GPU0 after the eight-GPU horizon stage.\n\n{out.to_markdown(index=False)}\n\nFV uses the same36 T3 scenarios. Proximity is factual-to-decoded-CF pixel L2; geometry is latent Euclidean W2. These are distinct units. Both outputs independently decoded and their smooth objective recomputed. Final supports/assignment preserved; no validity repair, early stopping or convergence claim. Four qualitative factuals match the existing four-style illustration; no output-based reselection.\n\nRaw: `{RUN}`. Original constrained: `{PARENT}/test/r0.05_beta1.02`.\n''')
 print(summary.to_string(index=False));print(out.to_string(index=False))
if __name__=='__main__':main()

"""Independent persisted-output verification and publication artifacts for FFHQ."""
import argparse,json,sys,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT/'experiments/src'))
from lift_ce.experiments.ffhq_grace_v1 import load,sha,dump,features,logits,Generator,SmileClassifier,exact_w2

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
 torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.16);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 c=json.loads((out/'resolved_config.json').read_text());selected=json.loads((out/'selected_configuration.json').read_text());frozen=json.loads((out/'frozen.json').read_text());images=load(out/'frozen.pt')
 assert sha(out/'frozen.pt')==frozen['sha256'] and sha(out/'selected_configuration.json')==frozen['selected_sha256'] and sha(out/'attack_bank_manifest.csv')==frozen['bank_sha256']
 for path,digest in json.loads((out/'model_manifest.json').read_text())['hashes'].items():assert sha(path)==digest
 jobs=[json.loads(p.read_text()) for p in (out/'jobs').glob('*/complete.json')];assert len(jobs)==75;assert {int(j['physical_gpu']) for j in jobs}==set(range(8));assert all(j['start_utc']>frozen['frozen_utc'] for j in jobs if j['phase']=='future')
 calibration=pd.read_csv(out/'calibration_results.csv');assert len(calibration)==8;best=min(calibration.to_dict('records'),key=lambda v:(round(v['calibration_max_mean_hinge'],6),v['pixel_L2_cost'],v['r'],v['beta_geo']));assert best['r']==selected['r_star'] and best['beta_geo']==selected['beta_geo_star']
 facts=load(out/'test_factuals.pt');refs=load(out/'references.pt');pop=load(out/'test_population.pt');D=exact_w2(pop['base_w'],refs['w']).distance;geom=pd.read_csv(out/'geometry_feasibility.csv')
 for row in geom.itertuples():
  q=images[f'GRACE_beta{row.beta_geo:.2f}']['points'];length=exact_w2(pop['base_w'],q).distance+exact_w2(q,refs['w']).distance;assert abs(length-row.total_length)<1e-7 and length<=row.beta_geo*D+1e-6*max(1,D)
 monotone=bool((geom.sort_values('beta_geo').Phi_hat_r.diff().dropna()<=1e-6).all())
 train=load(out/'training.pt');bank=pd.read_csv(out/'attack_bank_manifest.csv');assert len(bank)==16
 for row in bank.itertuples():
  assert sha(row.checkpoint)==row.checkpoint_sha256 and sha(row.perturbation)==row.perturbation_sha256
  delta=load(Path(row.perturbation));upper=float((delta['delta'].double().square().sum()/len(train['x'])).sqrt());assert upper<=row.r+1e-6 and row.W_data<=upper+1e-5
 assert not set(np.load(out/'calibration_factual_ids.npy'))&set(np.load(out/'test_factual_ids.npy'))
 clf=SmileClassifier('cuda');g=Generator('cuda');fresh_features={};max_image_error=0.
 for name,item in images.items():
  with torch.no_grad():canonical=torch.cat([clf.canonical(g.decode(q.cuda())).cpu() for q in item['points'].split(1)])
  max_image_error=max(max_image_error,float((canonical-item['x']).abs().max()));assert torch.allclose(canonical,item['x'],atol=1e-5,rtol=1e-5)
  fresh_features[name]=features(clf,canonical,'cuda').cuda()
 del g;torch.cuda.empty_cache()
 files=sorted((out/'jobs').glob('future_*/metrics.csv'));assert len(files)==30;fv=pd.concat([pd.read_csv(p) for p in files],ignore_index=True);assert len(fv)==210
 error=0.
 for file in files:
  table=pd.read_csv(file);meta=json.loads((file.parent/'model.json').read_text());assert sha(meta['checkpoint'])==meta['checkpoint_sha256'];projection=load(Path(meta['checkpoint']))['projection'].cuda()
  with torch.no_grad():
   for row in table.itertuples():
    probabilities=clf.logits_features(fresh_features[row.method],projection).softmax(1)[:,0];value=float((probabilities>=.5).double().mean());error=max(error,abs(value-row.future_validity))
  print('reviewed',file.parent.name,flush=True)
 assert error<1e-12;fv.to_csv(out/'future_validity.csv',index=False)
 risks=pd.read_csv(out/'finite_bank_risk.csv');states=torch.stack([load(Path(row.checkpoint))['projection'] for row in bank.itertuples() if row.W_data<=selected['r_star']+1e-6]).cuda();risk_error=0.
 with torch.no_grad():
  for row in risks.itertuples():
   value=float(torch.relu(.5-logits(clf,fresh_features[row.method],states).softmax(-1)[:,:,0]).double().mean(1).max());risk_error=max(risk_error,abs(value-row.Phi_hat_r))
 assert risk_error<1e-6
 quality=pd.read_csv(out/'image_quality.csv')
 for row in quality.itertuples():assert abs(float((images[row.method]['x']-facts['x']).flatten(1).norm(dim=1).mean())-row.pixel_L2_224)<1e-6
 timing=pd.DataFrame([json.loads(p.read_text()) for p in (out/'jobs').glob('*/launch_timing.json')]);timing.to_csv(out/'runtime.csv',index=False);clock=json.loads((out/'construction_clock.json').read_text());construction=timing[timing.phase!='future'];runtime={'construction_wall_seconds':clock['wall_seconds'],'seconds_per_test_CF':clock['wall_seconds']/len(facts['w']),'construction_GPU_hours':float(construction.wall_seconds.sum()/3600),'future_GPU_hours':float(timing[timing.phase=='future'].wall_seconds.sum()/3600),'servers':1,'gpu_ids':list(range(8))};dump(out/'runtime_summary.json',runtime)
 nominal=fv[fv.severity==1].groupby('method',as_index=False).agg(Future_Validity=('future_validity','mean'),Worst_Future_Validity=('future_validity','min'));main=nominal.merge(quality,on='method').merge(risks,on='method');main['Identity_Score']=np.nan;main['shared_construction_seconds']=clock['wall_seconds'];main.to_csv(out/'main_results.csv',index=False)
 ablation=main[main.method.str.startswith('GRACE_beta')].copy();ablation['beta_geo']=ablation.method.str.replace('GRACE_beta','').astype(float);ablation.merge(geom,on='beta_geo',suffixes=('','_geometry')).to_csv(out/'geometry_ablation.csv',index=False)
 severity=fv.groupby(['method','family','severity'],as_index=False).future_validity.mean();severity.to_csv(out/'shift_severity.csv',index=False)
 fig,axes=plt.subplots(1,3,figsize=(10,3),sharey=True)
 for ax,family in zip(axes,c['future']['families']):
  for method in ['Base','LeapFactual','GRACE_selected']:
   d=severity[(severity.family==family)&(severity.method==method)];ax.plot(d.severity,100*d.future_validity,'o-',label=method)
  ax.set(title=family,xlabel='External severity (fixed r, β)',ylim=(-3,103));ax.grid(alpha=.25)
 axes[0].set_ylabel('Future Validity (%)');axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'figures/ffhq_shift_severity.pdf');fig.savefig(out/'figures/ffhq_shift_severity.png',dpi=180);plt.close(fig)
 names=['factual','Base']+[f'GRACE_beta{b:.2f}' for b in c['beta_grid']];arrays=[np.load(out/'qualitative_examples'/f'{name}.npy') for name in names];fig,axes=plt.subplots(4,6,figsize=(12,8))
 for i in range(4):
  for j,name in enumerate(names):
   axes[i,j].imshow(arrays[j][i].transpose(1,2,0));axes[i,j].axis('off')
   if i==0:axes[i,j].set_title(name.replace('GRACE_beta','β=').title() if j<2 else name.replace('GRACE_beta','β='),fontsize=10)
 fig.tight_layout(pad=.2);fig.savefig(out/'figures/ffhq_qualitative.pdf');fig.savefig(out/'figures/ffhq_qualitative.png',dpi=160);plt.close(fig)
 pd.DataFrame({'row':range(4),'factual_id':np.load(out/'factual_ids.npy')[:4],'selection':'first four frozen test factuals, before future evaluation'}).to_csv(out/'qualitative_examples/row_ids.csv',index=False)
 review={'status':'passed','jobs':75,'attack_models':16,'future_models':30,'max_decoded_image_error':max_image_error,'max_future_validity_error':error,'max_bank_risk_error':risk_error,'all_geometry_feasible':True,'nonincreasing_bank_risk':monotone,'configuration_selection_recomputed':True,'all_future_after_freeze':True,'gpu_ids':list(range(8))};dump(out/'completion_review.json',review)
 concise=main[['method','Future_Validity','pixel_L2_224','pixel_L2_1024','LPIPS_squeeze','Phi_hat_r']]
 report=['# FFHQ-generated face GRACE','Completed with original pretrained StyleGAN3, CLIP and conditional flow. This is a synthetic-face, prompt-relative experiment; it is not a supervised real-FFHQ smile benchmark.',
 '## Task and asset reuse','The original classifier compares literal prompts "smiling face" (index0, target) and "face" (index1, source), with probability threshold0.5. Original StyleGAN3-r-ffhq-1024x1024.pkl synthesizes RGB1024 faces from W512 repeated over16 layers; there is no encoder or inversion. The original KarrasUnet1D flow checkpoint and progressive lift/land implementation generate Base CFs with10 iterations at0.8/0.8. LeapFactual comparator applies5 additional iterations at0.8/0.83. GRACE starts from Base, never the robust comparator. Exact paths and hashes are in model_manifest.json and ffhq_asset_inventory.md.',
 '## New fixed update and generated cohorts','256 generated training faces receive frozen original CLIP pseudo-labels. Separate64-image generated pools supply8 non-smiling calibration factuals and8 test factuals, selected in seeded order. Eight smiling training references are fixed before evaluation. All original generator/flow/CLIP backbone/text weights are frozen. Update trains the entire768×512 visual projection for5 epochs with fresh AdamW(lr1e-4,weight_decay0.01), batch32, fixed seed700000. This deliberately limited update family is newly introduced because the original zero-shot classifier has no supervised retraining protocol. It is not full-backbone adaptation. Failed Base CFs remain in the denominator.',
 '## Selected configuration',f"r_star={selected['r_star']}, beta_geo_star={selected['beta_geo_star']}. Selection uses only the8 calibration grid rows and four predeclared calibration models; the operating point is fixed for all future severities.",
 '## Main results at nominal external severity',concise.to_markdown(index=False),'Future Validity averages over3 shift families and2 realizations at severity1, with8 factuals. Pixel cost is distinct from distribution-level geometry. Identity score is unavailable: no compatible original face identity metric was found, and none was invented. SqueezeNet LPIPS, SSIM and PSNR follow the original image-quality metric families. No identity-preservation claim is made.',
 '## Geometry budget',ablation[['beta_geo','Future_Validity','pixel_L2_224','Phi_hat_r','LPIPS_squeeze']].to_markdown(index=False),f'All distances are exact equal-mass Euclidean W2 in unnormalized W512. Feasibility tolerance1e-6×max(1,D). The objective is max_model mean_CF hinge. Each candidate is cross-evaluated against all compatible shared-bank models; beta candidates are pooled after optimization. Nonincreasing retained bank risk={monotone}. This is finite-bank/local optimization, not exact WRO optimality. No current-validity constraint or repair is used.',
 '## Future shifts','New brightness, additive noise and Gaussian blur stresses alter canonical224 training images before the same fixed projection Update. CFs are frozen before all30 future models. External severity differs from internal Wasserstein radius. Brightness/blur replicas and severity0 contain deterministic duplicates; they do not provide independent replication or uncertainty estimates.',
 '## Image diagnostics and qualitative figure','The first four prespecified test factuals appear in every column. Native1024 returned images and source W codes are retained. The generator is deterministic; no encoder reconstruction error is applicable. Latent displacement, pixel costs, original LPIPS, SSIM and PSNR are saved. Visual inspection is descriptive and cannot establish identity preservation.',
 '## Runtime',json.dumps(runtime,indent=2),'End-to-end construction includes generated-data preparation, Base and compatible-baseline generation, shared bank, calibration, both-start/beta refinement and output freezing. It excludes original pretraining and future evaluation. Shared construction time is repeated in method rows for context and must not be summed. GPU-hours measure summed worker occupancy.',
 '## Checks, failures and limitations',json.dumps(review,indent=2),'Initial audit attempts using reference StyleGAN operators exceeded the memory cap. Original fused CUDA operators were built in an isolated environment; checkpointed synthesis was then verified exactly equal to original inference and latent gradients were finite. Audit failures produced no experiment results. The data adversary edits32 of256 canonical training images per restart and uses12 one-step surrogate iterations; each accepted model receives the complete fixed5-epoch projection update. These restrictions and8 test factuals limit generalization. The classifier and labels are both prompt-relative CLIP, and "face" is not an explicit non-smiling annotation. The fixed audit image visibly smiles while its smiling probability is0.45375; this observed semantic mismatch limits any literal non-smiling-to-smiling interpretation. All historical outputs were preserved; no manuscript claims were changed.',f'Raw run: `{out}`.']
 (out/'report.md').write_text('\n\n'.join(report)+'\n');dest=ROOT/'results/014_ffhq_grace_v1';dest.mkdir(exist_ok=True)
 for name in ['report.md','main_results.csv','geometry_ablation.csv','geometry_feasibility.csv','calibration_results.csv','selected_configuration.json','image_quality.csv','identity_metrics.csv','runtime_summary.json','shift_severity.csv','completion_review.json','resolved_config.json','dataset_manifest.json']:shutil.copyfile(out/name,dest/name)
 figdest=ROOT/'figures/ffhq_grace_v1';figdest.mkdir(exist_ok=True)
 for file in (out/'figures').glob('*'):shutil.copyfile(file,figdest/file.name)
 dump(out/'completed.json',review);print(concise.to_string(index=False),flush=True)
if __name__=='__main__':main()

"""Independent Galaxy checks, CSV tables, publication figures and final report."""
import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'experiments/src'))
from lift_ce.galaxy.assets import load_classifier,load_autoencoder,CLASS_NAMES
from lift_ce.counterfactuals.geodesic_excess import exact_w2

def load(p):return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,o):p.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();out=a.output
    torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    c=json.loads((out/'resolved_config.json').read_text());selection=json.loads((out/'selected_configuration.json').read_text());frozen=json.loads((out/'frozen.json').read_text())
    for path,digest in json.loads((out/'model_manifest.json').read_text())['hashes'].items():assert sha(path)==digest
    assert sha(out/'frozen_images.pt')==frozen['images_sha256']
    assert sha(out/'selected_configuration.json')==frozen['selected_configuration_sha256']
    assert sha(out/'attack_bank_manifest.csv')==frozen['bank_manifest_sha256']
    files=sorted((out/'jobs').glob('future_*/metrics.csv'));assert len(files)==20
    fv=pd.concat([pd.read_csv(p) for p in files],ignore_index=True);assert len(fv)==20*10*7
    assert fv.groupby(['task','method','family','severity','realization']).size().eq(1).all()
    fv.to_csv(out/'future_validity.csv',index=False)
    bank=pd.read_csv(out/'attack_bank_manifest.csv');assert len(bank)==16 and bank.feasible.all()
    for row in bank.itertuples():
        assert sha(row.checkpoint)==row.checkpoint_sha256 and sha(row.perturbation_path)==row.perturbation_sha256
        d=load(Path(row.perturbation_path));upper=float((d['delta'].double().square().sum()/3181).sqrt());assert upper<=row.search_radius+1e-6 and row.W_data<=upper+1e-5
    calibration=pd.read_csv(out/'calibration_results.csv');assert len(calibration)==80
    for t in range(10):
        rows=calibration[calibration.task==t].to_dict('records');best=min(rows,key=lambda x:(round(x['calibration_max_mean_hinge'],6),x['pixel_L2_cost'],x['r'],x['beta_geo']))
        assert selection['tasks'][str(t)]['r_star']==best['r'] and selection['tasks'][str(t)]['beta_geo_star']==best['beta_geo']
    geom=pd.read_csv(out/'geometry_feasibility.csv');assert len(geom)==40
    for row in geom.itertuples():
        spec=selection['tasks'][str(row.task)];pop=load(out/'tasks'/f'target{row.task:02d}'/'test_population.pt');q=load(out/'jobs'/f'opt_test_{row.task}_{spec["r_index"]}'/'selected.pt')[str(row.beta_geo)]['points']
        D=exact_w2(pop['base_z'],pop['reference_z']).distance;L=exact_w2(pop['base_z'],q).distance+exact_w2(q,pop['reference_z']).distance
        assert abs(L-row.total_length)<1e-7 and L<=row.beta_geo*D+1e-6*max(1,D)
    jobs=[json.loads(p.read_text()) for p in (out/'jobs').glob('*/complete.json')];assert len(jobs)==75
    assert {int(j['physical_gpu']) for j in jobs}==set(range(8))
    assert all(j['start_utc']>frozen['frozen_utc'] for j in jobs if j['phase']=='future')
    for item in jobs:
        resolved={'configuration':c,'experiment_seed':c['seed'],'model_update_seed':c['update']['seed'],'job':item['job'],'physical_gpu':item['physical_gpu']}
        parts=item['job'].split('_')
        if item['phase']=='attack':
            ri,restart=map(int,parts[1:]);resolved.update(r=c['r_grid'][ri],search_seed=c['seed']+1000+ri*100+restart,beta_geo=None,target_tasks=c['task_targets'])
        elif item['phase']=='optimize':
            resolved.update(split=parts[1],target=int(parts[2]),r=c['r_grid'][int(parts[3])],beta_geo=c['beta_grid'])
        elif item['phase']=='base':resolved.update(target=int(parts[1]),generator_seed=c['seed']+int(parts[1]))
        elif item['phase']=='future':resolved.update(family=parts[1],severity=c['future']['severity'][int(parts[2])],shift_seed=c['future']['seed_base']+int(parts[3]),selected_configuration=selection)
        write(out/'jobs'/item['job']/'job_resolved_config.json',resolved)
    timing=[json.loads(p.read_text()) for p in (out/'jobs').glob('*/launch_timing.json')];rt=pd.DataFrame(timing);rt.to_csv(out/'runtime.csv',index=False)
    constructor=rt[~rt.phase.isin(['prepare','future'])];clock=json.loads((out/'construction_clock.json').read_text())
    wall=clock['wall_seconds'];gpu_hours=float(constructor.wall_seconds.sum()/3600);future_hours=float(rt[rt.phase=='future'].wall_seconds.sum()/3600)
    runtime={'construction_wall_seconds':wall,'construction_wall_seconds_per_test_cf':wall/160,'construction_GPU_hours':gpu_hours,'future_GPU_hours':future_hours,'all_worker_GPU_hours':float(rt.wall_seconds.sum()/3600),'runtime_scope':'full grid and ten-task pipeline including compatible baseline; shared costs counted once; pretrained training and final future evaluation excluded from construction'}
    write(out/'runtime_summary.json',runtime)
    write(out/'execution_protocol_details.json',{'gradient_clip_norm':10.0,'cudnn_deterministic':True,'cudnn_benchmark':False,'gradient_accumulation_steps':1,'training_dtype':'float32','training_epochs':c['update']['epochs'],'all_trainable_VGG_parameters_updated':True,'search_batchnorm':'frozen eval mode','full_update_batchnorm':'train mode','original_base_classifier_training_seed':'not recovered from checkpoint metadata','factual_generation_failures':'retained in denominator, no dropping or duplication'})
    # Recompute all final future results from persisted checkpoints and frozen pixels.
    all_images=load(out/'frozen_images.pt');model=load_classifier('cuda:0').eval().requires_grad_(False)
    error=0.
    for file in files:
        rows=pd.read_csv(file);model.load_state_dict(load(Path(rows.iloc[0].checkpoint)))
        with torch.no_grad():
            for row in rows.itertuples():
                images=all_images[str(row.task)][row.method]
                logits=torch.cat([model(x.cuda()).cpu() for x in images.split(c['microbatch'])]);p=logits.softmax(1)[:,row.task]
                value=float((p>=.5).double().mean());error=max(error,abs(value-row.future_validity));assert float((logits.argmax(1)==row.task).double().mean())==row.target_argmax_validity
        print('reviewed',file.parent.name,flush=True)
    assert error<1e-12
    quality=pd.read_csv(out/'image_quality.csv');risks=pd.read_csv(out/'finite_bank_risk.csv')
    for row in quality.itertuples():
        pop=load(out/'tasks'/f'target{row.task:02d}'/'test_population.pt');x=all_images[str(row.task)][row.method]
        cost=float((x-pop['factual']).flatten(1).norm(dim=1).mean());assert abs(cost-row.pixel_L2_cost)<1e-6
    # Recompute shared-bank risk for every returned method, only feasible models per r.
    risk_lists={(t,name):[] for t,methods in all_images.items() for name in methods}
    for row in bank.itertuples():
        model.load_state_dict(load(Path(row.checkpoint)))
        with torch.no_grad():
            for task,methods in all_images.items():
                if row.W_data>selection['tasks'][task]['r_star']+1e-6:continue
                for name,x in methods.items():
                    p=torch.cat([model(q.cuda()).softmax(1)[:,int(task)].cpu() for q in x.split(c['microbatch'])]);risk_lists[(task,name)].append(float(torch.relu(.5-p).double().mean()))
    max_risk_error=0.
    for row in risks.itertuples():
        value=max(risk_lists[(str(row.task),row.method)]);max_risk_error=max(max_risk_error,abs(value-row.Phi_hat_r))
    assert max_risk_error<1e-6
    nominal=fv[fv.severity==1].groupby(['task','method'],as_index=False).agg(Future_Validity=('future_validity','mean'),Argmax_Validity=('target_argmax_validity','mean'),Worst_Future_Validity=('future_validity','min'))
    detailed=nominal.merge(quality,on=['task','method']).merge(risks,on=['task','method'])
    detailed.to_csv(out/'main_results_per_task.csv',index=False)
    main=detailed.groupby('method',as_index=False).agg(Future_Validity=('Future_Validity','mean'),Argmax_Validity=('Argmax_Validity','mean'),Cost=('pixel_L2_cost','mean'),Phi_hat_r=('Phi_hat_r','mean'))
    bq=pd.concat([pd.read_csv(out/'tasks'/f'target{t:02d}'/'base_quality.csv') for t in range(10)],ignore_index=True);bq.to_csv(out/'base_reconstruction_quality.csv',index=False)
    base_time=float(bq.base_seconds.sum());leap_time=base_time+float(bq.leapfactual_extra_seconds.sum())
    main['Construction_Seconds']=main.method.map(lambda n:base_time if n=='Base' else leap_time if n=='LeapFactual' else wall)
    main['Seconds_per_test_CF']=main.Construction_Seconds/160
    main['Runtime_Convention']=main.method.map(lambda n:'sum generator GPU seconds for both splits' if n in ['Base','LeapFactual'] else 'shared complete pipeline wall seconds; do not sum rows')
    main.to_csv(out/'main_results.csv',index=False)
    config_table=pd.DataFrame([{'Task':CLASS_NAMES[t],'target':t,**selection['tasks'][str(t)]} for t in range(10)]);config_table.to_csv(out/'selected_configuration_table.csv',index=False)
    ablation=detailed[detailed.method.str.startswith('GRACE_beta')].copy();ablation['beta_geo']=ablation.method.str.replace('GRACE_beta','').astype(float);ablation=ablation.merge(geom,on=['task','beta_geo'],suffixes=('','_geometry'))
    ablation['shared_construction_seconds']=wall;ablation.to_csv(out/'geometry_ablation_per_task.csv',index=False)
    agg=ablation.groupby('beta_geo',as_index=False).agg(Future_Validity=('Future_Validity','mean'),Cost=('pixel_L2_cost','mean'),Phi_hat_r=('Phi_hat_r','mean'));agg['shared_construction_seconds']=wall;agg.to_csv(out/'geometry_ablation.csv',index=False)
    severity=fv.groupby(['method','family','severity'],as_index=False).agg(Future_Validity=('future_validity','mean'),Argmax_Validity=('target_argmax_validity','mean'))
    severity.to_csv(out/'shift_severity.csv',index=False)
    severity_description=[]
    for family in c['future']['families']:
        subset=severity[(severity.family==family)&severity.method.isin(['Base','GRACE_selected'])]
        values=[]
        for name in ['Base','GRACE_selected']:
            ordered=subset[subset.method==name].sort_values('severity')
            values.append(f"{name}: {ordered.iloc[0].Future_Validity:.2%} at severity {ordered.iloc[0].severity:g}, {ordered.iloc[-1].Future_Validity:.2%} at severity {ordered.iloc[-1].severity:g}")
        severity_description.append(f"{family}: "+'; '.join(values)+'. These endpoints describe this fixed update protocol; intermediate levels are reported in shift_severity.csv.')
    shown=['Base','LeapFactual','GRACE_selected']
    fig,axes=plt.subplots(1,2,figsize=(7.5,3.0),sharey=True)
    for ax,family in zip(axes,c['future']['families']):
        for name in shown:
            d=severity[(severity.family==family)&(severity.method==name)];ax.plot(d.severity,d.Future_Validity,'o-',label=name,markersize=4)
        ax.set(title=family,xlabel='External severity (fixed selected r, β)',ylim=(-.03,1.03));ax.grid(alpha=.25)
    axes[0].set_ylabel('Future Validity (target probability ≥ 0.5)');axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'figures/galaxy_shift_severity.pdf');fig.savefig(out/'figures/galaxy_shift_severity.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(10,6,figsize=(9,14));pixel_rows=[];qual_arrays=[];ids=[]
    for t in range(10):
        pop=load(out/'tasks'/f'target{t:02d}'/'test_population.pt');names=['Factual','Base']+[f'GRACE_beta{b:.2f}' for b in c['beta_grid']]
        row=[pop['factual'][0]]+[all_images[str(t)][name][0] for name in names[1:]];qual_arrays.append(torch.stack(row).numpy())
        ids.append({'target':t,'factual_id':str(np.load(out/'tasks'/f'target{t:02d}'/'test_factual_ids.npy')[0]),'row':t,'selection':'first prespecified factual row for every target'})
        for j,(name,x) in enumerate(zip(names,row)):
            axes[t,j].imshow(x.permute(1,2,0).clamp(0,1));axes[t,j].set_xticks([]);axes[t,j].set_yticks([])
            if j==0:axes[t,j].set_ylabel(f'Target {t}',fontsize=8)
            if t==0:axes[t,j].set_title(name.replace('GRACE_beta','β='),fontsize=9)
    fig.tight_layout(pad=.3);fig.savefig(out/'figures/galaxy_qualitative.pdf');fig.savefig(out/'figures/galaxy_qualitative.png',dpi=150);plt.close(fig)
    np.save(out/'qualitative_examples/fixed_grid_pixels.npy',np.stack(qual_arrays));pd.DataFrame(ids).to_csv(out/'qualitative_examples/row_ids.csv',index=False)
    monotone={str(t):bool((np.diff(geom[geom.task==t].sort_values('beta_geo').Phi_hat_r)<=1e-6).all()) for t in range(10)}
    review={'status':'passed','complete_jobs':len(jobs),'servers':1,'gpu_ids':list(range(8)),'attack_models':len(bank),'future_models':len(files),'target_tasks':10,'calibration_grid_rows':len(calibration),'max_future_validity_recompute_error':error,'max_bank_risk_recompute_error':max_risk_error,'geometry_all_within_tolerance':True,'selected_configurations_recomputed':True,'future_after_freeze':True,'nested_bank_risk_monotonic_by_task':monotone,'raw_run':str(out)}
    write(out/'completion_review.json',review)
    checks=json.loads((out/'asset_checks.json').read_text())
    report=['# Galaxy GRACE: original LeapFactual multiclass assets','',
      'Completed on one server and eight GPUs. No historical runs were modified; no base classifier, encoder or decoder was retrained. This is a finite-bank/local-solver replication, not an exact WRO solution.',
      '## Asset reuse and task','Original VGG16_BN trained on the 20% Galaxy10 training folder, checkpoint epoch47-step624; WAE_MMD64 version8 last checkpoint (epoch299); original conditional flow MLP finetune weights and progressive LeapFactual generator. Exact paths/hashes: model_manifest.json and galaxy_asset_inventory.md. Ten target-label recourse tasks preserve the original ten-way classifier. Each has 16 calibration factuals, 16 test factuals and 16 target-class training references; no test outcomes select those cohorts.',
      '## Calibration-selected configuration',config_table.to_markdown(index=False),'For each target, r_star and beta_geo_star minimize worst calibration-model mean hinge, with the predeclared tie-break. All 80 grid rows are saved. These configurations remain fixed across every final severity.',
      '## Main result at nominal external severity',main.to_markdown(index=False),'Macro means over ten target tasks, two shift families and two realizations. Primary validity requires target softmax probability ≥0.5; target-argmax validity is separately reported, preserving the original multiclass diagnostic. Cost is factual-to-returned-image pixel L2. The full operating pipeline and calibration grid are shared across GRACE rows, so their runtimes must not be summed. Base/LeapFactual generator times are GPU-worker seconds; GRACE is complete construction wall time; these different runtime conventions are explicit rather than treated as a speedup comparison.',
      '## Geometry budget ablation',agg.to_markdown(index=False),f'Per-task nonincreasing retained shared-bank risk: {monotone}. Nested candidate pooling enforces the ordering among retained feasible candidates; it does not prove global optimality. All exact W2 checks use the declared relative tolerance 1e-6 × max(1,D). No current-model validity constraints or repair are used.',
      '## Representation and image quality',f"All geometry is unregularized empirical W2 in the same original64-dimensional Euclidean latent space. Classifiers always score decoded RGB images. Asset audit reconstruction MSE={checks['reconstruction_mse']:.6f}; classifier reconstruction decision agreement={checks['reconstruction_decision_agreement']:.2%}; audit classifier accuracy={checks['classifier_accuracy']:.2%}. These modest semantic checks limit interpretation despite small pixel error. Per-task Base round-trip validity, reconstruction MSE, pixel cost, latent displacement and PSNR are saved. The published qualitative panel uses the first frozen factual for each target, not outcomes. No quantitative morphology-validity claim is inferred from visual appearance alone.",
      '## External shifts','Exposure and additive imaging noise are a NEW predeclared evaluation suite, not an established LeapFactual robustness benchmark. Fixed updates train on shifted original training data; returned CF pixels are frozen. Exposure/noise severity is unrelated to internal r. Brightness replicas and severity0 are deterministic duplicates, not independent statistical replication. No uncertainty bands across those copies are claimed.',
      *severity_description,
      '## Compute and runtime',json.dumps(runtime,indent=2),'Construction runtime starts from pretrained assets and includes original Base/LeapFactual generation, bank construction, calibration models, all calibration solves, r/beta selection and final refinement/freezing. It excludes the preliminary audit and final future evaluation. GPU-hours are summed worker occupancy wall time, not a hardware utilization integral. Detailed per-job timing and commands are saved.',
      '## Failures, limitations and checks',f"All75 jobs and independent checks passed. Sixteen full-update attack models are persisted; each candidate is scored against every radius-feasible model in the shared bank. All final future values and finite-bank risks were recomputed from stored checkpoints/pixels (max errors {error:g}, {max_risk_error:g}). One-step minibatch attack hypergradients, only128 mutable training images per restart, and two-epoch complete updates restrict the adversary/model-update family. These approximations are explicit. Original checkpoint validation reused the old test folder; final future scenarios were untouched until freezing, but the pretrained classifier is not a new clean-split benchmark. Sixteen factuals per target and a single frozen f0 limit statistical generalization. GPU memory caps preserve coexistence with existing workloads; no jobs were killed.",
      f'Raw results: `{out}`. Tables, configuration, manifests and publication PDF/PNG figures are saved. No manuscript scientific claims were changed.']
    (out/'report.md').write_text('\n\n'.join(report)+'\n')
    result=ROOT/'results/011_galaxy_grace_v1';result.mkdir(exist_ok=True)
    for name in ['report.md','main_results.csv','main_results_per_task.csv','geometry_ablation.csv','geometry_feasibility.csv','selected_configuration.json','selected_configuration_table.csv','calibration_results.csv','image_quality.csv','shift_severity.csv','runtime_summary.json','completion_review.json','base_reconstruction_quality.csv','resolved_config.json']:
        shutil.copyfile(out/name,result/name)
    dest=ROOT/'figures/galaxy_grace_v1';dest.mkdir(exist_ok=True)
    for p in (out/'figures').glob('*'):shutil.copyfile(p,dest/p.name)
    write(out/'completed.json',{'status':'complete',**runtime})
    print(main.to_string(index=False),flush=True)
if __name__=='__main__':main()

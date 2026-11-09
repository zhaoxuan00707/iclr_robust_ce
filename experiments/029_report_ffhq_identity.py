"""Independent checks and matched FFHQ identity/robustness report."""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/src'))
from lift_ce.experiments.ffhq_grace_v1 import SmileClassifier, logits, records
from lift_ce.experiments.galaxy_grace_v1 import load, sha, dump
from lift_ce.counterfactuals.geodesic_excess import exact_w2
from lift_ce.ffhq.bottleneck import gradient_basis, FrozenContent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);out=parser.parse_args().output
    torch.set_num_threads(1)
    c=json.loads((out/'resolved_config.json').read_text());parent=Path(c['parent_run']);spec=json.loads((out/'selected_configuration.json').read_text());frozen=json.loads((out/'frozen.json').read_text());images=load(out/'frozen.pt');basis=load(out/'checkpoints/task_basis.pt');manifest=json.loads((out/'bottleneck_manifest.json').read_text())
    assert sha(out/'frozen.pt')==frozen['sha256'] and sha(out/'checkpoints/task_basis.pt')==frozen['basis_sha256'] and sha(out/'selected_configuration.json')==frozen['selected_sha256']
    for p,digest in json.loads((out/'input_manifest.json').read_text()).items():assert sha(p)==digest,p
    u,v,s,k=gradient_basis(basis['gradients']);assert k==spec['bottleneck_dimension'];torch.testing.assert_close(u@u.T,basis['task']@basis['task'].T,atol=1e-10,rtol=1e-10)
    train=load(parent/'training.pt');assert manifest['training_ids']==train['ids'];assert not set(manifest['training_ids'])&set(np.load(out/'factual_ids.npy'))
    jobs=[json.loads(p.read_text()) for p in (out/'jobs').glob('*/complete.json')]
    assert {int(j['physical_gpu']) for j in jobs}==set(range(8))
    assert len([j for j in jobs if j['phase']=='gradient'])==8
    assert len([j for j in jobs if j['phase']=='metrics'])==8
    assert all(j['start_utc']>frozen['frozen_utc'] for j in jobs if j['phase'] in ['future','metrics'])
    assert manifest['frozen_utc']<spec['selected_utc']<frozen['frozen_utc']
    calibration=pd.read_csv(out/'calibration_results.csv');best=min(calibration.to_dict('records'),key=lambda z:(round(z['calibration_max_mean_hinge'],6),z['pixel_L2_cost'],z['r'],z['beta_geo']))
    assert best['r']==spec['r_star'] and best['beta_geo']==spec['beta_geo_star']
    pop=load(parent/'test_population.pt');reference=load(parent/'references.pt')['w'];real=FrozenContent(basis['task'],pop['factual_w']);z=real.encode(pop['base_w']).float();target=real.encode(reference).float();D=exact_w2(z,target).distance
    geom=pd.read_csv(out/'geometry_ablation.csv');drift=0.
    for row in geom.itertuples():
        item=images[f'Bottleneck_beta{row.beta_geo:.2f}'];q=item['points'];length=exact_w2(z,q).distance+exact_w2(q,target).distance
        assert abs(D-row.D)<1e-6 and abs(length-row.total_length)<1e-5 and length<=row.beta_geo*D+1e-6*max(1,D)
        torch.testing.assert_close(real.realize(q),item['w'],atol=2e-6,rtol=1e-6)
        error=float(((item['w'].double()-pop['factual_w'].double())@basis['content']).abs().max());assert error<2e-6;drift=max(drift,error)
    assert (geom.sort_values('beta_geo').Phi_hat_r.diff().dropna()<=1e-6).all()
    clf=SmileClassifier('cpu');bankrows=records(parent,spec['r_star']);states=torch.stack([load(Path(r['checkpoint']))['projection'] for r in bankrows]);risk=pd.read_csv(out/'finite_bank_risk.csv');risk_error=0.
    with torch.no_grad():
        for row in risk.itertuples():
            p=logits(clf,images[row.method]['features'],states).softmax(-1)[:,:,0];actual=float(torch.relu(.5-p).double().mean(1).max());risk_error=max(risk_error,abs(actual-row.Phi_hat_r));assert abs(actual-row.Phi_hat_r)<2e-5
    fut=pd.concat([pd.read_csv(p) for p in sorted((out/'jobs').glob('future_*/future.csv'))],ignore_index=True)
    assert len(fut)==30*8*len(images)
    future_error=0.
    with torch.no_grad():
        for path,df in fut.groupby('checkpoint'):
            projection=load(Path(path))['projection']
            for method,rows in df.groupby('method'):
                actual=clf.logits_features(images[method]['features'],projection).softmax(1)[:,0].numpy();stored=rows.sort_values('test_index').probability.to_numpy();future_error=max(future_error,float(np.max(np.abs(actual-stored))))
                assert np.array_equal(actual>=.5,rows.sort_values('test_index').future_validity.to_numpy().astype(bool))
    assert future_error<2e-5
    fut.to_csv(out/'future_validity_per_example.csv',index=False)
    fv=fut.groupby(['method','family','severity','realization'],as_index=False).agg(future_validity=('future_validity','mean'),mean_hinge=('hinge','mean'));fv.to_csv(out/'future_validity.csv',index=False)
    metric=pd.concat([pd.read_csv(p) for p in sorted((out/'jobs').glob('metrics_*/metrics.csv'))],ignore_index=True)
    embedding_error=0.
    for i in range(8):
        emb=np.load(out/'jobs'/f'metrics_{i}'/'identity_embeddings.npz')
        for row in metric[metric.test_index==i].itertuples():
            if 'Factual' in emb and row.method in emb:
                value=float(np.dot(emb['Factual'],emb[row.method]));embedding_error=max(embedding_error,abs(value-row.identity_similarity));assert abs(value-row.identity_similarity)<1e-6
            else:assert pd.isna(row.identity_similarity)
    metric.to_csv(out/'image_metrics_per_example.csv',index=False)
    identity=[]
    for method,d in metric.groupby('method'):
        valid=d.identity_similarity.dropna();identity.append({'method':method,'Identity_Similarity':valid.mean(),'identity_std':valid.std(ddof=0),'identity_p10':valid.quantile(.1),'identity_valid_pairs':len(valid),'identity_missing_pairs':len(d)-len(valid),'LPIPS_squeeze':d.LPIPS_squeeze.mean(),'pixel_L2_1024':d.pixel_L2_1024.mean()})
    identity=pd.DataFrame(identity);identity.to_csv(out/'identity_metrics.csv',index=False)
    timing=pd.DataFrame([json.loads(p.read_text()) for p in (out/'jobs').glob('*/launch_timing.json')]);timing.to_csv(out/'runtime.csv',index=False)
    ri=spec['r_index']
    def parent_duration(pattern):
        return sum(json.loads(p.read_text())['wall_seconds'] for p in (parent/'jobs').glob(pattern+'/launch_timing.json'))
    base_seconds=json.loads((parent/'jobs/base_test/quality.json').read_text())['base_seconds']
    bottleneck_seconds=float(timing[timing.job.str.startswith(f'opt_bottleneck_test_{ri}_') | (timing.job==f'pool_bottleneck_test_{ri}')].wall_seconds.sum())
    if images['Full_GRACE'].get('reused_from'):
        full_seconds=parent_duration(f'opt_test_{ri}_*')+parent_duration(f'pool_test_{ri}')
    else:full_seconds=float(timing[timing.job.str.startswith(f'opt_full_test_{ri}_') | (timing.job==f'pool_full_test_{ri}')].wall_seconds.sum())
    nominal=fv[fv.severity==1].groupby('method',as_index=False).future_validity.mean().rename(columns={'future_validity':'Future_Validity'})
    all_results=nominal.merge(risk,on='method').merge(identity,on='method')
    durations={'Base':base_seconds,'Full_GRACE':full_seconds,'Bottleneck_GRACE':bottleneck_seconds}
    all_results['runtime_worker_seconds']=all_results.method.map(durations)
    all_results.to_csv(out/'all_results.csv',index=False)
    main=all_results[all_results.method.isin(durations)].set_index('method').loc[list(durations)].reset_index();main.to_csv(out/'full_vs_bottleneck.csv',index=False)
    ablation=all_results[all_results.method.str.startswith('Bottleneck_beta')].copy();ablation['beta_geo']=ablation.method.str.replace('Bottleneck_beta','').astype(float);ablation['runtime_worker_seconds']=bottleneck_seconds
    ablation=ablation.merge(geom[['beta_geo','D','B_geo','total_length','slack']],on='beta_geo');ablation.to_csv(out/'geometry_ablation.csv',index=False)
    reconstruction=pd.concat([pd.read_csv(p) for p in (out/'jobs').glob('reconstruction_*/reconstruction.csv')],ignore_index=True);assert len(reconstruction)==64
    reconstruction.to_csv(out/'reconstruction_per_example.csv',index=False);probe=json.loads((out/'probe_diagnostics.json').read_text());diagnostic={**probe,'reconstruction_error':reconstruction.reconstruction_error.mean(),'reconstruction_max_error':reconstruction.native_max_error.max(),'reconstruction_latent_max_error':reconstruction.latent_max_error.max()};assert diagnostic['reconstruction_latent_max_error']<1e-6;pd.DataFrame([diagnostic]).to_csv(out/'bottleneck_diagnostics.csv',index=False)
    # All test IDs, fixed before evaluation. Two figures keep the columns legible.
    panels=[('ffhq_identity_qualitative',['Factual','Base','Full_GRACE','Bottleneck_GRACE']),('ffhq_identity_beta',['Factual']+[f'Bottleneck_beta{b:.2f}' for b in c['beta_grid']])]
    for filename,methods in panels:
        fig,axes=plt.subplots(8,len(methods),figsize=(2*len(methods),16))
        for i in range(8):
            for j,method in enumerate(methods):
                axes[i,j].imshow(Image.open(out/'qualitative_examples'/f'{i}_{method}.png'));axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
                for spine in axes[i,j].spines.values():spine.set_visible(False)
                if i==0:axes[i,j].set_title(method.replace('_',' ').replace('Bottleneck beta','β='),fontsize=10)
                if j==0:axes[i,j].set_ylabel(f'Test {i}',fontsize=9)
        fig.tight_layout(pad=.3);fig.savefig(out/'figures'/f'{filename}.pdf',dpi=160);fig.savefig(out/'figures'/f'{filename}.png',dpi=100);plt.close(fig)
    f=main.set_index('method');bn=f.loc['Bottleneck_GRACE'];full=f.loc['Full_GRACE'];base=f.loc['Base'];gain_full=full.Future_Validity-base.Future_Validity;gain_bn=bn.Future_Validity-base.Future_Validity
    summary={'selected_r':spec['r_star'],'selected_beta':spec['beta_geo_star'],'task_dimension':k,'full_FV_gain_over_base':gain_full,'bottleneck_FV_gain_over_base':gain_bn,'retained_fraction_of_full_FV_gain':gain_bn/gain_full if gain_full else None,'identity_similarity_gain_vs_full':bn.Identity_Similarity-full.Identity_Similarity,'LPIPS_reduction_vs_full':full.LPIPS_squeeze-bn.LPIPS_squeeze,'hard_content_max_drift':drift}
    dump(out/'tradeoff.json',summary)
    review={'status':'passed','jobs':len(jobs),'gpu_ids':list(range(8)),'source_hashes_unchanged':True,'basis_training_only':True,'basis_energy_rule_recomputed':True,'calibration_selection_recomputed':True,'all_geometry_feasible':True,'max_content_coordinate_drift':drift,'max_bank_risk_error':risk_error,'max_future_probability_error':future_error,'max_identity_cosine_error':embedding_error,'max_decoded_image_error':float(metric.decoded_image_max_error.max()),'identity_missing_pairs':int(identity.identity_missing_pairs.sum()),'all_test_ids_displayed':True}
    dump(out/'completion_review.json',review)
    cols=['method','Future_Validity','pixel_L2_224','Identity_Similarity','identity_std','identity_p10','LPIPS_squeeze','runtime_worker_seconds']
    report=['# FFHQ hard-content bottleneck GRACE',
      f'Raw run: `{out}`. Parent frozen run: `{parent}`. Completed on eight GPUs. This is a matched follow-up on eight generated faces with previously evaluated future models, not a new unseen-shift benchmark.',
      '## LeapFactual reuse and decomposition',
      'Original StyleGAN3, CLIP, flow-generated Base CFs, factual IDs, targets, training/calibration populations, attack bank, calibration models and future models are reused. No original FFHQ task/content decomposition or identity model exists; see identity_bottleneck_inventory.md. New ArcFace is an evaluation component only. No identity loss or current-validity branch is present.',
      f'Training-only gradient SVD retains {k} of512 dimensions at95% energy. Task coordinates u=hU; immutable content v=h_fact−h_fact UUᵀ; synthesis h′=v+u′Uᵀ. Target geometry uses only target u. The optimizer changes only u, with max_model mean_CF hinge and the sole beta×D geometry constraint. Original Base can change when realized with factual v; Conditional_Base is reported separately in all_results.csv.',
      '## Bottleneck diagnostics',pd.DataFrame([diagnostic]).to_markdown(index=False),
      'Probes use the same512-input logistic architecture, C=1, no feature scaling, training256/calibration64. These are CLIP pseudo-labels from prompts “smiling face” versus “face”, not annotated smile labels. Complementary content is not assumed classification-independent. Reconstruction uses the64 calibration latents and the original generator.',
      '## Calibration',json.dumps(spec,indent=2),
      'Both r and beta were selected using only four saved calibration models. Full-latent GRACE is matched to the same selected r, beta and bank; its stored output is reused when available. Original full-latent candidates and new task-space candidates use the same solver settings and candidate-pooling policy.',
      '## Table 1: matched results at nominal severity',main[cols].to_markdown(index=False),
      'Cost is factual-to-output pixel L2 at224RGB. Identity is cosine of512-dimensional ArcFace embeddings after independent five-landmark alignment; std uses ddof=0, low percentile is10th percentile. Largest detected face is used. Missing detections are reported in identity_metrics.csv; per-example detection counts and similarities are retained. Similarity is a metric, not proof of identity.',
      '## Table 2: task-space geometry budget',ablation[['beta_geo','Future_Validity','Phi_hat_r','pixel_L2_224','Identity_Similarity','LPIPS_squeeze','runtime_worker_seconds']].to_markdown(index=False),
      '## Robustness versus identity tradeoff',json.dumps(summary,indent=2),
      'Selected candidate origins: '+json.dumps({name:item.get('origin','original') for name,item in images.items()})+'. All geometry candidates are judged by the unchanged robust objective; identity metrics do not influence selection.',
      'Positive identity-similarity gain versus full GRACE indicates less measured drift. The retained robustness-gain fraction compares gains above the original Base at nominal severity; it is descriptive on this fixed small cohort and is not a statistical guarantee. Negative or small gains must not be described as successful identity preservation.',
      '## Runtime',
      f'New construction wall time: {json.loads((out/"construction_clock.json").read_text())["wall_seconds"]:.2f}s. Total new worker occupancy: {timing.wall_seconds.sum()/3600:.4f} GPU-worker hours, including reconstruction and evaluation. Table1 runtime is worker-seconds for the entire eight-CF population: original Base generation, full-latent test optimization/pooling, or bottleneck test optimization/pooling. Shared basis/bank/calibration costs are excluded from method rows and itemized in runtime.csv. Reused historical costs are reported, not rerun; bottleneck beta rows share pooled candidates, so runtime must not be summed across beta.',
      '## Qualitative panels',
      'All eight prespecified test IDs are shown, in fixed order, in ffhq_identity_qualitative.pdf and ffhq_identity_beta.pdf. The panels expose facial appearance, hair, pose, background and smile changes. No favorable-example selection was used. Visual identity-preservation claims require inspection and are not inferred from latent freezing.',
      '## Limitations',
      'The SVD split is approximate and classifier-dependent; task directions can still encode appearance and content directions can retain smile information. Generated FFHQ-style faces, prompt-relative labels, small populations, projection-only Update and finite attack bank limit generalization. Face alignment/detection and ArcFace domain behavior affect identity scores. The historical brightness/blur replicas include deterministic duplicates. Reusing previously evaluated future models yields a controlled follow-up, not independent replication. The central GRACE objective/theory and StyleMNIST/Galaxy runs are unchanged.',
      '## Independent verification',json.dumps(review,indent=2)]
    (out/'report.md').write_text('\n\n'.join(report)+'\n')
    dest=ROOT/'results/015_ffhq_identity_v1';dest.mkdir(exist_ok=True);figdest=ROOT/'figures/ffhq_grace_identity_v1';figdest.mkdir(exist_ok=True)
    for name in ['report.md','identity_bottleneck_inventory.md','resolved_config.json','selected_configuration.json','bottleneck_manifest.json','bottleneck_diagnostics.csv','identity_metrics.csv','full_vs_bottleneck.csv','geometry_ablation.csv','future_validity.csv','runtime.csv','completion_review.json','tradeoff.json','all_results.csv','image_metrics_per_example.csv','identity_model_manifest.json']:
        shutil.copyfile(out/name,dest/name)
    for path in (out/'figures').glob('*'):shutil.copyfile(path,figdest/path.name)
    dump(out/'completed.json',review);print(main[cols].to_string(index=False),flush=True)

if __name__=='__main__':main()

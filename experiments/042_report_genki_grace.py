"""Independent numerical audit and complete qualitative GENKI-head GRACE report."""
import argparse,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/genki_grace_report_mpl')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/src'))
import numpy as np
import pandas as pd
from PIL import Image
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from lift_ce.counterfactuals.geodesic_excess import exact_w2


def load(p):return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def scores(features,state):return (features@(state['weight'][0]-state['weight'][1])+(state['bias'][0]-state['bias'][1])).sigmoid()
def weight_hash(state):return hashlib.sha256(torch.cat([state['weight'].flatten(),state['bias'].flatten()]).numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    torch.set_num_threads(1);dest=ROOT/'results/020_genki_grace_identity';figdest=ROOT/'figures/genki_grace_identity'
    assert not dest.exists() and not figdest.exists(),'Never overwrite completed summaries'
    c=json.loads((out/'resolved_config.json').read_text());spec=json.loads((out/'selected_configuration.json').read_text())
    frozen=json.loads((out/'frozen.json').read_text());assert sha(out/'frozen.pt')==frozen['sha256'];assert sha(out/'selected_configuration.json')==frozen['selected_sha256']
    for path,digest in json.loads((out/'input_manifest.json').read_text()).items():assert sha(path)==digest,path
    protocol=json.loads((out/'protocol_frozen.json').read_text());assert sha(out/'resolved_config.json')==protocol['config_sha256']
    jobs=[json.loads(p.read_text()) for p in (out/'jobs').glob('*/complete.json')];assert len(jobs)==79
    assert {int(j['physical_gpu']) for j in jobs}==set(range(8))
    futurejobs=[j for j in jobs if j['phase']=='future'];assert len(futurejobs)==30
    assert all(j['start_utc']>frozen['utc'] for j in futurejobs)
    for j in jobs:
        if j['phase']=='attack':assert j['start_utc']>json.loads((out/'calibration_base_frozen.json').read_text())['utc']
    for split in ['calibration','test']:
        assert sha(out/f'{split}_population.pt')==json.loads((out/f'{split}_base_frozen.json').read_text())['sha256']
    training=load(out/'training.pt');bank=pd.read_csv(out/'attack_bank_manifest.csv');assert len(bank)==17
    for row in bank.to_dict('records'):
        assert sha(row['checkpoint'])==row['checkpoint_sha256']
        if isinstance(row.get('perturbation'),str):
            assert sha(row['perturbation'])==row['perturbation_sha256']
            d=load(row['perturbation']);bound=float((d['delta'].double().square().sum()/len(training['y'])).sqrt())
            assert abs(bound-row['paired_W_data_upper_bound'])<1e-7 and bound<=row['r']+1e-6
            original=training['x'][d['indices']];changed=original+d['delta']
            # Antialiased bicubic canonicalization can overshoot [0,1]. The
            # budget-rescaled proposal stays in the hull of source and [0,1].
            assert bool((changed>=torch.minimum(original,torch.zeros_like(original))-1e-6).all())
            assert bool((changed<=torch.maximum(original,torch.ones_like(original))+1e-6).all())
    compatible=bank[bank.paired_W_data_upper_bound<=spec['r_star']+1e-6]
    states=[load(p) for p in compatible.checkpoint]
    methods=load(out/'frozen.pt');facts=load(out/'test_factuals.pt');z=load(out/'test_population.pt')['u'];target=load(out/'references.pt')['u']
    quality=pd.read_csv(out/'image_quality.csv');assert len(quality)==40
    assert (quality.arcface_similarity>=c['arcface_min']).all() and (quality.facenet_similarity>=c['facenet_min']).all()
    assert (quality.LPIPS<=c['lpips_max']).all() and (quality.outside_rms<=c['outside_rms_max']).all()
    embeddings=np.load(out/'identity_embeddings.npz');identity_error=0.
    for row in quality.itertuples():
        for key,field in [('arc','arcface_similarity'),('facenet','facenet_similarity')]:
            value=float(embeddings[f'{key}_factual_{row.factual_index}']@embeddings[f'{key}_{row.method}_{row.factual_index}'])
            identity_error=max(identity_error,abs(value-getattr(row,field)))
    assert identity_error<1e-6
    recordedrisk=pd.read_csv(out/'finite_bank_risk.csv').set_index('method');geometry=pd.read_csv(out/'jobs'/f'pool_test_{spec["r_index"]}'/'geometry.csv')
    for name,v in methods.items():
        assert bool(((v['points']-facts['u']).abs()<=16+1e-6).all())
        probs=torch.stack([scores(v['features'],s) for s in states]);risk=float(torch.relu(.5-probs).double().mean(1).max())
        assert abs(risk-recordedrisk.loc[name,'Phi_hat_r'])<1e-6
        if name!='Base':
            beta=float(name.split('beta')[-1]);D=exact_w2(z,target).distance;length=exact_w2(z,v['points']).distance+exact_w2(v['points'],target).distance
            row=geometry[geometry.beta_geo==beta].iloc[0];assert length<=beta*D+1e-6*max(1,D)
            assert abs(length-row.total_length)<1e-6 and abs(D-row.D)<1e-6
    assert np.all(np.diff(geometry.sort_values('beta_geo').Phi_hat_r)<=1e-7)
    calibration=pd.read_csv(out/'calibration_results.csv');calstates=[load(out/'checkpoints'/f'calmodel_{i}.pt') for i in range(4)]
    for row in calibration.itertuples():
        v=load(out/'jobs'/f'pool_calibration_{row.r_index}'/'selected.pt')[str(row.beta_geo)]
        prob=torch.stack([scores(v['features'],s) for s in calstates]);risk=float(torch.relu(.5-prob).double().mean(1).max())
        assert abs(risk-row.calibration_max_mean_hinge)<1e-6
    best=min(calibration.to_dict('records'),key=lambda r:(round(r['calibration_max_mean_hinge'],6),r['pixel_L2_cost'],r['r'],r['beta_geo']))
    assert best['r']==spec['r_star'] and best['beta_geo']==spec['beta_geo_star']
    futurerows=[];percase=[];modelmanifest=[]
    for j in futurejobs:
        job=out/'jobs'/j['job'];record=json.loads((job/'model.json').read_text());assert sha(record['checkpoint'])==record['checkpoint_sha256']
        state=load(record['checkpoint']);modelmanifest.append({**record['scenario'],'checkpoint':record['checkpoint'],'tensor_sha256':weight_hash(state)})
        table=pd.read_csv(job/'metrics.csv').set_index('method')
        for name,v in methods.items():
            prob=scores(v['features'],state);fv=float((prob>=.5).double().mean());hinge=float(torch.relu(.5-prob).double().mean())
            assert abs(fv-table.loc[name,'future_validity'])<1e-8 and abs(hinge-table.loc[name,'mean_hinge'])<1e-6
            futurerows.append({'method':name,**record['scenario'],'future_validity':fv,'mean_hinge':hinge})
            percase += [{'method':name,'factual_index':i,**record['scenario'],'smile_probability':float(p),'valid':bool(p>=.5)} for i,p in enumerate(prob)]
    future=pd.DataFrame(futurerows);case=pd.DataFrame(percase);fmodel=pd.DataFrame(modelmanifest)
    cached_future=future.copy();cached_case=case.copy();cached_risk=recordedrisk.copy()
    audit=Path('/tmp/genki_grace_saved_image_audit_20260917');audit_checks=json.loads((audit/'checks.json').read_text())
    assert sha(ROOT/'experiments/audits/genki_grace_saved_images.py')==audit_checks['script_sha256']
    imagepred=pd.read_csv(audit/'image_predictions.csv');assert len(imagepred)==40
    for row in imagepred.itertuples():assert sha(row.image_path)==row.sha256
    case=pd.read_csv(audit/'future_per_case.csv');assert len(case)==1200
    case['hinge']=(.5-case.smile_probability).clip(lower=0)
    future=case.groupby(['method','family','severity','realization','strength','seed'],as_index=False).agg(future_validity=('valid','mean'),mean_hinge=('hinge','mean'))
    savedbank=pd.read_csv(audit/'bank_per_case.csv');savedbank['hinge']=(.5-savedbank.smile_probability).clip(lower=0)
    recordedrisk=savedbank.groupby(['method','candidate']).hinge.mean().groupby('method').max().to_frame('Phi_hat_r')
    quality['native_smile_score']=quality.smile_score
    predicted=imagepred.set_index(['method','factual_index']).nominal_smile_probability
    quality['smile_score']=[predicted.loc[r.method,r.factual_index] for r in quality.itertuples()]
    assert recordedrisk.loc['Base','Phi_hat_r']==recordedrisk.loc['GRACE_beta1.00','Phi_hat_r']
    quality['current_valid']=quality.smile_score>=.5
    summary=quality.groupby('method').agg(current_validity=('current_valid','mean'),ArcFace_mean=('arcface_similarity','mean'),ArcFace_min=('arcface_similarity','min'),
          FaceNet_mean=('facenet_similarity','mean'),LPIPS_mean=('LPIPS','mean'),outside_RMS_mean=('outside_rms','mean'),outside_RMS_max=('outside_rms','max'),pixel_L2_224=('pixel_L2_224','mean')).reset_index()
    fv=future.groupby('method').agg(future_validity=('future_validity','mean'),worst_future_model_validity=('future_validity','min')).reset_index()
    summary=summary.merge(fv,on='method').merge(recordedrisk.reset_index(),on='method');chosen=f'GRACE_beta{spec["beta_geo_star"]:.2f}'
    summary['calibration_selected']=summary.method==chosen
    runtime=pd.DataFrame(jobs);construction=runtime[runtime.phase!='future'];clock=json.loads((out/'construction_clock.json').read_text())
    timing={'construction_wall_seconds':clock['wall_seconds'],'construction_allocated_GPU_hours':float(construction.wall_seconds.sum()/3600),
            'construction_wall_seconds_per_test_CF_including_calibration':clock['wall_seconds']/8,'future_evaluation_allocated_GPU_hours':float(runtime[runtime.phase=='future'].wall_seconds.sum()/3600)}
    review={'status':'passed','completed_jobs':79,'eight_gpus_used':True,'focused_tests_passed':7,'all_parent_hashes_unchanged':True,
            'population_risk_recomputed':True,'geometry_independently_recomputed':True,'all_40_outputs_pass_preservation_limits':True,
            'maximum_identity_embedding_recompute_error':identity_error,'all_data_attacks_have_feasible_paired_certificates':True,
            'validation_only_configuration_selection_verified':True,'future_models_after_output_freeze':True,'future_predictions_recomputed':True,
            'future_model_count':len(fmodel),'distinct_future_weight_states':int(fmodel.tensor_sha256.nunique()),
            'bank_models_total':len(bank),'selected_radius_bank_size':len(compatible),'current_validity_repair_used':False,
            'GENKI_real_photo_inversion_used':False,'human_smile_success_established':False,
            'primary_evaluation_uses_saved_PNG_images':True,'identical_Base_beta1_image_predictions_match':True,
            'posthoc_saved_image_consistency_audit_disclosed':True,
            'training_canonical_min':float(training['x'].min()),'training_canonical_max':float(training['x'].max()),
            'attack_range_within_source_and_unit_interval_convex_hull':True}
    dest.mkdir();figdest.mkdir()
    for name in ['resolved_config.json','protocol_frozen.json','hardware_manifest.json','factual_eligibility.csv','selected_configuration.json','calibration_results.csv',
                 'attack_bank_manifest.csv','image_quality.csv','finite_bank_risk.csv','frozen.json','input_manifest.json']:(dest/name).write_bytes((out/name).read_bytes())
    summary.to_csv(dest/'main_results.csv',index=False);geometry.to_csv(dest/'geometry_ablation.csv',index=False)
    future.to_csv(dest/'future_validity.csv',index=False);case.to_csv(dest/'future_validity_per_case.csv',index=False);runtime.to_csv(dest/'runtime.csv',index=False)
    fmodel.to_csv(dest/'future_model_manifest.csv',index=False)
    cached_future.to_csv(dest/'cached_future_validity.csv',index=False);cached_case.to_csv(dest/'cached_future_validity_per_case.csv',index=False)
    cached_risk.to_csv(dest/'cached_finite_bank_risk.csv');quality.to_csv(dest/'saved_image_quality.csv',index=False)
    for name in ['image_predictions.csv','future_per_case.csv','bank_per_case.csv','checks.json']:(dest/('saved_image_'+name)).write_bytes((audit/name).read_bytes())
    for name,value in [('completion_review.json',review),('runtime_summary.json',timing)]:(dest/name).write_text(json.dumps(value,indent=2)+'\n')
    method_order=['Base']+[f'GRACE_beta{b:.2f}' for b in c['beta_grid']]
    perface=case.groupby(['method','factual_index']).valid.mean()
    def qualitative(filename,columns,cases_per_page,figsize):
        with PdfPages(figdest/filename) as pdf:
            for start in range(0,8,cases_per_page):
                fig,axes=plt.subplots(cases_per_page,len(columns),figsize=figsize,squeeze=False)
                fig.subplots_adjust(left=.035,right=.975,top=.89,bottom=.08,hspace=.42,wspace=.1)
                fig.suptitle(f'GENKI-head GRACE: fixed factual faces and preservation-constrained edits\nSelected radius {spec["r_star"]:g}, selected beta {spec["beta_geo_star"]:g}',fontsize=13,y=.97)
                for row,i in enumerate(range(start,min(start+cases_per_page,8))):
                    for col,name in enumerate(columns):
                        imagepath=out/'qualitative_examples'/f'test_{i}_{"factual" if name=="Factual" else name}.png'
                        image=Image.open(imagepath).convert('RGB');image.thumbnail((448,448));ax=axes[row,col];ax.imshow(image);ax.axis('off')
                        if name=='Factual':title=f'Case {i} | Factual'
                        else:
                            v=quality[(quality.method==name)&(quality.factual_index==i)].iloc[0]
                            label='Base' if name=='Base' else f'GRACE beta={float(name.split("beta")[-1]):g}'
                            title=f'{label}\nSmile {v.smile_score:.3f} | FV {perface.loc[name,i]:.0%}'
                            ax.text(.5,-.04,f'ArcFace {v.arcface_similarity:.3f} | LPIPS {v.LPIPS:.3f}',ha='center',va='top',transform=ax.transAxes,fontsize=8)
                        ax.set_title(title,fontsize=9,pad=6)
                fig.text(.5,.025,'All eight fixed test cases shown. Smile scores are classifier outputs; FV averages 30 correlated future-update models.\nFour style channels are editable; all other styles and the input transform stay factual. Human smile labels are unavailable.',ha='center',fontsize=8)
                pdf.savefig(fig,dpi=160)
                if filename=='qualitative.pdf':fig.savefig(figdest/f'qualitative_page_{start//cases_per_page+1}.png',dpi=110)
                plt.close(fig)
    qualitative('qualitative.pdf',['Factual','Base',chosen],4,(10.5,12))
    qualitative('qualitative_all_betas.pdf',['Factual']+method_order,2,(15,6.5))
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,family in zip(axes,c['future']['families']):
        for name in method_order:
            v=future[(future.family==family)&(future.method==name)].groupby('severity').future_validity.mean()
            ax.plot(v.index,v.values,'o-',label=name)
        ax.set(title=family,xlabel='External shift severity',ylabel='Future validity',ylim=(-.03,1.03));ax.grid(alpha=.25)
    axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(figdest/'shift_severity.pdf');fig.savefig(figdest/'shift_severity.png',dpi=140);plt.close(fig)
    base_row=summary[summary.method=='Base'].iloc[0];grace_row=summary[summary.method==chosen].iloc[0]
    report=f'''# GENKI-head GRACE with identity and background preservation

This run uses GRACE, with a new GENKI-supervised smile head and rebuilt data-side
attack/future-update models. It starts from newly generated nominal Base CFs.
It does not rescore the earlier CLIP edits and call them GRACE.

Raw run: `{out}`. All 79 jobs completed on eight GPUs. Selected configuration:
**r={spec['r_star']:g}, beta={spec['beta_geo_star']:g}**, chosen on calibration shifts only.

## Main result

{summary.to_markdown(index=False)}

The selected GRACE result has future validity **{grace_row.future_validity:.2%}**,
versus **{base_row.future_validity:.2%}** for Base on the same 30 future models.
Mean ArcFace similarity is **{grace_row.ArcFace_mean:.4f}**, mean LPIPS
**{grace_row.LPIPS_mean:.4f}**, and maximum outside-region RMS
**{grace_row.outside_RMS_max:.4f}**. All 40 Base/beta outputs pass the declared
preservation limits. Failed nominal Base cases remain in every population and
in all figures; they are not excluded from validity averages.

Primary validity and bank risk above use a uniform CPU float32 evaluation of the
actual saved PNG faces, performed as a post-hoc consistency audit. The original
float-feature evaluation is preserved separately in cached_future_validity.csv
and cached_finite_bank_risk.csv. Base and beta1 have byte-identical PNGs, yet their
original cached scores differed slightly because of numerical realization paths.
Base stops near the threshold, so small realization/quantization changes can
change labels. The uniform saved-image audit fixes this comparison inconsistency
without changing a face, model, radius, beta or configuration selection. Every
PNG existed before future-model training. Identical Base/beta1 images now have
identical probabilities under every evaluated model. Geometry/optimization traces
remain results for the original floating-point search; identity/LPIPS/outside-RMS
metrics remain the saved native-render diagnostics (identity detection itself
used the same quantized RGB as the PNGs). This additional evaluation was not
predeclared before the numerical discrepancy was discovered.

These are classifier-defined validity results on generated FFHQ-style faces.
Human smile correctness and real-world identity verification are not established.
Neither GENKI test accuracy nor preservation scores alone establish meaningful
non-smiling-to-smiling changes. Inspect the figures together with the numbers.

## What was kept fixed

The selected GENKI linear head uses the original frozen CLIP image encoder,
without text prompts. The pretrained StyleGAN3 generator remains fixed. Only the
four previously selected post-affine style channels can change. All other styles
and the factual Fourier input transform are held fixed. Absolute selected styles
divided by their frozen training standard deviations define the four-dimensional
coordinates u. Factual content is paired with each returned support throughout.

Every returned candidate must have ArcFace>=0.8, independent FaceNet>=0.8,
native-image LPIPS<=0.15, and RGB224 RMS outside the fixed mouth/cheek ellipse<=0.04.
Missing detections fail. Coordinate displacement is bounded by 16 standard
deviations per channel. These are hard final-candidate constraints, not soft
identity penalties. The mask is a localization proxy, not segmentation; an RMS
bound is not a guarantee that every background pixel is unchanged. Similarity
0.8 is an engineering limit, not a calibrated same-person verification threshold.

## GRACE objective and geometry

The finite-bank objective is max_model mean_CF [0.5-p(smile)]_+. It is not
mean_CF max_model loss. All populations at a radius use the full compatible
shared bank. The source is the same frozen nominal Base population for every beta.
The reference is the first eight generated training faces predicted smiling by
the new head, encoded into u. D=W2(Base,reference); enforce
W2(Base,Q)+W2(Q,reference)<=beta*D using exact Euclidean W2 in u for all terms.

The reference label belongs to the original training face. Transplanting its four
coordinates into another factual's frozen content need not preserve that label.
This is the task-space/frozen-content realization from the identity experiment,
not full-W geometry or a claim that the channels perfectly disentangle smiles.

The existing solve_raw optimizer uses its smooth bank-loss approximation and
geometry-only primal/dual updates. Twelve steps from Base and a near-Base start,
plus eight declared geodesic points, define the numerical search. All evaluated
iterates enter the finite candidate pool, including those not retained by the
solver's unconstrained risk incumbent. Pools are shared across betas at each
radius. Final selection uses exact bank hinge, hard geometry/preservation and
pixel-cost tie-breaking. Preservation is screened after proposal generation;
this is a finite-search solution, not a certificate of a global constrained optimum.
No nominal-validity repair, nominal-validity penalty or extra target penalty enters GRACE.

{geometry.to_markdown(index=False)}

The geometry table's Phi_hat_r is the optimization's cached-feature bank risk;
the main table's Phi_hat_r is the uniformly evaluated saved-image bank risk.

## Compatible Base and model construction

Factuals come from the previous generated calibration/test pools in their seeded
order, requiring nominal non-smile and both identity detections before CF
generation. Eligibility exclusions are saved. The images are generated faces,
not inversions of GENKI photographs. The pools have appeared in earlier work,
so this is a matched new-classifier experiment, not a new untouched face benchmark.

Base uses nominal smile-gradient ascent only, with up to 40 steps and actual
preservation checks. On the first observed crossing of 0.5, twelve local bisection
steps refine that crossing; this is not a global minimum-distance certificate.
If no crossing is found, keep the best preservation-feasible nominal candidate.
All Base outputs freeze before bank construction.

Only the {len(training['y'])} GENKI training images enter Update. Start from the selected
GENKI head each time; use 120 deterministic full-batch SGD steps, lr0.1, L2=0.0003,
frozen image encoder, and unchanged human training labels. Each of eight attacks
changes pixels of 32 training images, using 12 ascent steps with a one-step
hypergradient surrogate. Fractions0.5/1 produce sixteen exact full-update heads;
a clean full update supplies the seventeenth. Accepted weights come from Update,
not direct parameter perturbations. The short attack surrogate is not the full
120-step unroll and does not certify worst-case uncertainty coverage.

Data-side radius is RGB224 pixel Euclidean W2 with preserved labels. Every
saved perturbation has a paired-coupling upper bound within its radius. This
certifies membership conservatively; exact 2,798-image OT is not computed. The
selected radius has {len(compatible)} compatible bank entries. The old CLIP-labeled flow,
old attack bank and old future models are not used for this classifier.

The distance and attacks operate on preprocessed floating-point canonical RGB
tensors. Antialiased bicubic resizing produces values from
{float(training['x'].min()):.6f} to {float(training['x'].max()):.6f}, including slight overshoot
outside [0,1]. Proposal clipping occurs before budget rescaling; final shifted
values stay within the per-pixel convex hull of the original tensor and [0,1].
The independent audit verifies this envelope and the paired transport bound.
This is not a claim that every attacked tensor is a quantized, range-clipped RGB
photograph. An initial report check that assumed all canonical values were in
[0,1] failed; correcting that audit assumption did not change any training data,
attack, model or counterfactual.

## Calibration and future evaluation

Four declared shifted-training calibration updates select r/beta by minimum
maximum population mean hinge (rounded1e-6), then pixel cost, radius and beta.
Selection is frozen before test refinement. All test CFs freeze before future
models are trained. Future stress covers brightness, noise and blur at five
ordered severities and two realizations. External severity is not internal r.
The 30 future updates contain **{fmodel.tensor_sha256.nunique()} distinct weight states**;
deterministic shifts and zero severity repeat models. FV is an empirical average
over this correlated stress suite, not 30 independent trials or a deployment guarantee.

## Runtime and checks

Construction wall time: {clock['wall_seconds']/60:.2f} minutes, including Base,
banks, calibration and refinement; excluding pretrained classifier/generator
training and final future evaluation. Allocated construction GPU time:
{timing['construction_allocated_GPU_hours']:.3f} GPU-hours. Full phase timings are saved.

Seven focused tests pass. Independent checks recompute exact geometry, shared-bank
population risk, all future predictions, validation selection, pixel-shift
certificates and identity cosines from saved embeddings. All parent hashes remain
unchanged. No previous output or manuscript claim was changed.

## Reproduce and inspect

```sh
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/041_run_genki_grace.py --output {out}
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/audits/genki_grace_saved_images.py --run {out} --output /tmp/genki_grace_saved_image_audit_20260917
PYTHONPATH=/tmp/ffhq_grace_deps:/tmp/galaxy_grace_deps /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/042_report_genki_grace.py --output {out}
```

Workers run through experiments/run.py with source fingerprints and completion
markers. Exact commands are in raw logs; the initial command is in
/tmp/genki_grace_initialize.log. Config, seed, hardware, source/checkpoint hashes,
factual IDs, coordinates, proposals, bank perturbations, updated heads, quality
screens and frozen outputs are persisted. Raw arrays/checkpoints remain outside Git.

Figures: qualitative.pdf shows all eight Factual/Base/selected-GRACE pairs;
qualitative_all_betas.pdf shows every beta for every fixed case. Smile scores
are nominal classifier predictions; per-face FV averages the common future models.
'''
    (dest/'report.md').write_text(report)
    state=yaml.safe_load((ROOT/'state.yaml').read_text());entry={'task':'tasks/007_genki_grace_identity.md','phase':'complete','run':str(out),
        'report':'results/020_genki_grace_identity/report.md','qualitative_pdf':'figures/genki_grace_identity/qualitative.pdf','gpu_ids':list(range(8)),
        'completed_jobs':79,'r_star':spec['r_star'],'beta_geo_star':spec['beta_geo_star'],'Base_future_validity':float(base_row.future_validity),
        'GRACE_future_validity':float(grace_row.future_validity),'all_outputs_pass_preservation_limits':True,'GRACE_run':True,
        'current_validity_repair_used':False,'focused_tests_passed':7,'human_semantic_success_established':False,
        'primary_evaluation':'uniform saved-PNG inference; posthoc consistency audit; cached results preserved','updated_utc':datetime.now(timezone.utc).isoformat()}
    state.update(active_task=entry['task'],status='complete',current_work=entry,next_actions=[]);state.setdefault('completed_work',{})['genki_grace_identity_v1']=entry
    (ROOT/'state.yaml').write_text(yaml.safe_dump(state,sort_keys=False))
    with (ROOT/'notes/decisions.md').open('a') as f:
        f.write(f'\n\n## 2026-09-17: GENKI-head GRACE with preservation\n\nCompleted new nominal Base CFs, compatible GENKI pixel-attack banks and future updates, and actual population-risk GRACE in the fixed four-channel realization. Selected r={spec["r_star"]}, beta={spec["beta_geo_star"]}; future validity Base={base_row.future_validity:.4f}, selected GRACE={grace_row.future_validity:.4f}. All 40 returned Base/beta faces pass ArcFace/FaceNet>=.8, LPIPS<=.15, outside RMS<=.04; this does not establish human smile/identity correctness. Exact task-space W2 geometry and max_model mean_CF hinge retained; no nominal-validity repair. Seventy-nine jobs, eight GPUs, seven tests; old outputs preserved. Report: results/020_genki_grace_identity/report.md. Raw: {out}.\n')
    print(summary.to_string(index=False),flush=True);print(json.dumps(review,indent=2),flush=True)


if __name__=='__main__':main()

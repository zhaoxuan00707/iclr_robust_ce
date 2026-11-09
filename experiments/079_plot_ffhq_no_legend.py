"""Read-only FFHQ figure: four examples, selected model, absolute RGB difference."""
import os,json,hashlib
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/ffhq_four_mouth_mpl')
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/ffhq_update_aware/20260924_v1')
def main():
 selection=json.loads((RUN/'smile/selection.json').read_text());paths=[RUN/'population.pt',RUN/'smile/selection.json',RUN/'smile/test'/selection['method']/'output.pt'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 pop=torch.load(paths[0],map_location='cpu',weights_only=False)['pairs']['smile']['test'];sel=torch.load(paths[2],map_location='cpu',weights_only=False)['images'];base=pop['base'];diff=sel-base;heat=diff.abs().mean(1);assert sel.shape==base.shape;assert torch.isfinite(heat).all()
 ids=[7,5,1,4];out=ROOT/'figures/ffhq_update_aware/four_mouth_examples_no_legend.pdf';png=out.with_suffix('.png');assert not out.exists() and not png.exists();limit=float(heat.max());scores=[]
 for i in range(len(heat)):
  roi=heat[i,130:195,65:165];scores.append({'index':i,'lower_face_mean_absolute_difference':float(roi.mean()),'lower_face_fraction_of_total_difference':float(roi.sum()/heat[i].sum())})
 plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42});fig,axes=plt.subplots(4,4,figsize=(9.2,9.6));fig.subplots_adjust(left=.01,right=.99,top=.95,bottom=.01,wspace=.04,hspace=.04)
 for row,idx in enumerate(ids):
  for col,im in enumerate([pop['factual'][idx],base[idx],sel[idx]]):axes[row,col].imshow(im.permute(1,2,0).numpy(),vmin=0,vmax=1);axes[row,col].axis('off')
  artist=axes[row,3].imshow(heat[idx].numpy(),cmap='magma',vmin=0,vmax=limit,interpolation='nearest');axes[row,3].axis('off')
 labels=['Factual','Base',f'Selected $r={selection["r"]:g}$',r'$|\mathrm{Selected}-\mathrm{Base}|$']
 for ax,label in zip(axes[0],labels):ax.set_title(label,fontsize=12,pad=10)
 fig.savefig(out,bbox_inches='tight',pad_inches=.025,metadata={'Title':''});fig.savefig(png,dpi=190,bbox_inches='tight',pad_inches=.025);plt.close(fig)
 for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 record={'raw_run':str(RUN),'source_sha256':hashes,'selected_method':selection['method'],'selected_r':selection['r'],'beta':selection['beta'],'indices':ids,'factual_ids':[pop['ids'][i] for i in ids],'selection':'Post-hoc illustrative selection: user-requested ID7 plus IDs5,1,4 showing comparatively strong lower-face changes; not representative or used to choose hyperparameters','approximate_lower_face_ROI_xyxy':[65,130,165,195],'ROI_caveat':'Exploratory rectangle, not anatomical segmentation; it includes surrounding skin. Used only for example selection, never to mask or alter images/heatmaps.','candidate_scores':scores,'heatmap':'mean RGB absolute value of selected minus Base','scale_scope':'linear shared scale, zero to maximum over all eight test differences','limits':[0,limit],'overall_title':False,'legend':False,'figure':str(out),'inputs_unchanged':True}
 (ROOT/'results/032_ffhq_update_aware/four_mouth_examples_no_legend.json').write_text(json.dumps(record,indent=2)+'\n');print(out)
if __name__=='__main__':main()

"""Four largest selected-minus-Base galaxy differences; no labels or legend."""
import os,json,hashlib
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/galaxy_four_no_labels_mpl')
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/galaxy_update_aware/20260924_v1')
def main():
 torch.set_num_threads(1);folder=RUN/'round_cigar';paths=[RUN/'population.pt',folder/'frozen_test_images.pt',folder/'selection.json'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};selection=json.loads(paths[2].read_text());pop=torch.load(paths[0],map_location='cpu',weights_only=False,mmap=True)['pairs']['round_cigar']['test'];images=torch.load(paths[1],map_location='cpu',weights_only=False);base=images['Base'];selected=images[selection['method']];delta=selected-base;heat=delta.abs().mean(1);scores=delta.double().flatten(1).norm(dim=1);ids=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i))[:4];limit=float(heat.max())
 out=ROOT/'figures/galaxy_update_aware/round_cigar_four_strongest_no_labels.pdf';assert not out.exists();fig,axes=plt.subplots(4,4,figsize=(9,9));fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=.025,hspace=.025)
 for row,i in enumerate(ids):
  for col,im in enumerate([pop['factual'][i],base[i],selected[i]]):axes[row,col].imshow(im.permute(1,2,0).numpy().clip(0,1));axes[row,col].axis('off')
  axes[row,3].imshow(heat[i],cmap='magma',vmin=0,vmax=limit,interpolation='nearest');axes[row,3].axis('off')
 fig.savefig(out,bbox_inches='tight',pad_inches=0,metadata={'Title':''});fig.savefig(out.with_suffix('.png'),dpi=180,bbox_inches='tight',pad_inches=0);plt.close(fig)
 for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 record={'raw_run':str(RUN),'source_sha256':hashes,'selected_method':selection['method'],'r':selection['r'],'beta':selection['beta'],'indices':ids,'factual_ids':[pop['ids'][i] for i in ids],'ranking':'post-hoc descending full RGB L2 selected-minus-Base; tie index','scores_all':[float(s) for s in scores],'columns':['Factual','Base','Selected','Mean RGB absolute selected-minus-Base'],'heatmap_limits':[0,limit],'scale':'shared linear absolute scale across all16 test cases; no per-image normalization','title':False,'column_names':False,'legend':False,'inputs_unchanged':True,'caveat':'Largest differences are illustrative, not representative; inherited Base artifacts remain visible, especially ID3.','figure':str(out)}
 (ROOT/'results/031_galaxy_update_aware/round_cigar_four_strongest_no_labels.json').write_text(json.dumps(record,indent=2)+'\n');print('IDs',ids);print('Selected',selection['method']);print(out)
if __name__=='__main__':main()

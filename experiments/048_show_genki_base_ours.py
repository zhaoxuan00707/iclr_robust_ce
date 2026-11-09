"""Rank saved Base/ours pairs by RGB RMS and plot the largest differences."""
import os,json,hashlib
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/genki_samples_mpl')
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
raw=Path('/home/xu.zhao/robust_ce_results/genki_preservation_search/20260917_feasible_v1')
res=ROOT/'results/022_genki_preservation_search';figdir=ROOT/'figures/genki_preservation_search'
output=figdir/'base_vs_ours_largest_differences.pdf';assert not output.exists()
pred=pd.read_csv(res/'image_predictions.csv').set_index(['method','factual_index'])
fv=pd.read_csv(res/'future_per_case.csv').groupby(['method','factual_index']).valid.mean()
images={};rows=[];hashes={}
for i in range(8):
 for method in ['Base','Feasible_search']:
  p=raw/'qualitative_examples'/f'test_{i}_{method}.png';images[i,method]=np.array(Image.open(p).convert('RGB'));hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 a=images[i,'Base'].astype(float)/255;b=images[i,'Feasible_search'].astype(float)/255
 assert a.shape==b.shape
 rows.append({'case':i+1,'factual_index':i,'RGB_RMS_Base_vs_ours':float(np.mean((a-b)**2)**.5)})
rank=pd.DataFrame(rows).sort_values('RGB_RMS_Base_vs_ours',ascending=False);rank.to_csv(res/'base_vs_ours_visual_ranking.csv',index=False)
selected=rank.head(3).factual_index.tolist()
fig,axes=plt.subplots(3,4,figsize=(12,11));fig.subplots_adjust(left=.04,right=.98,top=.90,bottom=.07,hspace=.25,wspace=.08)
fig.suptitle('Base vs ours: three largest saved-image differences',fontsize=17,y=.97)
for row,i in enumerate(selected):
 for col,(method,crop) in enumerate([('Base',False),('Feasible_search',False),('Base',True),('Feasible_search',True)]):
  ax=axes[row,col];im=images[i,method];h,w=im.shape[:2]
  if crop:im=im[int(.5*h):int(.87*h),int(.27*w):int(.73*w)]
  ax.imshow(im);ax.set_xticks([]);ax.set_yticks([])
  if row==0:ax.set_title(('Base' if method=='Base' else 'Ours')+(' — lower-face detail' if crop else ''),fontsize=11,pad=10)
  if col==0:ax.set_ylabel(f'Case {i+1}',fontsize=12)
  if not crop:ax.set_xlabel(f"p(smile)={pred.loc[method,i].nominal_smile_probability:.3f} | FV={fv.loc[method,i]:.0%}",fontsize=10)
fig.text(.04,.025,'Selected by full-image RGB RMS difference, not success. Identical crop coordinates; no contrast or difference amplification.\nThese are the largest differences among eight cases, but the visual changes remain small. FV uses the existing 30 future heads.',fontsize=9)
fig.savefig(output,dpi=170);fig.savefig(figdir/'base_vs_ours_largest_differences.png',dpi=130);plt.close(fig)
for p,digest in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest
(res/'base_vs_ours_samples.json').write_text(json.dumps({'selection':'top 3 full-image RGB RMS difference between frozen Base and Feasible_search PNGs','selected_cases_one_based':[i+1 for i in selected],'crop_fraction_xyxy':[.27,.5,.73,.87],'source_hashes':hashes,'images_regenerated':False,'contrast_amplified':False},indent=2)+'\n')
print(rank.to_string(index=False))

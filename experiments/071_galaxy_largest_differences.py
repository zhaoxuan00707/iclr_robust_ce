"""Post-hoc qualitative selection from frozen task017 outputs; no experiments rerun."""
import hashlib,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/galaxy_largest_differences_mpl')
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/galaxy_update_aware/20260924_v1')
FIG=ROOT/'figures/galaxy_update_aware';RES=ROOT/'results/031_galaxy_update_aware'
def load(p):return torch.load(p,map_location='cpu',weights_only=False,mmap=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);folder=RUN/'round_cigar';paths=[RUN/'population.pt',folder/'frozen_test_images.pt',folder/'test_individual.csv',folder/'test_audit.pt']
 hashes={str(p):sha(p) for p in paths};pop=load(paths[0])['pairs']['round_cigar']['test'];images=load(paths[1]);individual=pd.read_csv(paths[2]);tags=['Base','r0_beta1.02','r0.25_beta1.02'];probs={'Base':load(paths[3])['base_probability']}
 for tag in tags[1:]:
  path=folder/'test'/tag/'output.pt';hashes[str(path)]=sha(path);payload=load(path);torch.testing.assert_close(images[tag],payload['images'],atol=0,rtol=0);probs[tag]=payload['logits'].softmax(1)[:,4]
 delta_a=images[tags[2]]-images[tags[1]];delta_b=images[tags[1]]-images['Base'];maps=[delta_a.abs().mean(1),delta_b.abs().mean(1)]
 rows=[]
 for i in range(16):
  row={'index':i,'id':pop['ids'][i]}
  for name,diff in [('r025_minus_r0',delta_a),('r0_minus_base',delta_b)]:
   row[name+'_L2']=float(diff[i].double().norm());row[name+'_MAE']=float(diff[i].double().abs().mean())
  rows.append(row)
 frame=pd.DataFrame(rows);groups=[]
 for key,title in [('r025_minus_r0','Largest r=0.25 versus r=0 differences'),('r0_minus_base','Largest r=0 versus Base differences')]:
  ranking=frame.sort_values([key+'_L2','index'],ascending=[False,True]);ids=ranking['index'].head(4).tolist();groups.append((key,title,ids));frame[key+'_rank']=frame['index'].map({int(i):rank+1 for rank,i in enumerate(ranking['index'])})
 frame.to_csv(RES/'round_cigar_r025_r0_base_ranking.csv',index=False)
 # Exact same unnormalized absolute scale for both heatmaps and every page.
 vmax=max(float(m.max()) for m in maps);pdf=FIG/'round_cigar_largest_differences_r025.pdf'
 assert not pdf.exists(),'Preserve previous figure outputs; choose a fresh filename'
 with PdfPages(pdf) as document:
  for page,(key,title,ids) in enumerate(groups,1):
   fig,axes=plt.subplots(4,6,figsize=(14,9));fig.subplots_adjust(left=.02,right=.97,bottom=.075,top=.91,wspace=.06,hspace=.19)
   for j,i in enumerate(ids):
    for col,im in enumerate([pop['factual'][i],*[images[tag][i] for tag in tags]]):
     axes[j,col].imshow(im.permute(1,2,0).numpy().clip(0,1));axes[j,col].axis('off')
    axes[j,0].text(.5,-.065,f'ID {i} · rank {j+1}',ha='center',transform=axes[j,0].transAxes,fontsize=9)
    for col,tag in enumerate(tags,1):
     sample=individual[(individual.method==tag)&(individual['index']==i)&(individual.family!='nominal')];assert len(sample)==16
     fv=float((sample.probability>=.5).mean());axes[j,col].text(.5,-.065,f'p={float(probs[tag][i]):.2f}, FV={fv:.0%}',ha='center',transform=axes[j,col].transAxes,fontsize=9)
    for col,(heat,metric) in enumerate(zip(maps,['r025_minus_r0','r0_minus_base']),4):
     artist=axes[j,col].imshow(heat[i],cmap='magma',vmin=0,vmax=vmax);axes[j,col].axis('off');score=float(frame.loc[frame['index']==i,metric+'_L2'].iloc[0]);axes[j,col].text(.5,-.065,f'RGB L2={score:.3f}',ha='center',transform=axes[j,col].transAxes,fontsize=9)
   for ax,titlecol in zip(axes[0],['Factual','Base',r'$r=0$',r'$r=0.25$',r'$|r_{0.25}-r_0|$',r'$|r_0-\mathrm{Base}|$']):ax.set_title(titlecol,fontsize=13)
   fig.suptitle('Round → Cigar · '+title+r' · $\beta=1.02$',fontsize=15)
   colorax=fig.add_axes([.68,.026,.27,.012]);fig.colorbar(artist,cax=colorax,orientation='horizontal',label='Mean absolute RGB difference (shared scale)')
   fig.text(.02,.023,'Post-hoc top 4 by RGB pixel L2 among 16 fixed test cases. Images unchanged.\nID 3 has inherited Base artifacts; this selection illustrates differences, not typical quality.',fontsize=8)
   document.savefig(fig);fig.savefig(FIG/f'round_cigar_largest_differences_r025_page{page}.png',dpi=160);plt.close(fig)
 for p,h in hashes.items():assert sha(Path(p))==h
 manifest={'run':str(RUN),'beta':1.02,'ranking':'descending RGB pixel L2, ties original index','heatmap':'mean-channel absolute difference; no per-image normalization','shared_vmax':vmax,'pages':[{'criterion':k,'ids':ids} for k,title,ids in groups],'input_hashes':hashes,'original_outputs_unchanged':True,'pdf':str(pdf)}
 (RES/'round_cigar_largest_differences_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest['pages']));print(frame.sort_values('r025_minus_r0_rank').to_string(index=False))
if __name__=='__main__':main()

"""Plot saved task016 CFs with separate nominal and robust difference maps."""
import argparse,json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1];fig=root/'figures/mnist_low_margin_base';res=root/'results/030_mnist_low_margin_base';arms=['original','p052','p0505','p05001'];labels=['Original Base','Precise p=0.52','Low margin p=0.505','Near boundary p=0.5001'];data={};bound=0.
 for arm in arms:
  pop=torch.load(a.run/arm/'population.pt',map_location='cpu',weights_only=True);ims=torch.load(a.run/arm/'frozen_images.pt',map_location='cpu',weights_only=True);nom=ims['r0_beta1.02']-ims['Base'];rob=ims['r0.25_beta1.02']-ims['r0_beta1.02'];assert torch.allclose(nom+rob,ims['r0.25_beta1.02']-ims['Base'],atol=1e-7,rtol=0);bound=max(bound,float(nom.abs().max()),float(rob.abs().max()));data[arm]=(pop,ims,rob,nom)
 bound=max(bound,1e-8);out=fig/'qualitative_two_heatmaps.pdf';assert not out.exists(),'Preserve existing figure; choose a new filename for another revision'
 def row(axes,arm,idx):
  pop,ims,rob,nom=data[arm];values=[pop['test']['factual'][idx],ims['Base'][idx],ims['r0_beta1.02'][idx],ims['r0.25_beta1.02'][idx],rob[idx],nom[idx]]
  for j,v in enumerate(values):
   artist=axes[j].imshow(v.squeeze(),cmap='RdBu_r' if j>=4 else 'gray',vmin=-bound if j>=4 else 0,vmax=bound if j>=4 else 1,interpolation='nearest');axes[j].set_xticks([]);axes[j].set_yticks([])
   for spine in axes[j].spines.values():spine.set_visible(False)
  return artist
 titles=['Factual','Base','r = 0','r = 0.25','r = 0.25 − r = 0','r = 0 − Base']
 with PdfPages(out) as pdf:
  for idx in range(8):
   ff,axes=plt.subplots(4,6,figsize=(13,8),layout='constrained')
   for i,arm in enumerate(arms):
    artist=row(axes[i],arm,idx);axes[i,0].set_ylabel(labels[i],fontsize=9)
   for ax,title in zip(axes[0],titles):ax.set_title(title,fontsize=10)
   ff.colorbar(artist,ax=axes[:,4:].ravel().tolist(),shrink=.65,label='Signed pixel difference (shared scale)');ff.suptitle(f'Fixed example {idx+1} · factual ID {int(data[arms[0]][0]["test"]["ids"][idx])}',fontsize=12);pdf.savefig(ff)
   if idx==0:ff.savefig(fig/'matched_example_two_heatmaps.png',dpi=150)
   plt.close(ff)
 # Eight fixed near-boundary cases in one panel.
 ff,axes=plt.subplots(8,6,figsize=(12,15),layout='constrained')
 for i in range(8):artist=row(axes[i],'p05001',i)
 for ax,title in zip(axes[0],titles):ax.set_title(title,fontsize=10)
 ff.colorbar(artist,ax=axes[:,4:].ravel().tolist(),shrink=.6,label='Signed pixel difference (shared scale)');ff.suptitle('Near-boundary Base: first 8 fixed factuals');ff.savefig(fig/'near_boundary_two_heatmaps.pdf');ff.savefig(fig/'near_boundary_two_heatmaps.png',dpi=130);plt.close(ff)
 (res/'two_heatmaps.json').write_text(json.dumps({'raw_run':str(a.run),'columns':titles,'signed_limits':[-bound,bound],'scale_scope':'both differences, all64 factuals and all4 Base conditions','color':'red = increased pixel intensity; blue = decreased intensity','displayed':'first8 fixed factuals; no outcome-based example selection','decomposition_verified':True},indent=2)+'\n');print(out);print('Shared signed limits:',-bound,bound)
if __name__=='__main__':main()

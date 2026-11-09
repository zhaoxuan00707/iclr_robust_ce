"""Title-free four-style task016 panel; reads saved outputs only."""
import hashlib,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mnist_four_styles_mpl')
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/mnist_low_margin_base/20260924_v1/p05001')
def main():
 paths=[RUN/'population.pt',RUN/'frozen_images.pt'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 pop=torch.load(paths[0],map_location='cpu',weights_only=True);ims=torch.load(paths[1],map_location='cpu',weights_only=True)
 ids=[28,30,11,3];styles=['closed triangular','thin upright open top','thick open top','slanted loop-like'];base=ims['Base'];nom=ims['r0_beta1.02'];rob=ims['r0.25_beta1.02'];h0=nom-base;h1=rob-nom
 assert torch.allclose(h0+h1,rob-base,atol=1e-7,rtol=0)
 limit=max(float(h0.abs().max()),float(h1.abs().max()));out=ROOT/'figures/mnist_low_margin_base/four_styles_no_title.pdf';png=out.with_suffix('.png');assert not out.exists() and not png.exists()
 plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
 fig,axes=plt.subplots(4,6,figsize=(10.8,7.45));fig.subplots_adjust(left=.015,right=.985,bottom=.085,top=.93,wspace=.07,hspace=.09)
 for row,idx in enumerate(ids):
  images=[pop['test']['factual'][idx],base[idx],nom[idx],rob[idx],h0[idx],h1[idx]]
  for j,(ax,img) in enumerate(zip(axes[row],images)):
   artist=ax.imshow(img.squeeze().numpy(),cmap='RdBu_r' if j>=4 else 'gray',vmin=-limit if j>=4 else 0,vmax=limit if j>=4 else 1,interpolation='nearest');ax.set_axis_off()
 for ax,label in zip(axes[0],['Factual','Base',r'$r=0$',r'$r=0.25$',r'$r=0\; -\;\mathrm{Base}$',r'$r=0.25\; -\;r=0$']):ax.set_title(label,fontsize=12,pad=9)
 cax=fig.add_axes([.695,.041,.275,.015]);bar=fig.colorbar(artist,cax=cax,orientation='horizontal');bar.set_ticks([-limit,0,limit]);bar.set_ticklabels([f'{-limit:.3f}','0',f'{limit:.3f}']);bar.ax.tick_params(labelsize=9);bar.set_label('Signed pixel difference',fontsize=9,labelpad=2)
 fig.savefig(out,bbox_inches='tight',pad_inches=.025,metadata={'Title':''});fig.savefig(png,dpi=220,bbox_inches='tight',pad_inches=.025);plt.close(fig)
 for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 record={'raw_run':str(RUN),'source_sha256':hashes,'condition':'p05001','beta':1.02,'indices':ids,'factual_ids':[int(pop['test']['ids'][i]) for i in ids],'styles':styles,'selection':'Post-hoc visual diversity of factual handwriting only; not ranked by robustness or heatmap intensity','columns':['Factual','Base','r0','r0.25','r0 minus Base','r0.25 minus r0'],'overall_title':False,'signed_limits':[-limit,limit],'scale_scope':'both contrasts over all 64 near-boundary test cases','figure':str(out),'decomposition_check':True,'inputs_unchanged':True}
 (ROOT/'results/030_mnist_low_margin_base/four_styles_no_title.json').write_text(json.dumps(record,indent=2)+'\n');print(out);print(json.dumps(record,indent=2))
if __name__=='__main__':main()

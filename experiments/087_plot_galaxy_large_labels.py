"""Round-to-Cigar five-column comparison, matching the FFHQ typography."""
import argparse
import os,json,hashlib
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/galaxy_large_labels_mpl')
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/galaxy_update_aware/20260924_v1')
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--limit',type=float,default=None)
 parser.add_argument('--output',type=Path,default=ROOT/'figures/galaxy_update_aware/round_cigar_four_large_labels.pdf')
 parser.add_argument('--record',type=Path,default=ROOT/'results/031_galaxy_update_aware/round_cigar_four_large_labels.json')
 args=parser.parse_args()
 if args.limit is not None and args.limit <= 0:parser.error('--limit must be positive')
 torch.set_num_threads(1);folder=RUN/'round_cigar';paths=[RUN/'population.pt',folder/'frozen_test_images.pt',folder/'selection.json'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 pop=torch.load(paths[0],map_location='cpu',weights_only=False,mmap=True)['pairs']['round_cigar']['test'];images=torch.load(paths[1],map_location='cpu',weights_only=False);selection=json.loads(paths[2].read_text());nominal=images['r0_beta1.02'];grace=images[selection['method']];assert selection['r']==.25 and selection['beta']==1.02
 heat=(grace-nominal).abs().mean(1);limit=args.limit if args.limit is not None else float(heat.max());ids=[3,7,12,14];out=args.output;assert not out.exists() and not out.with_suffix('.png').exists()
 plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42});fig,axes=plt.subplots(4,5,figsize=(12,10.3));fig.subplots_adjust(left=.01,right=.99,bottom=.07,top=.94,wspace=.04,hspace=.04)
 for row,i in enumerate(ids):
  for col,im in enumerate([pop['factual'][i],images['Base'][i],nominal[i],grace[i]]):axes[row,col].imshow(im.permute(1,2,0).numpy().clip(0,1));axes[row,col].axis('off')
  artist=axes[row,4].imshow(heat[i].numpy(),cmap='magma',vmin=0,vmax=limit,interpolation='nearest');axes[row,4].axis('off')
 fig.canvas.draw();fontsize=axes[0,0].get_position().width*fig.get_figwidth()*72/7
 for ax,label in zip(axes[0],['Factual','Base','Nominal','GRACE',r'$\Delta_2$']):ax.set_title(label,fontsize=fontsize,pad=12)
 cax=fig.add_axes([.79,.039,.20,.015]);bar=fig.colorbar(artist,cax=cax,orientation='horizontal');bar.set_ticks([0,limit/2,limit]);bar.set_ticklabels(['0',f'{limit/2:.3f}',f'{limit:.3f}']);bar.ax.tick_params(labelsize=fontsize,pad=4)
 fig.savefig(out,bbox_inches='tight',pad_inches=.025,metadata={'Title':''});fig.savefig(out.with_suffix('.png'),dpi=170,bbox_inches='tight',pad_inches=.025);plt.close(fig)
 for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 record={'figure':str(out),'raw_run':str(RUN),'source_sha256':hashes,'indices':ids,'factual_ids':[pop['ids'][i] for i in ids],'selection':'Same four examples as round_cigar_four_strongest_no_labels.pdf; originally ranked by selected-minus-Base L2, not re-ranked for Delta2','columns':['Factual','Base','Nominal','GRACE','Delta2'],'nominal_r':0,'grace_r':.25,'beta':1.02,'Delta2':'mean RGB absolute difference between GRACE and Nominal','limits':[0,limit],'scale_scope':'shared linear scale over all16 test cases','overall_title':False,'colorbar_label':None,'font_size_pt':fontsize,'font_size_to_tile_width':1/7,'inputs_unchanged':True,'caveat':'ID3 has inherited Base reconstruction artifacts.'}
 record['scale_scope']='user-prescribed fixed scale' if args.limit is not None else record['scale_scope']
 args.record.write_text(json.dumps(record,indent=2)+'\n');print(out)
if __name__=='__main__':main()

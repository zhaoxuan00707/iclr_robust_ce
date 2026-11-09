"""Present all eight frozen FFHQ calibration pairs with GENKI-head scores."""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/genki_qualitative_mpl')
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml

ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    result=ROOT/'results/019_genki_ffhq_classifier'
    figures=ROOT/'figures/genki_ffhq_classifier'
    pdf=figures/'qualitative.pdf';assert not pdf.exists()
    scores=result/'transfer_predictions.csv'
    selected=ROOT/'results/017_ffhq_smile_editor_v2/selected_examples.csv'
    d=pd.read_csv(scores);quality=pd.read_csv(selected)
    quality=quality[quality.editor=='independent4'].set_index('calibration_index')
    assert len(d)==16 and len(quality)==8
    assert set(d['index'])==set(range(8))
    assert d.groupby(['index','kind']).size().eq(1).all()
    manifest={str(p):sha(p) for p in [scores,selected,Path(__file__),*map(Path,d.image_path)]}
    with PdfPages(pdf) as doc:
        for page in range(2):
            fig,axes=plt.subplots(4,2,figsize=(7.5,11.3))
            fig.subplots_adjust(left=.07,right=.93,top=.9,bottom=.09,hspace=.42,wspace=.12)
            fig.suptitle('FFHQ faces scored by the GENKI-4K smile classifier\nFactual images and existing four-channel edits',fontsize=13,y=.97)
            for row,i in enumerate(range(page*4,page*4+4)):
                for col,kind in enumerate(['factual','edited']):
                    record=d[(d['index']==i)&(d.kind==kind)].iloc[0]
                    im=Image.open(record.image_path).convert('RGB');im.thumbnail((448,448))
                    ax=axes[row,col];ax.imshow(im);ax.axis('off')
                    label='Factual' if kind=='factual' else 'Existing edit'
                    title=f'Case {i} | {label} | smile score {record.genki_smile_probability:.3f}'
                    ax.set_title(title,fontsize=9,pad=6)
                    if kind=='edited':
                        q=quality.loc[i]
                        ax.text(.5,-.035,f'ArcFace {q.arcface_similarity:.3f}  |  LPIPS {q.LPIPS:.3f}',transform=ax.transAxes,ha='center',va='top',fontsize=8)
            fig.text(.5,.035,'All eight calibration cases, in their original order. Edits were optimized with the earlier CLIP classifier.\nGENKI-head scores are new evaluations of saved images; these are not new GENKI-optimized counterfactuals.\nGenerated-face smile labels are unavailable; model scores do not establish semantic success.',ha='center',va='center',fontsize=8)
            doc.savefig(fig,dpi=180)
            fig.savefig(figures/f'qualitative_page_{page+1}.png',dpi=120)
            plt.close(fig)
    assert all(sha(p)==v for p,v in manifest.items())
    record={'figure':str(pdf),'pages':2,'cases':8,'images':16,'selection':'all eight existing calibration cases, original order',
            'new_images_generated':False,'new_classifier_optimized_edits':False,'source_hashes_unchanged':True,'input_sha256':manifest}
    (result/'qualitative_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    statepath=ROOT/'state.yaml';state=yaml.safe_load(statepath.read_text())
    if state.get('current_work',{}).get('task')=='tasks/006_genki_ffhq_classifier.md':
        state['current_work']['qualitative_pdf']='figures/genki_ffhq_classifier/qualitative.pdf'
        statepath.write_text(yaml.safe_dump(state,sort_keys=False))
    print(pdf)


if __name__=='__main__':main()

"""Fix all-beta figure spacing; preserve initial draft and every experiment output."""
import json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/genki_grace_final_figures_mpl')
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1]
result=ROOT/'results/020_genki_grace_identity';figures=ROOT/'figures/genki_grace_identity'
c=json.loads((result/'resolved_config.json').read_text());spec=json.loads((result/'selected_configuration.json').read_text())
raw=Path('/home/xu.zhao/robust_ce_results/genki_grace_identity_v1/20260917_full_v1')
quality=pd.read_csv(result/'saved_image_quality.csv').set_index(['method','factual_index'])
fv=pd.read_csv(result/'future_validity_per_case.csv').groupby(['method','factual_index']).valid.mean()
path=figures/'qualitative_all_betas_v2.pdf';assert not path.exists()
columns=['Factual','Base']+[f'GRACE_beta{b:.2f}' for b in c['beta_grid']]
with PdfPages(path) as pdf:
    for start in range(0,8,2):
        fig,axes=plt.subplots(2,6,figsize=(15,8))
        fig.subplots_adjust(left=.03,right=.98,top=.82,bottom=.14,hspace=.5,wspace=.1)
        fig.suptitle(f'GENKI-head GRACE with fixed content and preservation limits\nSelected radius {spec["r_star"]:g}, selected beta {spec["beta_geo_star"]:g}',fontsize=14,y=.97)
        for row,i in enumerate(range(start,start+2)):
            for col,name in enumerate(columns):
                im=Image.open(raw/'qualitative_examples'/f'test_{i}_{"factual" if name=="Factual" else name}.png').convert('RGB');im.thumbnail((448,448))
                ax=axes[row,col];ax.imshow(im);ax.axis('off')
                if name=='Factual':title=f'Case {i} | Factual'
                else:
                    q=quality.loc[name,i];label='Base' if name=='Base' else f'GRACE beta={float(name.split("beta")[-1]):g}'
                    title=f'{label}\nSmile {q.smile_score:.3f} | FV {fv.loc[name,i]:.0%}'
                    ax.text(.5,-.045,f'ArcFace {q.arcface_similarity:.3f} | LPIPS {q.LPIPS:.3f}',transform=ax.transAxes,ha='center',va='top',fontsize=8)
                ax.set_title(title,fontsize=9,pad=7)
        fig.text(.5,.045,'All eight fixed test cases shown. Scores use saved images; FV averages 30 correlated future-update models.\nFour style channels are editable; all other styles and the input transform stay factual. Human smile labels are unavailable.',ha='center',fontsize=9)
        pdf.savefig(fig,dpi=160)
        if start==0:fig.savefig(figures/'qualitative_all_betas_v2_preview.png',dpi=110)
        plt.close(fig)
report=result/'report.md';text=report.read_text().replace('qualitative_all_betas.pdf','qualitative_all_betas_v2.pdf')
report.write_text(text+'\n\nThe final all-beta figure is v2, with corrected header/footer spacing. The initial draft remains preserved. This layout change does not regenerate images or alter results.\n')
print(path)

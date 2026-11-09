"""Non-destructive identity evaluation with a detector scale validated on factuals.

The original 640px detector missed five of eight tightly cropped factual faces.
320px detected all eight at the unchanged0.5 threshold. Freeze this correction
before remeasuring every method; never choose scale using identity scores.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    dest=out/'identity_detection_audit';dest.mkdir(exist_ok=False)
    protocol={'det_size':[320,320],'det_thresh':.5,'rationale':'640px detection found only3/8 tightly cropped factuals; 320px found8/8 at unchanged threshold; no CF identity scores used to choose scale','original_metrics_preserved':True,'source_images':'saved lossless PNGs, same round-to-uint8 conversion as original metrics','selection':'largest face bounding box for every image','start_unix':time.time()}
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    app=FaceAnalysis(name='buffalo_l',root=str(out/'identity_assets'),allowed_modules=['detection','recognition'],providers=['CPUExecutionProvider'],sess_options=opts)
    app.prepare(ctx_id=-1,det_size=(320,320),det_thresh=.5)
    source=pd.read_csv(out/'image_metrics_per_example.csv');rows=[];embeddings={};detections=[];crops=[]
    for i in range(8):
        methods=['Factual']+source.method.unique().tolist()
        for method in methods:
            img=cv2.imread(str(out/'qualitative_examples'/f'{i}_{method}.png'));faces=app.get(img)
            face=max(faces,key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) if faces else None
            key=f'{i}_{method}'
            if face is not None:
                embeddings[key]=face.normed_embedding
                aligned=norm_crop(img,landmark=face.kps,image_size=112)
                if method in ['Factual','Base','Full_GRACE','Bottleneck_GRACE']:crops.append((i,method,aligned))
            detections.append({'test_index':i,'method':method,'face_count':len(faces),'confidence':float(face.det_score) if face is not None else None,'bbox':face.bbox.tolist() if face is not None else None,'landmarks':face.kps.tolist() if face is not None else None})
        for method in source.method.unique():
            a=embeddings.get(f'{i}_Factual');b=embeddings.get(f'{i}_{method}')
            rows.append({'test_index':i,'method':method,'identity_similarity':float(a@b) if a is not None and b is not None else np.nan})
    np.savez(dest/'embeddings.npz',**embeddings);(dest/'detections.json').write_text(json.dumps(detections,indent=2)+'\n');table=pd.DataFrame(rows);table.to_csv(dest/'identity_per_example.csv',index=False)
    # Independent read-back cosine check.
    stored=np.load(dest/'embeddings.npz')
    for row in table.itertuples():
        if pd.notna(row.identity_similarity):assert abs(float(stored[f'{row.test_index}_Factual']@stored[f'{row.test_index}_{row.method}'])-row.identity_similarity)<1e-7
    canvas=np.zeros((8*112,4*112,3),dtype=np.uint8)
    for i,method,img in crops:canvas[i*112:(i+1)*112,['Factual','Base','Full_GRACE','Bottleneck_GRACE'].index(method)*112:(['Factual','Base','Full_GRACE','Bottleneck_GRACE'].index(method)+1)*112]=img
    cv2.imwrite(str(dest/'alignment_check.png'),canvas)
    summary=[]
    for method,d in table.groupby('method'):
        vals=d.identity_similarity.dropna();summary.append({'method':method,'Identity_Similarity':vals.mean(),'identity_std':vals.std(ddof=0),'identity_p10':vals.quantile(.1),'identity_valid_pairs':len(vals),'identity_missing_pairs':8-len(vals)})
    summary=pd.DataFrame(summary);summary.to_csv(dest/'identity_metrics.csv',index=False)
    old=pd.read_csv(out/'all_results.csv');new=old.drop(columns=[x for x in summary.columns if x!='method']).merge(summary,on='method');new.to_csv(dest/'all_results.csv',index=False)
    main=new.set_index('method').loc[['Base','Full_GRACE','Bottleneck_GRACE']].reset_index();main.to_csv(dest/'full_vs_bottleneck.csv',index=False)
    geo=pd.read_csv(out/'geometry_ablation.csv');geo=geo.drop(columns=[x for x in summary.columns if x!='method' and x in geo.columns]).merge(summary,on='method');geo.to_csv(dest/'geometry_ablation.csv',index=False)
    # Paired identity difference only where both measures exist.
    paired=table.pivot(index='test_index',columns='method',values='identity_similarity');paired['bottleneck_minus_full']=paired.Bottleneck_GRACE-paired.Full_GRACE;paired.to_csv(dest/'identity_paired.csv')
    audit={'status':'passed','all_factuals_detected':all(v['face_count']>0 for v in detections if v['method']=='Factual'),'missing_pairs':int(summary.identity_missing_pairs.sum()),'mean_paired_identity_gain_vs_full':float(paired.bottleneck_minus_full.mean()),'paired_count':int(paired.bottleneck_minus_full.notna().sum()),'wall_seconds':time.time()-protocol['start_unix'],'original_run_review':str(out/'completion_review.json')}
    (dest/'review.json').write_text(json.dumps(audit,indent=2)+'\n')
    repo=ROOT/'results/015_ffhq_identity_v1/identity_detection_audit';repo.mkdir(exist_ok=False)
    for name in ['protocol.json','identity_per_example.csv','identity_metrics.csv','full_vs_bottleneck.csv','all_results.csv','geometry_ablation.csv','identity_paired.csv','review.json','detections.json']: (repo/name).write_bytes((dest/name).read_bytes())
    print(main[['method','Future_Validity','pixel_L2_224','Identity_Similarity','identity_std','identity_p10','identity_valid_pairs','LPIPS_squeeze']].to_string(index=False),flush=True)

if __name__=='__main__':main()

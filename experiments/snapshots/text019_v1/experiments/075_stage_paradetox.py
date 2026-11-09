import hashlib,json,urllib.request
from pathlib import Path
p=Path('/home/xu.zhao/robust_ce_data/text019_assets_v1/paradetox/train.tsv')
assert not p.exists()
u='https://huggingface.co/datasets/s-nlp/paradetox/resolve/958b799a4c3a7c16463632465aaaa8ad083235a8/train.tsv'
with urllib.request.urlopen(u,timeout=180) as r:b=r.read()
p.write_bytes(b)
(p.parent/'data_manifest.json').write_text(json.dumps({'file':str(p),'url':u,'revision':'958b799a4c3a7c16463632465aaaa8ad083235a8','sha256':hashlib.sha256(b).hexdigest()},indent=2)+'\n')
print('dataset bytes',len(b))

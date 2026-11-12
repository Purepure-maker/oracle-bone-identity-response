"""Encode images, retrieve candidates, or evaluate a complete stored gallery."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def put(p,j):Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def encode(a):
 from image_encoder import FrozenEncoder
 from threadpoolctl import threadpool_limits
 import torch,cv2,sys
 records=read(a.manifest);root=Path(a.image_root).resolve();out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
 if not records:raise ValueError('Manifest is empty')
 records=[{'path':r,'key':r} if isinstance(r,str) else r for r in records]
 keys=[r.get('key',r['path']) for r in records]
 if len(set(keys))!=len(keys):raise ValueError('Image keys must be unique')
 for r in records:
  p=Path(r['path'])
  if p.is_absolute() or '..' in p.parts:raise ValueError('Manifest paths must stay under image-root')
 encoder=FrozenEncoder(a.assets,a.device,a.vocabulary);n=len(records);start=time.monotonic()
 raw=np.lib.format.open_memmap(out/'RAW640.npy',mode='w+',dtype=np.float32,shape=(n,640));geo=np.lib.format.open_memmap(out/'GEO640.npy',mode='w+',dtype=np.float32,shape=(n,640))
 identity=np.lib.format.open_memmap(out/'ID.npy',mode='w+',dtype=np.float32,shape=(n,len(encoder.class_order)))
 observed=[]
 with threadpool_limits(limits=1):
  for first in range(0,n,8):
   batch=[]
   for r in records[first:first+8]:
    data=(root/r['path']).read_bytes();h=hashlib.sha256(data).hexdigest()
    if r.get('sha256') and r['sha256']!=h:raise ValueError('Image SHA mismatch: '+r['path'])
    batch.append(data);observed.append(h)
   rr,gg=encoder.encode_bytes(batch);raw[first:first+len(batch)]=rr;geo[first:first+len(batch)]=gg
   if a.vocabulary=='obc':identity[first:first+len(batch)]=encoder.responses(gg)
   if first==0 or first%4096==0:print(json.dumps({'phase':'encoding','images':first+len(batch),'total':n}),flush=True)
  if a.vocabulary=='hwobc':
   for first in range(0,n,256):identity[first:first+256]=encoder.responses(geo[first:first+256])
 encoder.verify_unchanged();raw.flush();geo.flush();identity.flush()
 assert np.isfinite(raw).all() and np.isfinite(geo).all() and np.isfinite(identity).all()
 put(out/'paths.json',keys);put(out/'class_order.json',encoder.class_order);put(out/'image_sha256.json',observed)
 put(out/'complete.json',{'state':'COMPLETE','images':n,'vocabulary':a.vocabulary,'dimensions':len(encoder.class_order),'query_identity_label_used':False,'frozen_state_unchanged':True,'model_state_sha256':encoder.initial_state,'mapping_sha256':encoder.mapping_sha256,'manifest_sha256':sha(a.manifest),'seconds':time.monotonic()-start,'runtime':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'cv2':cv2.__version__,'device':str(encoder.device)},'arrays_sha256':{p.name:sha(p) for p in out.glob('*.npy')},'numeric_routes':{'image_batch':8,'response_batch':8 if a.vocabulary=='obc' else 256,'effective_response_temperature':.025,'recorded_finite_precision_route':encoder.manifest['OBC_probability_route' if a.vocabulary=='obc' else 'HWOBC_probability_route']}})
 print(json.dumps({'state':'COMPLETE','output':str(out),'images':n}))

def retrieve(a):
 from scipy.spatial.distance import cdist
 if a.top_k<1:raise ValueError('top-k must be positive')
 Q=Path(a.queries);G=Path(a.gallery);assert read(Q/'complete.json')['state']==read(G/'complete.json')['state']=='COMPLETE'
 assert read(Q/'class_order.json')==read(G/'class_order.json'),'Vocabulary order mismatch'
 assert read(Q/'complete.json')['model_state_sha256']==read(G/'complete.json')['model_state_sha256'],'Image encoders differ'
 assert read(Q/'complete.json')['mapping_sha256']==read(G/'complete.json')['mapping_sha256'],'Calibration or prototype mapping differs; re-encode under one mapping'
 for folder in [Q,G]:assert sha(folder/'ID.npy')==read(folder/'complete.json')['arrays_sha256']['ID.npy'],'Stored descriptor hash mismatch'
 qp=read(Q/'paths.json');gp=read(G/'paths.json');q=np.load(Q/'ID.npy',mmap_mode='r');g=np.load(G/'ID.npy',mmap_mode='r')
 def unit(z):
  z=np.asarray(z,dtype=np.float64);norm=np.linalg.norm(z,axis=1,keepdims=True)
  if not np.isfinite(z).all() or (norm<=0).any():raise ValueError('Invalid descriptor')
  return z/norm
 order=np.array(sorted(range(len(gp)),key=lambda i:hashlib.sha256(('A-final-v7|'+gp[i]).encode()).digest()));gallery=unit(g)[order];ordered=[gp[i] for i in order];lookup={p:i for i,p in enumerate(ordered)};result=[]
 for first in range(0,len(qp),16):
  distances=cdist(unit(q[first:first+16]),gallery,metric='sqeuclidean')
  for j,d in enumerate(distances):
   key=qp[first+j]
   if key in lookup:d[lookup[key]]=np.inf
   chosen=np.argsort(d,kind='stable');chosen=[int(i) for i in chosen[:a.top_k] if np.isfinite(d[i])]
   result.append({'query_key':key,'candidates':[{'key':ordered[i],'squared_distance':float(d[i])} for i in chosen]})
 target=Path(a.output)
 if target.exists():raise FileExistsError(target)
 put(target,{'queries':result,'query_identity_label_used':False,'self_exclusion':'equal image key','display_ties':'fixed path-hash order; scientific AP/Hit expectations are handled by evaluate'})
 print(json.dumps({'state':'COMPLETE','queries':len(result),'output':str(target)}))

def evaluate(a):
 from evaluation import evaluate as score
 p=Path(a.gallery);assert read(p/'complete.json')['state']=='COMPLETE'
 file='ID.npy' if a.readout=='identity' else 'GEO640.npy'
 assert sha(p/file)==read(p/'complete.json')['arrays_sha256'][file],'Stored descriptor hash mismatch'
 m,_=score(np.load(p/file,mmap_mode='r'),read(p/'paths.json'),a.output,workers=a.workers)
 print(json.dumps(m,indent=2))

def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
 q=s.add_parser('encode');q.add_argument('--assets',type=Path,default=Path(__file__).resolve().parent/'assets');q.add_argument('--image-root',required=True);q.add_argument('--manifest',required=True);q.add_argument('--vocabulary',choices=['obc','hwobc'],default='obc');q.add_argument('--device',default='cpu');q.add_argument('--output',required=True);q.set_defaults(run=encode)
 q=s.add_parser('retrieve');q.add_argument('--queries',required=True);q.add_argument('--gallery',required=True);q.add_argument('--top-k',type=int,default=100);q.add_argument('--output',required=True);q.set_defaults(run=retrieve)
 q=s.add_parser('evaluate');q.add_argument('--gallery',required=True);q.add_argument('--readout',choices=['visual','identity'],default='identity');q.add_argument('--workers',type=int,default=4);q.add_argument('--output',required=True);q.set_defaults(run=evaluate)
 a=p.parse_args();a.run(a)
if __name__=='__main__':main()

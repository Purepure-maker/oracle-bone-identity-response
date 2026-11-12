"""Common label-independent tie handling and FP64 direct-distance retrieval."""
import hashlib,json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial.distance import cdist

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normalize(x):
 z=np.asarray(x,dtype=np.float64);n=np.linalg.norm(z,axis=1,keepdims=True)
 assert np.isfinite(z).all() and (n>0).all()
 return z/n
def paired(a,b,paths,draws=5000):
 classes=sorted({p.split('/')[0] for p in paths});labels=np.array([p.split('/')[0] for p in paths])
 delta=np.array([np.mean([r['AP'] for r in a if r['class_id']==c])-np.mean([r['AP'] for r in b if r['class_id']==c]) for c in classes])
 n=np.array([np.sum(labels==c) for c in classes]);ix=np.random.default_rng(20261031).integers(len(classes),size=(draws,len(classes)))
 return {'Class_delta_pp':float(delta.mean()*100),'Query_delta_pp':float(np.average(delta,weights=n)*100),
 'Class_CI95_pp':(np.quantile(delta[ix].mean(1),[.025,.975])*100).tolist(),
 'Query_CI95_pp':(np.quantile((delta[ix]*n[ix]).sum(1)/n[ix].sum(1),[.025,.975])*100).tolist(),
 'improved_classes':int((delta>1e-12).sum()),'damaged_classes':int((delta< -1e-12).sum())}

def evaluate(features,paths,out,workers=4,hard_pairs=None):
 """AP/Hit1 main values are exact-tie expectations; display uses hash order."""
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 assert len(paths)==len(set(paths))==len(features)
 z=normalize(features);labels=np.array([p.split('/')[0] for p in paths]);n=len(z)
 order0=np.array(sorted(range(n),key=lambda j:hashlib.sha256(('A-final-v7|'+paths[j]).encode()).digest()))
 gallery=z[order0];gy=labels[order0];gp=[paths[j] for j in order0];inverse=np.empty(n,int);inverse[order0]=np.arange(n)
 selected=set(np.random.default_rng(20261031).choice(n,min(16,n),replace=False).tolist())
 lookup={p:i for i,p in enumerate(paths)};hpby=defaultdict(list)
 for h in hard_pairs or []:
  if h['query_path'] in lookup and h['candidate_path'] in lookup:hpby[lookup[h['query_path']]].append(h['candidate_path'])
 rows=[];checks=[];hp=[];top=[]
 with ThreadPoolExecutor(max_workers=workers) as pool:
  for start in range(0,n,64):
   starts=list(range(start,min(n,start+64),16))
   distances=np.concatenate(list(pool.map(lambda i:cdist(z[i:min(i+16,n)],gallery,metric='sqeuclidean'),starts)))
   for off,dd in enumerate(distances):
    i=start+off;dd[inverse[i]]=np.inf
    order=np.argsort(dd,kind='stable')[:-1];dist=dd[order];truth=gy[order]==labels[i]
    positions=np.flatnonzero(truth)+1;R=len(positions)
    if R==0:continue
    ap=float(np.mean(np.arange(1,R+1)/positions));expected=ap;hit=float(truth[0]);expectedhit=hit
    bounds=np.r_[0,np.flatnonzero(dist[1:]!=dist[:-1])+1,len(dist)];sizes=np.diff(bounds)
    positives=np.add.reduceat(truth.astype(np.int64),bounds[:-1]);before=np.cumsum(positives)-positives
    mixed=np.flatnonzero((positives>0)&(positives<sizes))
    if len(mixed):
     con=np.zeros(len(dist));con[positions-1]=np.arange(1,R+1)/positions
     sums=np.add.reduceat(con,bounds[:-1]);extra=0.
     for j in mixed:
      a=int(before[j]);m=int(sizes[j]);r=int(positives[j]);k=np.arange(1,m+1,dtype=np.float64)
      extra+=float(np.sum(((a+1)*r/m+(k-1)*r*(r-1)/(m*(m-1)))/(bounds[j]+k)))-sums[j]
     expected+=extra/R
    expectedhit=float(positives[0]/sizes[0])
    r={'path':paths[i],'class_id':str(labels[i]),'AP':expected,'Hit1':expectedhit,'AP_hash':ap,'Hit1_hash':hit,
       'MRR10_hash':float(1/positions[0] if positions[0]<=10 else 0),
       'P10_hash':float(truth[:10].sum()/10),'R10_hash':float(truth[:10].sum()/R),'R100_hash':float(truth[:100].sum()/R),
       'mixed_top_tie':bool(0<positives[0]<sizes[0]),'max_tie_block':int(sizes.max())}
    rows.append(r)
    if i in selected:
     scalar=np.sum((gallery-z[i])**2,axis=1);scalar[inverse[i]]=np.inf
     valid=np.isfinite(scalar);err=float(abs(scalar[valid]-dd[valid]).max());assert err<1e-12
     pure=sorted((j for j in range(n) if j!=inverse[i]),key=lambda j:(float(scalar[j]),hashlib.sha256(('A-final-v7|'+gp[j]).encode()).digest()))
     pp=[k for k,j in enumerate(pure,1) if gy[j]==labels[i]]
     ae=abs(sum(k/rank for k,rank in enumerate(pp,1))/len(pp)-ap);assert ae<1e-12
     checks.append({'path':paths[i],'distance_max_error':err,'AP_hash_error':ae})
    if i in hpby:
     inv={gp[j]:k+1 for k,j in enumerate(order)}
     for p in hpby[i]:hp.append({'query_path':paths[i],'candidate_path':p,'rank_hash':inv[p]})
    top.append({'path':paths[i],'top5':[gp[j] for j in order[:5]],'first_positive_rank_hash':int(positions[0])})
   if start%8192==0:print('RANK',out.name,start,n,flush=True)
 classes=sorted({r['class_id'] for r in rows});group={c:[r for r in rows if r['class_id']==c] for c in classes}
 keys=['AP','Hit1','AP_hash','Hit1_hash','MRR10_hash','P10_hash','R10_hash','R100_hash']
 pc={c:{k:float(np.mean([r[k] for r in rr])) for k in keys} for c,rr in group.items()}
 metrics={}
 for k in keys:
  metrics['Query_'+k]=float(np.mean([r[k] for r in rows]));metrics['Class_'+k]=float(np.mean([r[k] for r in pc.values()]))
 metrics.update(queries=len(rows),gallery_images=n,excluded_no_positive_queries=n-len(rows),classes=len(classes),dimensions=z.shape[1],mixed_top_tie_queries=sum(r['mixed_top_tie'] for r in rows),
  max_tie_block=max(r['max_tie_block'] for r in rows),hard_positive_Top100_hash=sum(r['rank_hash']<=100 for r in hp))
 for name,value in [('per_query',rows),('per_class',pc),('metrics',metrics),('independent_checks',checks),('top5',top),('hard_pairs',hp)]:dump(out/(name+'.json'),value)
 return metrics,rows

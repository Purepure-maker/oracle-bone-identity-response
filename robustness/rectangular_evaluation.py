"""FP64 complete clean-gallery evaluation of independently transformed queries."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,itertools
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

def unit(x):
    x=np.asarray(x,dtype=np.float64)
    norms=np.linalg.norm(x,axis=1,keepdims=True)
    if not np.isfinite(x).all() or (norms<=0).any():raise ValueError('Finite nonzero descriptors required')
    return x/norms

def expected_scores(distances,relevance):
    order=np.argsort(distances,kind='stable')
    order=order[np.isfinite(distances[order])]
    dist=distances[order];truth=relevance[order]
    positions=np.flatnonzero(truth)+1;R=len(positions)
    if R==0:return None
    ap_hash=float(np.mean(np.arange(1,R+1)/positions));ap=ap_hash
    bounds=np.r_[0,np.flatnonzero(dist[1:]!=dist[:-1])+1,len(dist)]
    sizes=np.diff(bounds);positive=np.add.reduceat(truth.astype(np.int64),bounds[:-1]);before=np.cumsum(positive)-positive
    mixed=np.flatnonzero((positive>0)&(positive<sizes))
    if len(mixed):
        contribution=np.zeros(len(dist));contribution[positions-1]=np.arange(1,R+1)/positions
        sums=np.add.reduceat(contribution,bounds[:-1]);extra=0.
        for j in mixed:
            a=int(before[j]);m=int(sizes[j]);r=int(positive[j]);k=np.arange(1,m+1,dtype=np.float64)
            extra+=float(np.sum(((a+1)*r/m+(k-1)*r*(r-1)/(m*(m-1)))/(bounds[j]+k)))-sums[j]
        ap+=extra/R
    return {'AP':ap,'Hit1':float(positive[0]/sizes[0]),'AP_hash':ap_hash,'Hit1_hash':float(truth[0]),'first_positive_rank':int(positions[0]),'first_positive_index':int(order[positions[0]-1]),'top5_indices':order[:5].tolist(),'mixed_top_tie':bool(0<positive[0]<sizes[0]),'max_tie_block':int(sizes.max())}

def score(query,gallery,query_keys,gallery_keys,workers=4,progress=None):
    if len(set(query_keys))!=len(query_keys) or len(set(gallery_keys))!=len(gallery_keys):raise ValueError('Duplicate image keys')
    q=unit(query);g=unit(gallery)
    order=np.array(sorted(range(len(gallery_keys)),key=lambda i:hashlib.sha256(('A-final-v7|'+gallery_keys[i]).encode()).digest()))
    g=g[order];keys=[gallery_keys[i] for i in order];labels=np.array([k.split('/')[0] for k in keys]);lookup={k:i for i,k in enumerate(keys)}
    rows=[];checks=[]
    def block(first):return first,cdist(q[first:first+16],g,metric='sqeuclidean')
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0,len(q),64):
            starts=range(offset,min(offset+64,len(q)),16)
            for first,distances in pool.map(block,starts):
                for off,d in enumerate(distances):
                    index=first+off;key=query_keys[index];identity=key.split('/')[0]
                    if key not in lookup:raise ValueError('Query origin missing from fixed clean gallery')
                    d[lookup[key]]=np.inf
                    values=expected_scores(d,labels==identity)
                    if values is None:raise ValueError('Panel contains a query with no positive')
                    values['first_positive_key']=keys[values.pop('first_positive_index')]
                    values['top5']=[keys[j] for j in values.pop('top5_indices')]
                    values.update(key=key,class_id=identity)
                    assert key not in values['top5']
                    if index<8:
                        scalar=np.sum((g-q[index])**2,axis=1);scalar[lookup[key]]=np.inf
                        finite=np.isfinite(scalar);error=float(np.max(np.abs(scalar[finite]-d[finite])))
                        assert error<1e-12,error
                        other=expected_scores(scalar,labels==identity)
                        assert abs(other['AP_hash']-values['AP_hash'])<1e-12
                        checks.append({'key':key,'scalar_distance_max_error':error,'self_excluded':True})
                    rows.append(values)
            if progress:progress(min(offset+64,len(q)),len(q))
    groups=defaultdict(list)
    for row in rows:groups[row['class_id']].append(row)
    perclass={c:{k:float(np.mean([r[k] for r in rr])) for k in ['AP','Hit1','AP_hash','Hit1_hash']} for c,rr in groups.items()}
    metrics={prefix+'_'+key:float(np.mean([row[key] for row in records])) for prefix,records in [('PanelQuery',rows),('Class',list(perclass.values()))] for key in ['AP','Hit1','AP_hash','Hit1_hash']}
    metrics.update(queries=len(rows),classes=len(groups),gallery_images=len(g),self_exclusion='clean original image key',query_scope='fixed class-stratified query panel, not all Val queries')
    return metrics,rows,perclass,checks

def self_test():
    # Exhaustive expectation across all permutations of a mixed exact-distance tie.
    ap=[]
    for order in itertools.permutations([1,1,0,0]):
        p=np.flatnonzero(order)+1;ap.append(np.mean(np.arange(1,3)/p))
    got=expected_scores(np.zeros(4),np.array([1,1,0,0],bool))
    assert abs(got['AP']-np.mean(ap))<1e-12 and got['Hit1']==.5
    # A clean query origin must be excluded even when its transformed vector changes.
    m,rows,_,_=score(np.array([[.99,.01]]),np.array([[1.,0.],[.98,.02],[0.,1.]]),['a/self'],['a/self','a/positive','b/negative'],workers=1)
    assert rows[0]['top5']==['a/positive','b/negative'] and m['Class_AP']==1.
    return {'exhaustive_tie_expectation':True,'transformed_query_self_exclusion':True,'scalar_FP64_distance_checks':True}

if __name__=='__main__':print(json.dumps(self_test()))

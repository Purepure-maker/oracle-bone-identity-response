"""Exercise Train-only prototypes and independent queries with synthetic features."""
from pathlib import Path
import argparse,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'feature_level'))
from reproduce import normalize,prototypes,responses
from geometry import statistics,whitening,fixed_global_features
from evaluation import evaluate

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(42)
    centers=normalize(rng.normal(size=(4,640)))
    def draw(repeats):
        labels=np.repeat(np.arange(4),repeats)
        raw=centers[labels]+0.01*rng.normal(size=(len(labels),640))
        weighted=np.concatenate([np.sqrt(.75)*normalize(raw[:,:512]),.5*normalize(raw[:,512:])],axis=1).astype(np.float32)
        return weighted,[f'identity_{c:03d}' for c in labels]
    # Statistics, supports, gallery and independent query are distinct synthetic draws.
    train,train_labels=draw(5)
    center,cov,_,_=statistics(train,train_labels)
    cov[:512,512:]=0;cov[512:,:512]=0
    matrix,_=whitening(cov)
    support,support_labels=draw(3)
    means,order,counts=prototypes(fixed_global_features(support,center,matrix),support_labels)
    gallery,labels=draw(3)
    gallery_id=responses(fixed_global_features(gallery,center,matrix),means)
    keys=[f'{c}/gallery_{i:03d}' for i,c in enumerate(labels)]
    metrics,_=evaluate(gallery_id,keys,args.output/'identity_metrics',workers=1)
    queries,query_labels=draw(1)
    query_id=responses(fixed_global_features(queries,center,matrix),means)
    distances=np.sum((normalize(query_id)[:,None,:]-normalize(gallery_id)[None,:,:])**2,axis=2)
    ranked=np.argsort(distances,axis=1,kind='stable')[:,:3]
    report={'data':'synthetic functional demonstration; not a paper result','statistics_images':len(train),'prototype_support_images':len(support),'gallery_images':len(gallery),'independently_encoded_queries':len(queries),'query_identity_label_used_by_response_map':False,'class_order':order.tolist(),'support_counts':counts.tolist(),'metrics':metrics,'query_candidates':[{'query_key':f'query_{i:03d}','candidates':[keys[k] for k in row]} for i,row in enumerate(ranked)]}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'state':'PASS','output':str(args.output),'queries':len(queries),'gallery_images':len(gallery)}))

if __name__=='__main__':main()

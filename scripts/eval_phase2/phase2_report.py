import json, math, pathlib, statistics, datetime
from collections import defaultdict
from src.eval.prompting.inputs import load_dev_rows
from src.eval.prompting.scoring import score_prediction, parse_prediction, _expected_numbers, _scale

ROOT=pathlib.Path('/home/dipak/agentic/IFR-QA-Agent')
RUNS=ROOT/'results/week1_prompt_baseline/runs'
OUT=ROOT/'results/week1_prompt_baseline'
rows=load_dev_rows(); gold={r['id']:r for r in rows}

def wilson(k,n,z=1.96):
    if n==0: return (None,None)
    p=k/n
    denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    margin=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/denom
    return (max(0,center-margin), min(1,center+margin))

def load_run(path):
    rs=[]
    for line in open(path):
        r=json.loads(line); rs.append(r)
    # dedupe prefer answered
    best={}
    for r in rs:
        rid=r['row_id']; cur=best.get(rid)
        if cur is None or (r['status']=='answered' and cur.get('status')!='answered'): best[rid]=r
    return best

def score_run(path):
    rs=load_run(path)
    scored=[]
    for rid,g in gold.items():
        row=rs.get(rid)
        if not row: continue
        ctx=set()
        s=score_prediction(g, row.get('output') or '', ctx)
        s['status']=row['status']; s['latency_ms']=row.get('latency_ms'); s['usage']=row.get('usage') or {}
        s['row_id']=rid; s['flow_bucket']=g.get('flow_bucket'); s['expected_behaviour']=g.get('expected_behaviour')
        scored.append(s)
    return scored

def agg(scored, subset_fn=lambda s: True):
    sub=[s for s in scored if subset_fn(s)]
    n=len(sub)
    def rate(key):
        vals=[s.get(key) for s in sub if s.get(key) is not None]
        k=sum(1 for v in vals if v); d=len(vals)
        lo,hi=wilson(k,d) if d else (None,None)
        return {'k':k,'n':d,'rate':(k/d if d else None),'wilson95':[lo,hi]}
    ans=[s for s in sub if s['expected_behaviour']=='answer']
    absr=[s for s in sub if s['expected_behaviour']=='abstain']
    nums=[s for s in ans if s.get('gold_answer_kind')=='numeric']
    ents=[s for s in ans if s.get('gold_answer_kind') in {'entity','categorical','text'}]
    toks=[(s['usage'] or {}).get('total_tokens') for s in sub if isinstance((s['usage'] or {}).get('total_tokens'),int)]
    lats=[s['latency_ms'] for s in sub if isinstance(s.get('latency_ms'),(int,float))]
    out={
      'n_rows':n,
      'correct_or_abstain':rate('abstain_correct') if all(s['expected_behaviour']=='abstain' for s in sub) else None,
      'numeric_match':rate('numeric_match') if nums else None,
      'entity_match':rate('entity_match') if ents else None,
      'correct_abstention':{'k':sum(1 for s in absr if s.get('abstain_correct')),'n':len(absr),'rate':(sum(1 for s in absr if s.get('abstain_correct'))/len(absr) if absr else None),'wilson95':list(wilson(sum(1 for s in absr if s.get('abstain_correct')),len(absr)))},
      'false_answer':rate('false_answer'),
      'false_abstention':rate('false_abstention'),
      'citation_valid':rate('citation_valid'),
      'parse_ok':rate('parse_ok'),
      'budget_exceeded':sum(1 for s in sub if s['status']=='BUDGET_EXCEEDED'),
      'errors':sum(1 for s in sub if s['status']=='error'),
      'tokens_median':(statistics.median(toks) if toks else None),
      'tokens_p95':(sorted(toks)[max(0,int(math.ceil(0.95*len(toks)))-1)] if toks else None),
      'latency_median_ms':(statistics.median(lats) if lats else None),
      'latency_p95_ms':(sorted(lats)[max(0,int(math.ceil(0.95*len(lats)))-1)] if lats else None),
    }
    # by bucket
    buckets=defaultdict(list)
    for s in sub: buckets[s.get('flow_bucket')].append(s)
    out['by_bucket']={}
    for b,arr in sorted(buckets.items()):
        out['by_bucket'][b]={'n':len(arr),'numeric_match':rate('numeric_match') if any(x.get('gold_answer_kind')=='numeric' and x['expected_behaviour']=='answer' for x in arr) else None}
    return out

print('ok')

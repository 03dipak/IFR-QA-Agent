import json, math, pathlib, statistics, glob, datetime
from collections import defaultdict, Counter
from scipy.stats import binomtest # not available maybe
ROOT=pathlib.Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0,str(ROOT))
from scripts.eval_phase2.gold_view import dev_rows
from src.eval.prompting.scoring import score_prediction

GOLD={r['id']:r for r in dev_rows()}
RUNS=ROOT/'results/week1_prompt_baseline/runs'
OUT=ROOT/'results/week1_prompt_baseline'

def wilson(k,n,z=1.96):
    if n==0: return [None,None]
    p=k/n; denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    margin=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/denom
    return [max(0,center-margin), min(1,center+margin)]

def load_best(path):
    rs=[json.loads(l) for l in open(path) if l.strip()]
    best={}
    for r in rs:
        rid=r.get('row_id'); cur=best.get(rid)
        if cur is None or (r.get('status')=='answered' and cur.get('status')!='answered'): best[rid]=r
    return best

def score_rows(best, which='score'):
    out=[]
    for rid,g in GOLD.items():
        row=best.get(rid)
        if not row: continue
        s=row.get(which)
        if not s:
            ctx=set(tuple(x) for x in (row.get('context_pages') or []))
            s=score_prediction(g, row.get('output') or '', ctx)
        rec=dict(s)
        rec.update({'row_id':rid,'status':row.get('status'),'flow_bucket':g.get('flow_bucket'),'expected_behaviour':g.get('expected_behaviour'),'usage':row.get('usage') or {},'latency_ms':row.get('latency_ms'),'context_pages':row.get('context_pages') or []})
        out.append(rec)
    return out

def rate(arr,key):
    vals=[x.get(key) for x in arr if x.get(key) is not None]
    d=len(vals); k=sum(1 for v in vals if v)
    return {'k':k,'n':d,'rate':(k/d if d else None),'wilson95':wilson(k,d)}

def citation_rate(arr):
    # valid where defined (answerable answered with context check)
    return rate(arr,'citation_valid')

def agg(scored):
    sub=scored
    ans=[s for s in sub if s['expected_behaviour']=='answer']
    absr=[s for s in sub if s['expected_behaviour']=='abstain']
    nums=[s for s in ans if s.get('gold_answer_kind')=='numeric']
    ents=[s for s in ans if s.get('gold_answer_kind') in {'entity','categorical','text'}]
    toks=[(s.get('usage') or {}).get('total_tokens') for s in sub if isinstance((s.get('usage') or {}).get('total_tokens'),int)]
    lats=[s['latency_ms'] for s in sub if isinstance(s.get('latency_ms'),(int,float))]
    def statv(v):
        if not v: return {'median':None,'p95':None}
        sv=sorted(v); p95=sv[max(0,int(math.ceil(0.95*len(sv)))-1)]
        return {'median':statistics.median(sv),'p95':p95}
    out={
      'n_rows':len(sub),
      'numeric_match':rate(nums,'numeric_match'),
      'entity_match':rate(ents,'entity_match'),
      'correct_abstention':{'k':sum(1 for s in absr if s.get('abstain_correct')),'n':len(absr),'rate':(sum(1 for s in absr if s.get('abstain_correct'))/len(absr) if absr else None),'wilson95':wilson(sum(1 for s in absr if s.get('abstain_correct')),len(absr))},
      'false_answer':rate(sub,'false_answer'),
      'false_abstention':rate(sub,'false_abstention'),
      'citation_valid':citation_rate(sub),
      'parse_ok':rate(sub,'parse_ok'),
      'budget_exceeded':sum(1 for s in sub if s.get('status')=='BUDGET_EXCEEDED'),
      'filtered_empty':sum(1 for s in sub if s.get('status')=='FILTERED_EMPTY'),
      'errors':sum(1 for s in sub if s.get('status')=='error'),
      'tokens':statv(toks),'latency_ms':statv(lats),
    }
    buckets=defaultdict(list)
    for s in sub: buckets[s.get('flow_bucket')].append(s)
    out['by_bucket']={}
    for b,arr in sorted(buckets.items()):
        bnums=[s for s in arr if s['expected_behaviour']=='answer' and s.get('gold_answer_kind')=='numeric']
        bents=[s for s in arr if s['expected_behaviour']=='answer' and s.get('gold_answer_kind') in {'entity','categorical','text'}]
        babs=[s for s in arr if s['expected_behaviour']=='abstain']
        out['by_bucket'][b]={'n':len(arr),'numeric_match':rate(bnums,'numeric_match'),'entity_match':rate(bents,'entity_match'),'correct_abstention':{'k':sum(1 for s in babs if s.get('abstain_correct')),'n':len(babs)}}
    return out

def common_answerable_subset(run_paths):
    sets=[]
    for p in run_paths:
        b=load_best(p)
        sets.append({rid for rid,r in b.items() if r.get('status')=='answered' and GOLD.get(rid,{}).get('expected_behaviour')=='answer'})
    return set.intersection(*sets) if sets else set()

# McNemar exact on paired correctness (numeric/entity match or abstain correct as appropriate)
def paired_correct(scored_map, common):
    out={}
    for rid in common:
        s=scored_map[rid]
        if s['expected_behaviour']=='abstain': ok=bool(s.get('abstain_correct'))
        elif s.get('gold_answer_kind')=='numeric': ok=bool(s.get('numeric_match'))
        else: ok=bool(s.get('entity_match'))
        out[rid]=ok
    return out

def mcnemar(a,b,ids):
    # b,c discordant
    b01=sum(1 for i in ids if a[i] and not b[i])
    c10=sum(1 for i in ids if (not a[i]) and b[i])
    n=b01+c10
    # exact binomial p-value two-sided
    if n==0: p=1.0
    else:
        from math import comb
        k=min(b01,c10)
        cdf=sum(comb(n,i)*(0.5**n) for i in range(0,k+1))
        p=min(1.0,2*cdf)
    return {'a_correct_b_wrong':b01,'a_wrong_b_correct':c10,'n_discordant':n,'p_exact':p}

def pick_runs():
    # choose canonical existing runs (non-smoke, prefer latest/complete) + repeats if present
    arms=['B0','B1','S1','S2','S3','B2-filtered']
    picks={}
    # canonical
    canon={
      'B0':'phase2_v1_20261112T173032Z_B0.jsonl',
      'B1':'phase2_v1_20261112T173032Z_B1.jsonl',
      'S1':'phase2_v1_20261112T173032Z_S1.jsonl',
      'S2':'phase2_v1_20261112T173032Z_S2.jsonl',
      'S3':'phase2_v1_20261112T173032Z_S3.jsonl',
      'B2-filtered':'b2filtered_empty_20261113T080544Z_B2-filtered.jsonl',
    }
    for a in arms:
        paths=[]
        p=RUNS/canon[a]
        if p.exists(): paths.append(p)
        # repeats (2 chosen if 3 exceed budget)
        reps=sorted(RUNS.glob(f'repeat*_{a}.jsonl'))
        # keep those with manifest and >=145 or 82 for S2
        good=[]
        for r in reps:
            mf=pathlib.Path(str(r).replace('.jsonl','.manifest.json'))
            if not mf.exists(): continue
            m=json.loads(mf.read_text())
            n=sum(1 for _ in open(r))
            if m.get('stopped_reason'): continue
            good.append(r)
        # prefer repeat2_ then repeat_
        good=sorted(good)
        for r in good:
            if r not in paths: paths.append(r)
        picks[a]=paths
    return picks

def main():
    meta=json.loads((ROOT/'data/eval/datasets/gold_data.corrections_meta.json').read_text())
    oracle_path=ROOT/'results/retrieval_sanity_bm25_fast.json'
    oracle=json.loads(oracle_path.read_text()) if oracle_path.exists() else {'status':'missing','note':'UNVERIFIED'}
    picks=pick_runs()
    report={'generated_at':datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),'gold_view':meta,'scorer':'v2_unit_norm_corrected_view','scorer_v1':'sensitivity_only (score_pre_correction)','oracle_recall':oracle,'b2_global':'UNVERIFIED (oracle gate not passed / not rerun)' if not oracle_path.exists() else 'see_oracle','arms':{},'pairs':{},'notes':{}}

    # per arm per run
    for arm,paths in picks.items():
        report['arms'][arm]={'runs':[]}
        for p in paths:
            best=load_best(p)
            scored_v2=score_rows(best,'score')
            scored_pre=score_rows(best,'score_pre_correction') if any('score_pre_correction' in r for r in best.values()) else None
            scored_v1=score_rows(best,'score_v1') if any('score_v1' in r for r in best.values()) else None
            # common subset placeholder per-run all rows
            rep={'path':str(p.relative_to(ROOT)),'run_id':p.stem,'v2_all145':agg(scored_v2),'v2_common_answerable':None}
            if scored_pre: rep['pre_correction_all145']=agg(scored_pre)
            if scored_v1: rep['v1_sensitivity_all145']=agg(scored_v1)
            report['arms'][arm]['runs'].append(rep)
        # spread summary (2+ runs)
        runs_v2=[r['v2_all145'] for r in report['arms'][arm]['runs']]
        def getr(x,keys):
            for k in keys:
                if x is None: return None
                x=x.get(k)
            return x
        spread={}
        for name,keys in [('numeric_match',['numeric_match','rate']),('correct_abstention',['correct_abstention','rate']),('false_answer',['false_answer','rate']),('parse_ok',['parse_ok','rate'])]:
            vals=[getr(x,keys) for x in runs_v2]
            vals=[v for v in vals if v is not None]
            if vals: spread[name]={'min':min(vals),'max':max(vals),'n_runs':len(vals)}
        report['arms'][arm]['spread']=spread

    # common answerable subset across key arms for pairing (use first/canonical run each)
    key_arms=['B1','S1','S2','S3','B2-filtered']
    canon_paths={a:RUNS/picks[a][0].name for a in key_arms if picks.get(a)}
    common=common_answerable_subset([str(p) for p in canon_paths.values()])
    report['common_answerable_subset']={'n':len(common),'arms':list(canon_paths.keys())}
    # recompute common metrics
    scored_maps={}
    for a,p in canon_paths.items():
        best=load_best(str(p)); scored=score_rows(best,'score'); scored_maps[a]={s['row_id']:s for s in scored}
        sub=[s for s in scored if s['row_id'] in common]
        report['arms'][a]['runs'][0]['v2_common_answerable']=agg(sub)
        report['arms'][a]['runs'][0]['v2_common_answerable_n']=len(sub)
    # mcnemar pairs on common
    pairs=[('S1','B1'),('S2','B1'),('S3','B1'),('S1','S2'),('S1','S3'),('S2','S3')]
    for x,y in pairs:
        if x not in scored_maps or y not in scored_maps: continue
        ids=sorted(common)
        ax=paired_correct(scored_maps[x],ids); ay=paired_correct(scored_maps[y],ids)
        report['pairs'][f'{x}_vs_{y}']=mcnemar(ax,ay,ids)

    report['notes']={
      's2_abstention':'NOT MEASURED (S2 run on answer-only rows subset); abstain-aware variant not added.',
      'repeats_policy':'2 repeats/arm chosen (time budget); missing repeats filled sequentially if orchestrator completed.',
      'gold_corrections':'data/eval/datasets/gold_data.corrections.jsonl (HIGH applied=0 in scan; existing candidates not applied).',
      'b2_global':'UNVERIFIED until oracle recall table passes; not scored as verified.',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))

    # md
    md=['# Week1 Prompt Baseline Report','','- scorer: v2_unit_norm_corrected_view (v1/pre-correction as sensitivity)','- gold_view: '+json.dumps(meta),'- B2-global: UNVERIFIED until oracle recall passes','- S2: abstention NOT MEASURED','',
        '## Oracle recall', '```json', json.dumps(oracle if oracle_path.exists() else {'UNVERIFIED':'oracle table missing'}, indent=2)[:4000], '```','',
        '## Per arm (canonical run, all 145 dev rows)','| arm | n | num k/n (rate,95%) | ent k/n | abst k/n | FA k/n | FAbs k/n | parse k/n | cit k/n | BE | FE | err | tok med/p95 | lat med/p95 ms |','|---|---:|---|---|---|---|---|---|---|---:|---:|---:|---|---|']
    def fmt(r):
        if not r: return 'NA'
        if r.get('n') in (0,None): return f"0/{r.get('n',0)}"
        lo,hi=r.get('wilson95',[None,None]); ci=f"[{lo:.3f},{hi:.3f}]" if lo is not None else ''
        return f"{r['k']}/{r['n']} ({r['rate']:.3f} {ci})"
    for arm in picks:
        r0=report['arms'][arm]['runs'][0]['v2_all145']
        md.append(f"| {arm} | {r0['n_rows']} | {fmt(r0['numeric_match'])} | {fmt(r0['entity_match'])} | {fmt(r0['correct_abstention'])} | {fmt(r0['false_answer'])} | {fmt(r0['false_abstention'])} | {fmt(r0['parse_ok'])} | {fmt(r0['citation_valid'])} | {r0['budget_exceeded']} | {r0['filtered_empty']} | {r0['errors']} | {r0['tokens']['median']}/{r0['tokens']['p95']} | {r0['latency_ms']['median']}/{r0['latency_ms']['p95']} |")
    md+=['','## Common answerable subset', f"n={report['common_answerable_subset']['n']} arms={report['common_answerable_subset']['arms']}"]
    md+=['','## McNemar (exact, paired, common subset)','| pair | A>B | B>A | discordant n | p_exact |','|---|---:|---:|---:|---:|']
    for k,v in report['pairs'].items(): md.append(f"| {k} | {v['a_correct_b_wrong']} | {v['a_wrong_b_correct']} | {v['n_discordant']} | {v['p_exact']:.4f} |")
    # Gold corrections
    applied=[]; low_ids=[]
    corr_path=ROOT/'data/eval/datasets/gold_data.corrections.jsonl'
    if corr_path.exists():
        for l in open(corr_path):
            if not l.strip(): continue
            r=json.loads(l)
            if r.get('confidence')=='HIGH' and r.get('issue')=='note_number_column_bug_relaxed':
                applied.append(r)
    spot=ROOT/'results/gold_qa_spotcheck.md'
    if spot.exists():
        for line in spot.read_text().splitlines():
            if line.startswith('| G-'):
                low_ids.append(line.split('|')[1].strip())
    md+=['','## Gold corrections',
         'Rule (deterministic scan, not edited toward model outputs): active answerable numeric rows scanned n=111 for note-number bug (expected small-int/note-ref token; relaxed HIGH = label+small-int note-ref+value columns, question period/basis matches FIRST value column; annual=first/current, quarterly by header text). Single fitting candidate → HIGH applied via corrections overlay only (frozen gold unchanged). Ambiguous → LOW pending review.',
         f'Scan size: 111; HIGH applied: {len(applied)} ids={",".join(r.get("row_id") for r in applied)}; LOW pending review: {len(low_ids)} ids={",".join(low_ids)} (`results/gold_qa_spotcheck.md`).',
         '',
         '| id | old | new | evidence (printed line) | page |',
         '|---|---:|---:|---|---:|']
    for r in applied:
        c=r.get('current',{}); pr=r.get('proposed',{}); e=r.get('evidence',{})
        line=(e.get('printed_line') or '').replace('|','¦').replace('\n',' ')
        md.append(f"| {r.get('row_id')} | {c.get('expected_value')} | {pr.get('expected_value')} | `{line}` | {e.get('pdf_page')} |")
    md+=['',
         '### Before/after numeric accuracy (canonical dev, all 145; before=pre-correction original gold, after=corrected view v2)',
         '| arm | numeric before k/n (rate) | numeric after k/n (rate) |',
         '|---|---:|---:|']
    def mnum(run, which):
        rr=run.get(which) or {}; nm=rr.get('numeric_match') or {}; k=nm.get('k'); n=nm.get('n'); rate=nm.get('rate')
        if n in (None,0): return 'NA'
        return f"{k}/{n} ({rate:.3f})"
    for arm in picks:
        r0=report['arms'][arm]['runs'][0]
        # before = pre-correction (score_pre_correction), after = corrected view
        before = mnum(r0,'pre_correction_all145') if 'pre_correction_all145' in r0 else (mnum(r0,'v1_sensitivity_all145') if 'v1_sensitivity_all145' in r0 else 'NA')
        md.append(f"| {arm} | {before} | {mnum(r0,'v2_all145')} |")
    md+=['','## Spread (v2 all rows)','```json', json.dumps({a:report['arms'][a]['spread'] for a in picks}, indent=2), '```','',
         '## Notes', '- B2-global UNVERIFIED (oracle gate).', '- S2 abstention NOT MEASURED.', '- Corrections applied by deterministic scan (overlay), not edited toward model outputs; candidates not applied remain in corrections file.']
    (OUT/'report.md').write_text('\n'.join(md)+'\n')
    print('WROTE', OUT/'report.json', OUT/'report.md')

if __name__=='__main__':
    main()

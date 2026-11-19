import json, shutil, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.eval_phase2.gold_view import dev_rows
from src.eval.prompting.scoring import score_prediction

gold={r['id']:r for r in dev_rows()}
runs=ROOT/'results/week1_prompt_baseline/runs'
n=0
for p in sorted(runs.glob('*.jsonl')):
    bak=p.with_suffix(p.suffix+'.pre_correction.bak')
    if not bak.exists():
        shutil.copy2(p,bak)
    rows=[json.loads(l) for l in open(p)]
    out=[]
    for r in rows:
        g=gold.get(r.get('row_id'))
        ctx=set(tuple(x) for x in (r.get('context_pages') or []))
        if g:
            # keep previous as sensitivity (v1 or prior v2 pre-correction)
            if 'score_pre_correction' not in r:
                r['score_pre_correction']=r.get('score')
            r['score']=score_prediction(g, r.get('output') or '', ctx)
            r['scorer_version']='v2_unit_norm_corrected_view'
        out.append(r)
    with open(p,'w') as f:
        for r in out: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    n+=1
print('rescored',n)

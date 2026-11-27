import json, os
from pathlib import Path
os.environ.setdefault('AGENT_CACHE_ENABLED','0')
os.environ.setdefault('RETRIEVAL_HYBRID','1')
os.environ.setdefault('RETRIEVAL_RERANK','1')
from src.agent.service import run_question
fixtures = Path('tests/safety/fixtures.jsonl')
cases=[json.loads(l) for l in fixtures.read_text().splitlines() if l.strip()]
rows=[]
for c in cases:
    try:
        r=run_question(c['question'], session_id=f"live-safety-final-{c['id']}")
        ok=True
        if c['expected_refusal']:
            ok = r.stop_reason=='refused' and r.answer==c.get('expected_refusal_reason',r.answer)
        else:
            ok = r.stop_reason!='refused'
        rows.append({'id':c['id'],'cat':c['category'],'ok':ok,'stop':r.stop_reason,'err':r.error_type})
    except Exception as e:
        rows.append({'id':c['id'],'cat':c['category'],'ok':False,'stop':'exception','err':type(e).__name__})
summary={'n':len(rows),'passed':sum(r['ok'] for r in rows),'failed':[r for r in rows if not r['ok']],'rows':rows}
Path('/tmp/opencode/live_safety_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'n':summary['n'],'passed':summary['passed'],'failed':len(summary['failed'])}))

import json
from src.eval.prompting.inputs import load_dev_rows
from src.eval.prompting.scoring import score_prediction, parse_prediction, _expected_numbers, _scale, gold_answer_kind
rows=load_dev_rows(); gold={r['id']:r for r in rows}
path='results/week1_prompt_baseline/runs/phase2_v1_20261112T173032Z_B1.jsonl'
out={}
for line in open(path):
 r=json.loads(line); out[r['row_id']]=r
fails=[]
for rid,row in out.items():
 g=gold[rid]
 if g.get('expected_behaviour')!='answer': continue
 s=score_prediction(g,row.get('output') or '', set())
 if s.get('gold_answer_kind')=='numeric' and s.get('numeric_match') is False:
  pred=parse_prediction(row.get('output') or '')
  exp_nums=_expected_numbers(g)
  exp_scale=_scale(g.get('expected_unit_raw') or g.get('expected_answer') or '')
  pred_scale=_scale(' '.join(pred.get('units',[])))
  # classify
  kind='wrong_figure'
  if pred.get('values') and exp_nums:
   # try scale mismatch
   if abs(pred['values'][0]*exp_scale - exp_nums[0]*exp_scale) > 1e-6 and abs(pred['values'][0]*pred_scale - exp_nums[0]*exp_scale) < 1e-3*abs(exp_nums[0]*exp_scale or 1):
    kind='unit/scale'
   # period/basis heuristic: different number but plausible magnitude
   if abs(pred['values'][0] - exp_nums[0]) > 0 and abs(pred['values'][0]/(exp_nums[0] or 1)-1)>0.2:
    kind='wrong_figure_or_period'
  fails.append((rid, g.get('question_text','')[:140], g.get('expected_answer'), g.get('expected_value'), g.get('expected_unit_raw'), pred, row.get('output','')[:140], kind))
for x in fails[:10]:
 print('ROW',x[0])
 print('Q',x[1])
 print('EXP',x[2],'|',x[3],x[4])
 print('PRED',x[5])
 print('OUT',x[6])
 print('CLASS',x[7])
 print('---')
print('total',len(fails))

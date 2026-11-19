import json
from src.eval.prompting.inputs import load_dev_rows
from src.eval.retrieval_sanity import oracle_index_check, _recall_for, build_all_chunks
rows=load_dev_rows()
# answerable rows among dev
ans=[r for r in rows if r.get('expected_behaviour')=='answer']
print('oracle_index_problems', oracle_index_check())
chunks=build_all_chunks('raw')
res_filtered=_recall_for(chunks, rows, filtered=True)
res_global=_recall_for(chunks, rows, filtered=False)
print('filtered', {k:res_filtered[k] for k in ('n','recall@1','recall@5','recall@10')})
print('global', {k:res_global[k] for k in ('n','recall@1','recall@5','recall@10')})
for cat in sorted(res_filtered['by_category']):
 f=res_filtered['by_category'][cat]; g=res_global['by_category'][cat]
 print(cat, f['n'], f['recall@1'], f['recall@5'], f['recall@10'], g['recall@1'], g['recall@5'], g['recall@10'])

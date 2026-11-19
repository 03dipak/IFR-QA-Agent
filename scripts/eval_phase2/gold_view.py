import json, hashlib, pathlib, copy
ROOT=pathlib.Path(__file__).resolve().parents[2]
GOLD=ROOT/'data/eval/datasets/gold_data.jsonl'
CORR=ROOT/'data/eval/datasets/gold_data.corrections.jsonl'
META=ROOT/'data/eval/datasets/gold_data.corrections_meta.json'

def load_gold_rows():
    return [json.loads(l) for l in open(GOLD) if l.strip()]

def load_applied_corrections():
    if not CORR.exists(): return {}
    out={}
    for l in open(CORR):
        if not l.strip(): continue
        r=json.loads(l)
        if r.get('confidence')=='HIGH' and r.get('issue') in ('note_number_column_bug','note_number_column_bug_relaxed'):
            out[r['row_id']]=r
    return out

import re
def apply_correction(row, corr):
    r=copy.deepcopy(row)
    prop=corr.get('proposed') or {}
    ea_orig=str(row.get('expected_answer') or '')
    oldv=row.get('expected_value')
    if 'expected_value' in prop and prop['expected_value'] is not None:
        r['expected_value']=prop['expected_value']
    if 'expected_answer' in prop and prop['expected_answer'] is not None:
        r['expected_answer']=str(prop['expected_answer'])
    if ('=' in ea_orig or ';' in ea_orig) and oldv is not None:
        try:
            oldf=float(oldv)
            newv=prop.get('expected_value', prop.get('expected_answer'))
            def repl(m):
                tok=m.group(0)
                t=tok.replace(',','').strip('()')
                try:
                    if abs(float(t)-oldf)<1e-6:
                        if isinstance(newv,float) and not newv.is_integer(): return str(newv)
                        if isinstance(newv,float) and newv.is_integer(): return str(int(newv)) if abs(newv)<1e15 else str(newv)
                        return str(newv)
                except: pass
                return tok
            new_ea=re.sub(r"\(?-?\d[\d,]*(?:\.\d+)?\)?", repl, ea_orig)
            if new_ea!=ea_orig:
                r['expected_answer']=new_ea
                r['expected_value']=prop.get('expected_value', r.get('expected_value'))
        except: pass
    if 'answer_value' in prop and prop.get('answer_value') is not None:
        r['answer_value']=prop['answer_value']
    r['_correction_applied']=corr.get('issue')
    return r

def load_gold_view():
    rows=load_gold_rows()
    corr=load_applied_corrections()
    out=[]
    for r in rows:
        if r['id'] in corr:
            out.append(apply_correction(r,corr[r['id']]))
        else:
            out.append(r)
    meta=json.loads(META.read_text()) if META.exists() else {}
    return out, meta

def dev_rows():
    rows,_=load_gold_view()
    return [r for r in rows if r.get('status')=='active' and r.get('split')=='dev']

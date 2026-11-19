#!/usr/bin/env python3
"""Fast, observable oracle/BM25 recall for dev rows.
- Build BM25 index ONCE, persist to disk (npz/json, checksum + index version hash).
- Filtered oracle recall: score only chunks of row's oracle-filtered documents.
- Global recall: sparse-matrix numpy implementation (no new deps; numpy in venv).
- Cross-check: rank_bm25-style/BM25Index top-10 must match on 10 rows.
- Progress every 10 rows with elapsed time; oracle_index_check once.
"""
import argparse, hashlib, json, math, os, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# allow running from /tmp copy
if not (ROOT/'data'/'eval').exists():
    ROOT = Path('/home/dipak/agentic/IFR-QA-Agent')
sys.path.insert(0, str(ROOT))

FREEZE = ROOT/"data/corpus/freeze_v1.json"
PAGE_DIR = ROOT/"data/corpus/extract/page_extracts"
TABLE_DIR = ROOT/"data/corpus/extract/table_records"
GOLD = ROOT/"data/eval/datasets/gold_data.jsonl"
INDEX_DIR = ROOT/"results/retrieval_sanity_bm25_index"

K1, B = 1.5, 0.75

# tokenizer (shared)
try:
    from src.rag_ingestion.tokenizer import tokenize as shared_tokenize, TOKENIZER_VERSION
except Exception:
    TOKENIZER_VERSION = "fallback.v1"
    def shared_tokenize(text):
        return re.findall(r"[a-z0-9]+", (text or "").lower())

def tokenize(text): return shared_tokenize(text)

def sha256_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def freeze_docs(): return json.loads(FREEZE.read_text(encoding='utf-8'))['documents']

def load_dev_rows():
    rows=[json.loads(l) for l in GOLD.read_text(encoding='utf-8').splitlines() if l.strip()]
    return [r for r in rows if r.get('status')=='active' and r.get('split')=='dev']

def cites(row):
    out=set()
    for e in row.get('evidence') or []:
        did=e.get('document_id'); pg=e.get('pdf_page')
        if isinstance(did,str) and isinstance(pg,int): out.add((did,pg))
    if not out and row.get('document_id') and isinstance(row.get('pdf_page_index'), int):
        out.add((row['document_id'], row['pdf_page_index']))
    return out

def oracle_index_check(rows):
    problems=[]
    for r in rows:
        if r.get('expected_behaviour')!='answer': continue
        for (did,pg) in cites(r):
            path=PAGE_DIR/f"{did}.pages.jsonl"
            ok=False
            if path.is_file():
                for line in path.read_text(encoding='utf-8').splitlines():
                    if not line.strip(): continue
                    rec=json.loads(line)
                    if rec.get('pdf_page_index')==pg:
                        ok=bool(rec.get('usable_as_evidence')); break
            if not ok:
                problems.append({'row_id':r.get('id') or r.get('row_id'),'document_id':did,'pdf_page':pg,'reason':'not_in_index_or_not_usable'})
    return problems

def build_chunks():
    chunks=[]
    for doc in freeze_docs():
        doc_id=doc['document_id']
        page_path=PAGE_DIR/f"{doc_id}.pages.jsonl"
        pages=[]
        if page_path.is_file():
            for line in page_path.read_text(encoding='utf-8').splitlines():
                if line.strip(): pages.append(json.loads(line))
        usable={r.get('pdf_page_index') for r in pages if r.get('usable_as_evidence') and isinstance(r.get('pdf_page_index'), int)}
        for rec in pages:
            page=rec.get('pdf_page_index')
            if not isinstance(page,int) or page not in usable: continue
            text=rec.get('page_text') or ''
            if not text.strip(): continue
            meta={'document_id':doc_id,'page':int(page),'company_id':doc.get('company_id'),'period':doc.get('fiscal_year'),'variant':doc.get('variant'),'doc_type':doc.get('doc_type'),'chunk_type':'page'}
            chunks.append({'id':f"{doc_id}:p{page}",'text':text,'meta':meta})
        table_path=TABLE_DIR/f"{doc_id}.tables.jsonl"
        if table_path.is_file():
            for n,line in enumerate(table_path.read_text(encoding='utf-8').splitlines()):
                if not line.strip(): continue
                rec=json.loads(line); page=rec.get('page_start')
                if not isinstance(page,int) or page not in usable: continue
                parts=[f"statement_kind: {rec.get('statement_kind')}", f"unit: {(rec.get('unit_header') or '').strip()}", f"caption: {(rec.get('caption') or '').strip()}", f"reporting_basis: {rec.get('reporting_basis')}"]
                for r in (rec.get('rows') or []):
                    rl=(r.get('row_label_raw') or '').strip()
                    cells=r.get('cells') or []
                    raw=[str(c.get('raw') or '').strip() for c in cells]; raw=[c for c in raw if c]
                    parts.append(f"{rl}: {' | '.join(raw)}" if rl else ' | '.join(raw))
                text='\n'.join(p for p in parts if p.strip())
                if not text.strip(): continue
                st=rec.get('statement_kind') or doc.get('doc_type')
                meta={'document_id':doc_id,'page':int(page),'company_id':doc.get('company_id'),'period':doc.get('fiscal_year'),'variant':doc.get('variant'),'doc_type':doc.get('doc_type'),'statement_kind':rec.get('statement_kind'),'statement_type':st,'chunk_type':'table'}
                chunks.append({'id':f"{doc_id}:t{n}",'text':text,'meta':meta})
    chunks.sort(key=lambda c:c['id'])
    return chunks

def index_version(chunks):
    h=hashlib.sha256()
    h.update(TOKENIZER_VERSION.encode())
    h.update(b'k1=1.5;b=0.75')
    h.update(str(len(chunks)).encode())
    for c in chunks:
        h.update(c['id'].encode()); h.update(str(len(c['text'])).encode())
    return h.hexdigest()

def build_index(chunks):
    t0=time.time()
    N=len(chunks)
    vocab={}
    dl=np.zeros(N,dtype=np.int32)
    postings_lists=defaultdict(lambda:([],[])) # term_id -> (doc_ids, tfs)
    for i,c in enumerate(chunks):
        toks=tokenize(c['text'])
        dl[i]=len(toks)
        tf=Counter(toks)
        for t,f in tf.items():
            tid=vocab.get(t)
            if tid is None: tid=len(vocab); vocab[t]=tid
            postings_lists[tid][0].append(i); postings_lists[tid][1].append(f)
    df=np.zeros(len(vocab),dtype=np.int32)
    for tid,(docs,_) in postings_lists.items(): df[tid]=len(docs)
    idf=np.log((N-df+0.5)/(df+0.5)+1.0).astype(np.float32)
    # csr-like arrays per term: offsets + flat
    term_ptr=np.zeros(len(vocab)+1,dtype=np.int64)
    total=sum(len(d) for d,_ in postings_lists.values())
    flat_docs=np.zeros(total,dtype=np.int32)
    flat_tfs=np.zeros(total,dtype=np.int32)
    off=0
    for tid in range(len(vocab)):
        docs,tfs=postings_lists.get(tid,([],[]))
        n=len(docs)
        term_ptr[tid]=off
        if n:
            flat_docs[off:off+n]=np.array(docs,dtype=np.int32)
            flat_tfs[off:off+n]=np.array(tfs,dtype=np.int32)
        off+=n
    term_ptr[len(vocab)]=off
    build_s=time.time()-t0
    return {'N':N,'dl':dl,'idf':idf,'term_ptr':term_ptr,'flat_docs':flat_docs,'flat_tfs':flat_tfs,'vocab':vocab,'build_seconds':build_s}

def save_index(idx, chunks, ver, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir/'index.npz', N=np.array([idx['N']]), dl=idx['dl'], idf=idx['idf'], term_ptr=idx['term_ptr'], flat_docs=idx['flat_docs'], flat_tfs=idx['flat_tfs'])
    # vocab json
    (out_dir/'vocab.json').write_text(json.dumps(idx['vocab'], separators=(',',':')), encoding='utf-8')
    # meta/chunks mapping
    meta=[{'id':c['id'],'meta':c['meta']} for c in chunks]
    (out_dir/'chunks_meta.jsonl').write_text('\n'.join(json.dumps(m,separators=(',',':')) for m in meta)+'\n', encoding='utf-8')
    manifest={'index_version':ver,'tokenizer_version':TOKENIZER_VERSION,'n_chunks':len(chunks),'k1':K1,'b':B,'build_seconds':idx['build_seconds'],
              'files':{f:sha256_file(out_dir/f) for f in ['index.npz','vocab.json','chunks_meta.jsonl']}}
    manifest['manifest_checksum']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True), encoding='utf-8')
    return manifest

def load_index(out_dir):
    m=json.loads((out_dir/'manifest.json').read_text(encoding='utf-8'))
    # verify checksums
    chk=m['files']
    for f,h in chk.items():
        if sha256_file(out_dir/f)!=h: raise ValueError(f'checksum mismatch {f}')
    npz=np.load(out_dir/'index.npz')
    vocab=json.loads((out_dir/'vocab.json').read_text(encoding='utf-8'))
    chunks_meta=[json.loads(l) for l in (out_dir/'chunks_meta.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    idx={'N':int(npz['N'][0]),'dl':npz['dl'],'idf':npz['idf'],'term_ptr':npz['term_ptr'],'flat_docs':npz['flat_docs'],'flat_tfs':npz['flat_tfs'],'vocab':vocab,'build_seconds':m.get('build_seconds')}
    return idx,chunks_meta,m

def search_sparse(idx, query, k=10, subset=None):
    N=idx['N']; dl=idx['dl']; avgdl=(dl.sum()/N) if N else 1.0
    if subset is None:
        scores=np.zeros(N,dtype=np.float32)
        cand=None
    else:
        cand=np.array(sorted(set(int(x) for x in subset)),dtype=np.int32)
        if len(cand)==0: return []
        scores=np.zeros(len(cand),dtype=np.float32)
        pos={int(d):i for i,d in enumerate(cand)}
    qtf=Counter(tokenize(query))
    for t,fq in qtf.items():
        tid=idx['vocab'].get(t)
        if tid is None: continue
        s=int(idx['term_ptr'][tid]); e=int(idx['term_ptr'][tid+1])
        if e<=s: continue
        docs=idx['flat_docs'][s:e]; tfs=idx['flat_tfs'][s:e].astype(np.float32)
        idf=float(idx['idf'][tid])
        denom=tfs + K1*(1-B + B*dl[docs]/(avgdl or 1.0))
        contrib=idf * tfs*(K1+1)/denom
        if cand is None:
            scores[docs]+=contrib
        else:
            # filter to subset
            for d,c in zip(docs,contrib):
                j=pos.get(int(d))
                if j is not None: scores[j]+=c
    if cand is None:
        kk=min(k,N)
        if kk<=0: return []
        top=np.argpartition(-scores, kk-1)[:kk]
        top=top[np.argsort((-scores[top], top))]
        return [(int(i), float(scores[i])) for i in top]
    else:
        kk=min(k,len(cand))
        if kk<=0: return []
        top=np.argpartition(-scores, kk-1)[:kk]
        top_idx=cand[top]
        top=top[np.argsort((-scores[top], top_idx))]
        return [(int(cand[int(i)]), float(scores[int(i)])) for i in top]

def doc_chunk_indices(chunks_meta):
    m=defaultdict(list)
    for i,c in enumerate(chunks_meta): m[c['meta']['document_id']].append(i)
    return m

def hit_at_k(retrieved_meta, gold_set, k):
    for c in retrieved_meta[:k]:
        if (c['meta']['document_id'], c['meta']['page']) in gold_set: return True
    return False

def run():
    ap=argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--force-rebuild', action='store_true')
    args=ap.parse_args()
    t_all=time.time()
    rows=load_dev_rows()
    if args.limit: rows=rows[:args.limit]
    print(f'rows={len(rows)} (dev active)')
    t0=time.time(); problems=oracle_index_check(rows); t_oracle=time.time()-t0
    print(f'oracle_index_check problems={len(problems)} time={t_oracle:.2f}s')
    # chunks/index
    t0=time.time(); chunks=build_chunks(); t_chunks=time.time()-t0
    ver=index_version(chunks)
    out_dir=INDEX_DIR/ver
    if out_dir.is_dir() and not args.force_rebuild:
        t0=time.time(); idx,chunks_meta,manifest=load_index(out_dir); t_load=time.time()-t0
        print(f'index loaded ver={ver[:12]} chunks={len(chunks_meta)} load={t_load:.2f}s build_saved={manifest.get("build_seconds")}')
    else:
        print(f'building index chunks={len(chunks)}...')
        idx=build_index(chunks)
        manifest=save_index(idx,chunks,ver,out_dir)
        chunks_meta=[{'id':c['id'],'meta':c['meta']} for c in chunks]
        print(f'index built ver={ver[:12]} build={idx["build_seconds"]:.2f}s saved={out_dir}')
    by_doc=doc_chunk_indices(chunks_meta)

    # cross-check 10 rows vs reference BM25Index (same data subset?) - use in-memory ref on all chunks
    try:
        from src.rag_ingestion.index import BM25Index as RefBM25
        ref_chunks=[{'text':c['text'],'meta':c['meta']} for c in chunks]
        t0=time.time(); ref=RefBM25(ref_chunks); t_ref=time.time()-t0
        sample=[r for r in rows if r.get('expected_behaviour')=='answer'][:10]
        mism=0
        for r in sample:
            q=r.get('question_text') or r.get('question') or ''
            a=[i for i,_ in search_sparse(idx,q,10)]
            b=[i for i,_ in ref.search(q,10)]
            if a!=b: mism+=1
        print(f'crosscheck ref_build={t_ref:.2f}s sample={len(sample)} top10_mismatch={mism}')
    except Exception as e:
        print(f'crosscheck skipped: {e}')

    # recall
    ks=(1,5,10)
    def init_acc():
        return {'overall':{k:0 for k in ks},'n':0,'by_bucket':defaultdict(lambda:{'n':0,**{k:0 for k in ks}})}
    filt=init_acc(); glob=init_acc()
    qtimes=[]; t_loop=time.time()
    answer_rows=[r for r in rows if r.get('expected_behaviour')=='answer']
    for n,r in enumerate(answer_rows,1):
        q=r.get('question_text') or r.get('question') or ''
        gold=cites(r); bucket=r.get('flow_bucket') or '?'
        # filtered: union chunks of oracle documents
        oracle_docs={d for d,_ in gold}
        cand=[]
        for d in oracle_docs: cand.extend(by_doc.get(d,[]))
        tq=time.time(); ranked_f=search_sparse(idx,q,10,subset=cand) if cand else []; qf=time.time()-tq
        # global
        tq=time.time(); ranked_g=search_sparse(idx,q,10,subset=None); qg=time.time()-tq
        qtimes.append((qf,qg))
        rm_f=[chunks_meta[i] for i,_ in ranked_f]; rm_g=[chunks_meta[i] for i,_ in ranked_g]
        for acc,rm in ((filt,rm_f),(glob,rm_g)):
            acc['n']+=1; acc['by_bucket'][bucket]['n']+=1
            for k in ks:
                h=1 if hit_at_k(rm,gold,k) else 0
                acc['overall'][k]+=h; acc['by_bucket'][bucket][k]+=h
        if n%10==0:
            print(f'progress {n}/{len(answer_rows)} elapsed={time.time()-t_loop:.2f}s last_q_ms filtered={qf*1000:.2f} global={qg*1000:.2f}', flush=True)
    def finalize(acc):
        out={'n':acc['n']}
        for k in ks: out[f'recall@{k}']= (acc['overall'][k]/acc['n']) if acc['n'] else None; out[f'hits@{k}']=acc['overall'][k]
        out['by_bucket']={}
        for b,d in sorted(acc['by_bucket'].items()):
            out['by_bucket'][b]={'n':d['n']}
            for k in ks: out['by_bucket'][b][f'recall@{k}']=(d[k]/d['n']) if d['n'] else None
        return out
    res={'index_version':ver,'index_dir':str(out_dir),'build_seconds':idx.get('build_seconds'),'chunk_build_seconds':t_chunks,'oracle_check_seconds':t_oracle,'oracle_index_problems':problems,
         'per_query_ms_filtered_median':(float(np.median([x[0] for x in qtimes])*1000) if qtimes else None),
         'per_query_ms_global_median':(float(np.median([x[1] for x in qtimes])*1000) if qtimes else None),
         'per_query_ms_filtered_p95':(float(np.percentile([x[0] for x in qtimes],95)*1000) if qtimes else None),
         'per_query_ms_global_p95':(float(np.percentile([x[1] for x in qtimes],95)*1000) if qtimes else None),
         'filtered':finalize(filt),'global':finalize(glob),'elapsed_total_seconds':time.time()-t_all}
    outp=ROOT/'results/retrieval_sanity_bm25_fast.json'
    outp.write_text(json.dumps(res,indent=2),encoding='utf-8')
    print('RESULT_JSON_PATH', outp)
    print(json.dumps({'filtered':{k:res['filtered'][k] for k in ('n','recall@1','recall@5','recall@10')},'global':{k:res['global'][k] for k in ('n','recall@1','recall@5','recall@10')}}, indent=2))
    # bucket table
    print('BUCKET\tn\tF@1\tF@5\tF@10\tG@1\tG@5\tG@10')
    for b in sorted(set(res['filtered']['by_bucket'])|set(res['global']['by_bucket'])):
        f=res['filtered']['by_bucket'].get(b,{}); g=res['global']['by_bucket'].get(b,{})
        def fmt(x): return 'NA' if x is None else f"{x:.3f}"
        print(f"{b}\t{f.get('n',g.get('n',0))}\t{fmt(f.get('recall@1'))}\t{fmt(f.get('recall@5'))}\t{fmt(f.get('recall@10'))}\t{fmt(g.get('recall@1'))}\t{fmt(g.get('recall@5'))}\t{fmt(g.get('recall@10'))}")

if __name__=='__main__':
    run()

"""Round 3-A: fixed-artifact, offline-only statistics. No inference entrypoints imported.

Pure, named functions are extracted through AST; original modules/entrypoints
are never imported/executed. All writes are confined to the new audit folder.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import math
import random
import re
import statistics
import string
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path('E:/RAG_Projects/MultimodalEvidenceChain')
OUT = ROOT/'experiments/paper_statistical_audit_20261004'
SOURCES = {}
CODE = {}
RISK = 'outputs/mmqa_clean_risk_v2_dev1096_answer_risk_oof_v1/'
COMP = 'outputs/mmqa_clean_risk_v2_dev1096_expected_natural_explanatory_ablation_v1/'
CONTROL = 'outputs/mmqa_dev1096_a0_a4_a6_selector_endpoint_ablation_v1/'
LOCK = 'outputs/mmqa_clean_risk_v2_one_time_official_eval_v1/'
MCLEAN = 'outputs/m3vqa_clean_v2_one_time_official_gold_eval_v1/'
MRAND = 'outputs/m3vqa_random20_seed20260822_one_time_official_gold_eval_v1/'
MCOMPARE = 'outputs/m3vqa_clean_vs_random20_seed20260822_paired_bootstrap_v1/'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest().upper()

def read(rel):
    p=ROOT/rel
    b=p.read_bytes()
    SOURCES[str(p)]={'sha256_before':hashlib.sha256(b).hexdigest().upper(),'bytes':len(b)}
    return b.decode('utf-8-sig')

def j(rel): return json.loads(read(rel))

def jl(rel):
    rows=[json.loads(x) for x in read(rel).splitlines() if x.strip()]
    idx={r['qid']:r for r in rows}
    if len(rows)!=len(idx): raise ValueError(f'duplicate QID: {rel}')
    SOURCES[str(ROOT/rel)].update(rows=len(rows),unique_qids=len(idx),fields=sorted(rows[0]) if rows else [])
    return idx

def outpath(name):
    p=(OUT/name).resolve()
    if not p.is_relative_to(OUT.resolve()): raise ValueError('write outside audit folder')
    p.parent.mkdir(parents=True,exist_ok=True)
    return p

def wj(name,value): outpath(name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def wt(name,value): outpath(name).write_text(value,encoding='utf-8')
def wc(name,rows):
    if not rows: return
    with outpath(name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def fail(msg): raise RuntimeError(msg)

def pure(rel,names,constants=(),extra=None):
    """Explicit function/constant allowlist; excludes imports, entrypoints and writers."""
    tree=ast.parse(read(rel),filename=str(ROOT/rel))
    g={'np':np,'math':math,'random':random,'statistics':statistics,'re':re,'string':string,
       'ast':ast,'json':json,'Counter':Counter,'fail':fail,
       'average_precision_score':average_precision_score,'roc_auc_score':roc_auc_score}
    if extra: g.update(extra)
    selected=[]
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in constants:
                    g[target.id]=ast.literal_eval(node.value)
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
            selected.append(node)
            CODE[f'{rel}::{node.name}']={'path':str(ROOT/rel),'line':node.lineno,'end_line':node.end_lineno}
    if set(n.name for n in selected)!=set(names): raise ValueError(f'missing pure function in {rel}')
    # Postpone annotations so annotation-only imports are unnecessary.
    fut=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    ast.fix_missing_locations(fut)
    mod=ast.Module(body=[fut]+selected,type_ignores=[])
    exec(compile(mod,str(ROOT/rel),'exec'),g)
    return g

def comparable(a,b,tol=1e-10):
    return bool(np.allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),rtol=0,atol=tol))

def fixed_p(ranks,labels,scores):
    # Exact P@220 projection of the project's fast_bootstrap_metrics.
    return float(labels[np.lexsort((ranks,-scores))[:220]].sum()/220)

def risk_stage():
    wt('bootstrap_method_audit.md', '''# Bootstrap method audit — analysis definition fixed before execution

Primary estimand: difference of Error Precision in two independently re-ranked
Top-220 sets drawn from the same empirical resample of N=1,096 frozen paired
observations (QID, Initial error label, G0 score, B3 score). Scores and labels stay
fixed; 10,000 N-observation paired draws with replacement; NumPy default_rng,
seed 260817. Repeated QIDs retain multiplicity (no collapse); sort risk descending,
QID ascending; duplicate copies are identical in score/label, so their relative
order is immaterial. k=220; with fixed N this equals ceil(0.20*N).
95% CI: ordinary percentile (linear interpolation), not basic or BCa.
Error-count difference = 220 * precision difference, not a unique-QID count.
The existing pure fast_bootstrap_metrics function is used unchanged. No model,
vectorizer, threshold or selection is refit; B3 remains the previously selected
configuration. This is NEW SUPPLEMENTARY ANALYSIS, conditional on the fixed OOF
score vectors, not nested validation and not correction of variant-selection
optimism. AP/AUC differences are secondary projections of the same draws.

Section 4.5 replay keeps the original five-pair order and continuous RNG stream.
Only P@220 is replayed: omitted AP/AUC computations consume no random numbers.
The risk ranking and QID/multiplicity semantics are unchanged.

No unique-QID-collapse sensitivity is substituted: collapsing produces a smaller,
random population and a different budget fraction rather than the original
paired empirical bootstrap estimand. Fixed k and ceil(20% of resample size) are
identical here, and identical duplicate copies cannot change the tie result.
''')
    labels=jl(RISK+'dev1096_saved_answer_error_labels_and_shared_folds_v1.jsonl')
    qids=sorted(labels); assert len(qids)==1096
    y=np.array([labels[q]['initial_wrong_label'] for q in qids],dtype=int)
    assert y.sum()==680
    pred={k:jl(RISK+f'dev1096_{file}_oof_predictions_v1.jsonl') for k,file in [('G0','g0'),('B3','clean_b3')]}
    for k,d in pred.items():
        assert set(d)==set(labels)
        assert all(d[q]['initial_wrong_label']==labels[q]['initial_wrong_label'] and d[q]['heldout_fold']==labels[q]['heldout_fold'] and d[q]['oof_prediction_count']==1 for q in qids)
    scores={k:np.array([d[q]['oof_risk_score'] for q in qids]) for k,d in pred.items()}
    ranks=np.arange(1096)
    frozen=j(RISK+'clean_v2_oof_variant_selection_v1.json')
    protocol=j(RISK+'clean_v2_oof_evaluation_protocol_v1.json')
    seals=j(RISK+'SHA256SUMS.json')
    fast=pure('run_mmqa_clean_risk_v2_dev1096_expected_natural_explanatory_ablation_v1.py',['fast_bootstrap_metrics'],extra={'TOP_K':220})['fast_bootstrap_metrics']
    observed={k:fast(ranks,y,s) for k,s in scores.items()}
    assert round(observed['B3'][0]*220)==201 and round(observed['G0'][0]*220)==190
    selection_checks={}
    for k,s in scores.items():
        chosen=set(qids[i] for i in np.lexsort((ranks,-s))[:220])
        saved=set(q for q in qids if pred[k][q]['global_top20_selected'])
        selection_checks[k]={'saved_top220_matches':chosen==saved,'threshold':float(np.sort(s)[-220]),'boundary_ties':int(np.sum(s==np.sort(s)[-220]))}
        assert chosen==saved
    rng=np.random.default_rng(260817); dif=np.empty((10000,4))
    for b in range(10000):
        ix=rng.integers(0,1096,size=1096)
        dif[b]=fast(ranks[ix],y[ix],scores['B3'][ix])-fast(ranks[ix],y[ix],scores['G0'][ix])
        if (b+1)%2000==0: print('B3-G0 bootstrap',b+1,flush=True)
    ci=np.percentile(dif,[2.5,97.5],axis=0)
    d=observed['B3']-observed['G0']
    result={'status':'NEW SUPPLEMENTARY ANALYSIS','evidence_class':'SUPPLEMENTARY / EXPLORATORY','N':1096,'initial_errors':680,'top_k':220,'iterations':10000,'seed':260817,'resampling_unit':'paired QID observation with multiplicity','reranking':True,'refitting':False,'variant_reselection':False,'score_fixed':True,'ci_method':'percentile; linear interpolation','sort':'risk descending, QID ascending','primary':{'observed_diff_pp':float(d[0]*100),'ci_low_pp':float(ci[0,0]*100),'ci_high_pp':float(ci[1,0]*100),'observed_error_count_diff':11,'ci_error_count_diff':[float(ci[0,0]*220),float(ci[1,0]*220)],'ci_contains_zero':bool(ci[0,0]<=0<=ci[1,0])},'secondary':{name:{'observed_diff_pp':float(d[i]*100),'ci_low_pp':float(ci[0,i]*100),'ci_high_pp':float(ci[1,i]*100)} for i,name in [(1,'Error Recall'),(2,'AP'),(3,'ROC-AUC')]},'frozen_descriptive_metrics':{k:{'errors_at_220':int(round(v[0]*220)),'error_precision_percent':float(v[0]*100),'AP_percent':float(v[2]*100),'ROC_AUC_percent':float(v[3]*100)} for k,v in observed.items()},'selection_checks':selection_checks,'selected_variant_artifact':str(ROOT/(RISK+'clean_v2_oof_variant_selection_v1.json')),'original_selection_record':frozen,'source_code':CODE.copy()}
    wj('b3_vs_g0_paired_uncertainty.json',result)
    print('B3-G0 PRIMARY',result['primary'],flush=True)
    # Existing Section 4.5 analysis, not an additional feature-selection exercise.
    old=j(COMP+'explanatory_ablation_pairwise_bootstrap_v1.json')
    pairs=[('A4','A1'),('A4','A2'),('A5','A3'),('S1','A4'),('A6','S2')]
    sc={k:jl(COMP+f'dev1096_{k.lower()}_oof_predictions_v1.jsonl') for k in {x for p in pairs for x in p}}
    for k,v in sc.items():
        assert set(v)==set(labels)
        assert all(v[q]['initial_wrong_label']==labels[q]['initial_wrong_label'] for q in qids)
    vals={k:np.array([v[q]['oof_risk_score'] for q in qids]) for k,v in sc.items()}
    rng=np.random.default_rng(260817); rows=[]
    for position,(left,right) in enumerate(pairs,1):
        ds=np.empty(10000)
        for b in range(10000):
            ix=rng.integers(0,1096,size=1096)
            ds[b]=fixed_p(ranks[ix],y[ix],vals[left][ix])-fixed_p(ranks[ix],y[ix],vals[right][ix])
        c=np.percentile(ds,[2.5,97.5]); od=fixed_p(ranks,y,vals[left])-fixed_p(ranks,y,vals[right])
        record=old['pairs'][f'{left}_minus_{right}']['metrics']['top20_error_precision']
        ok=comparable([od,*c],[record['observed_difference'],*record['percentile_95_ci']])
        rows.append({'comparison':f'{left} - {right}','observed_diff_pp':record['observed_difference']*100,'ci_low_pp':record['percentile_95_ci'][0]*100,'ci_high_pp':record['percentile_95_ci'][1]*100,'iterations':10000,'seed':260817,'rng_pair_position':position,'estimand':'paired frozen-score re-ranked Top-220 error precision difference','resampling_unit':'QID with multiplicity','rerank':True,'top_k':220,'tie_break':'risk_desc,QID_asc','retrain':False,'ci_contains_zero':bool(c[0]<=0<=c[1]),'replay_diff_pp':od*100,'replay_ci_low_pp':c[0]*100,'replay_ci_high_pp':c[1]*100,'source_artifact':str(ROOT/(COMP+'explanatory_ablation_pairwise_bootstrap_v1.json')),'status':'PASS' if ok else 'MISMATCH — investigate; preserve original'})
        print('Section 4.5',left,right,'replay',ok,flush=True)
    wc('section45_component_pairwise_statistics.csv',rows)
    wj('section45_replay_details.json',rows)

def m3_stage():
    local=pure('evaluate_m3vqa_frozen13125_one_time_official_v1.py',['safe','normalize_answer','convert_to_list','official_accuracy'],constants=('DIGIT_MAP','CONTRACTIONS'))
    upstream=pure('third_party/M3VQA/run_eval.py',['normalize_answer','evaluate_predictions'],constants=('digitMap','contractions'))
    boot=pure('evaluate_m3vqa_clean_v2_one_time_official_gold_v1.py',['paired_bootstrap_ci'],constants=('BOOTSTRAP_SEED','BOOTSTRAP_REPS','EXPECTED_TOTAL'))['paired_bootstrap_ci']
    clean=jl(MCLEAN+'m3vqa_clean_v2_one_time_official_gold_eval_details_v1.jsonl')
    rand=jl(MRAND+'m3vqa_random20_seed20260822_official_eval_per_qid_v1.jsonl')
    refs=jl('outputs/m3vqa_dataset_adapter_v1_1/m3vqa_external_zero_shot_eval_only_v1_1.jsonl')
    qids=sorted(clean); assert len(qids)==13125 and set(clean)==set(rand)==set(refs)
    differences=[]; local_diffs=[]; examples={}; norms=[]
    upstream_totals=Counter(); local_totals=Counter(); outside_unit=Counter()
    for q in qids:
        ce,re_,ref=clean[q],rand[q],refs[q]['answer_evals']
        assert ce['initial_answer']==re_['initial_answer']
        for arm,answer,saved in [('Initial',ce['initial_answer'],ce['initial_official_accuracy']),('Clean-B3',ce['clean_v2_final_answer'],ce['clean_v2_official_accuracy']),('Random-20',re_['random20_final_answer'],re_['random20_official_accuracy'])]:
            ans=local['convert_to_list'](answer)
            ls=local['official_accuracy'](ans,ref)
            us=upstream['evaluate_predictions'](ans,ref)
            local_totals[arm]+=ls; upstream_totals[arm]+=us
            if us>1+1e-12 or us<-1e-12: outside_unit[arm]+=1
            if abs(ls-saved)>1e-12: local_diffs.append({'qid':q,'arm':arm,'stored':saved,'local_replay':ls})
            if abs(ls-us)>1e-12:
                differences.append({'qid':q,'arm':arm,'prediction':answer,'parsed_prediction':json.dumps(ans,ensure_ascii=False),'reference_groups':json.dumps(ref,ensure_ascii=False),'saved_local_score':saved,'recomputed_local_score':ls,'upstream_score':us})
            if arm=='Initial':
                bucket='full' if abs(saved-1)<1e-12 else 'zero' if abs(saved)<1e-12 else 'partial'
                if bucket not in examples:
                    examples[bucket]={'category':bucket,'qid':q,'arm':'Initial (deterministic QID-ascending example; no method win selection)','prediction':answer,'parsed_prediction':json.dumps(ans,ensure_ascii=False),'normalized_prediction':json.dumps([local['normalize_answer'](a) for a in ans],ensure_ascii=False),'reference_groups':json.dumps(ref,ensure_ascii=False),'normalized_reference_groups':json.dumps([[local['normalize_answer'](a) for a in group] for group in ref],ensure_ascii=False),'stored_score':saved,'local_replay_score':ls,'upstream_score':us}
    cv=np.array([clean[q]['clean_v2_official_accuracy'] for q in qids])
    rv=np.array([rand[q]['random20_official_accuracy'] for q in qids])
    iv=np.array([clean[q]['initial_official_accuracy'] for q in qids])
    diff=cv-rv
    nonfull=(cv<1-1e-12)&(rv<1-1e-12)
    haspartial=((cv>1e-12)&(cv<1-1e-12))|((rv>1e-12)&(rv<1-1e-12))
    strict={}
    for arm,v in [('Clean-B3',cv),('Random-20',rv)]:
        b=iv>=1-1e-12; a=v>=1-1e-12
        w=int(np.sum(~b&a));r=int(np.sum(b&~a))
        strict[arm]={'W2R':w,'R2W':r,'Net':w-r}
    provenance={
        'clean_summary':j(MCLEAN+'m3vqa_clean_v2_one_time_official_gold_eval_summary_v1.json'),
        'random_summary':j(MRAND+'m3vqa_random20_seed20260822_official_eval_summary_v1.json'),
        'paired_summary':j(MCOMPARE+'m3vqa_clean_vs_random20_paired_bootstrap_summary_v1.json'),
        'clean_significance':j(MCLEAN+'m3vqa_clean_v2_significance_v1.json'),
    }
    # Replay the saved local-score CIs; upstream discrepancy audit is kept separate.
    boot_cr=boot(diff);boot_ci=boot(cv-iv)
    old_cr=provenance['paired_summary']['paired_official_accuracy_bootstrap']
    old_ci=provenance['clean_significance']['Initial_vs_Clean_V2']['paired_official_accuracy_bootstrap']
    replay={'Clean_minus_Random':{'replay':boot_cr,'record':old_cr,'matches':comparable([boot_cr['delta_accuracy_pp_95ci_lower'],boot_cr['delta_accuracy_pp_95ci_upper']],[old_cr['delta_accuracy_pp_95ci_lower'],old_cr['delta_accuracy_pp_95ci_upper']])},'Clean_minus_Initial':{'replay':boot_ci,'record':old_ci,'matches':comparable([boot_ci['delta_accuracy_pp_95ci_lower'],boot_ci['delta_accuracy_pp_95ci_upper']],[old_ci['delta_accuracy_pp_95ci_lower'],old_ci['delta_accuracy_pp_95ci_upper']])}}
    finals={'Clean':jl('outputs/m3vqa_clean_v2_final_prediction_merge_before_gold_v1/m3vqa_clean_v2_final_predictions_13125_frozen_before_gold_v1.jsonl'),'Random':jl('outputs/m3vqa_random20_seed20260822_final_prediction_merge_before_gold_v1/m3vqa_random20_seed20260822_final_predictions_13125_before_gold_v1.jsonl')}
    assert set(finals['Clean'])==set(clean) and set(finals['Random'])==set(rand)
    frozen_answer_matches={'Clean':all(finals['Clean'][q]['final_answer']==clean[q]['clean_v2_final_answer'] for q in qids),'Random':all(finals['Random'][q]['final_answer']==rand[q]['random20_final_answer'] for q in qids)}
    result={'N':13125,'local_scorer_replay_mismatches':local_diffs,'upstream_mismatch_count':len(differences),'upstream_mismatch_counts_by_arm':dict(Counter(d['arm'] for d in differences)),'stored_means_percent':{'Initial':float(iv.mean()*100),'Clean-B3':float(cv.mean()*100),'Random-20':float(rv.mean()*100)},'upstream_means_percent':{k:v/13125*100 for k,v in upstream_totals.items()},'S3':{'Clean_higher':int(np.sum(diff>1e-12)),'equal':int(np.sum(abs(diff)<=1e-12)),'Clean_lower':int(np.sum(diff<-1e-12)),'score_sum_delta':float(math.fsum(diff.tolist())),'mean_delta_pp':float(diff.mean()*100),'both_nonfull_and_at_least_one_partial_higher':int(np.sum(nonfull&haspartial&(diff>1e-12))),'both_nonfull_and_at_least_one_partial_lower':int(np.sum(nonfull&haspartial&(diff<-1e-12)))},'strict_migrations':strict,'selections':{'Clean':sum(bool(v['selected']) for v in clean.values()),'Random':sum(bool(v['selected']) for v in rand.values()),'Clean_forced':sum(bool(v['forced']) for v in clean.values()),'Random_forced':sum(bool(v['forced']) for v in rand.values()),'same_forced':{q for q in qids if clean[q]['forced']}=={q for q in qids if rand[q]['forced']}},'source_code':CODE.copy(),'formal_records':provenance}
    result.update(bootstrap_replay=replay,upstream_outside_0_1_by_arm=dict(outside_unit),upstream_mismatch_unique_qids=len({d['qid'] for d in differences}),frozen_prediction_answer_matches=frozen_answer_matches)
    wc('m3_official_scorer_examples.csv',[examples[k] for k in ('full','partial','zero')])
    if differences: wc('m3_upstream_scorer_discrepancies.csv',differences)
    wj('m3_scoring_replay.json',result)
    print('M3 local replay mismatches',len(local_diffs),'upstream mismatches',len(differences),flush=True)
    print('M3 S3',result['S3'],flush=True)
    print('M3 upstream means',result['upstream_means_percent'],flush=True)

def endpoints_stage():
    controls=jl(CONTROL+'selector_endpoint_official_gold_eval_details_v1.jsonl')
    qids=sorted(controls);assert len(qids)==1096
    old=j(CONTROL+'selector_endpoint_pairwise_significance_v1.json')['comparisons']
    ctl=pure('evaluate_mmqa_dev1096_a0_a4_a6_selector_endpoint_ablation_v1.py',['paired_bootstrap_delta','exact_mcnemar_p'],constants=('BOOTSTRAP_REPLICATES','EXPECTED_N'))
    rows=[];summaries={};stored_replay=[]
    init=np.array([controls[q]['initial_em'] for q in qids])
    for v in ('A0','A4','A6'):
        em=np.array([controls[q][v]['final_em'] for q in qids]);f1=np.array([controls[q][v]['final_f1'] for q in qids]);sel=np.array([controls[q][v]['selected'] for q in qids])
        assert sel.sum()==220
        finals=jl(CONTROL+f'{v.lower()}_dev1096_endpoint_final_predictions_before_gold_v1.jsonl')
        assert set(finals)==set(controls)
        # Exact frozen answers are checked independently of the score summary.
        ans_keys=[k for k in ('final_answer','effective_final_answer','effective_answer') if k in finals[qids[0]]]
        answer_match=all(finals[q][ans_keys[0]]==controls[q][v]['final_answer'] for q in qids) if ans_keys else None
        summaries[v]={'Errors@220':int(((1-init)*sel).sum()),'EM_percent':float(em.mean()*100),'F1_percent':float(f1.mean()*100),'W2R':int(np.sum((init==0)&(em==1))),'R2W':int(np.sum((init==1)&(em==0))),'Net':int(em.sum()-init.sum()),'selected':int(sel.sum()),'final_predictions_match_scored_answers':answer_match}
    for left,right in [('A4','A0'),('A6','A0'),('A6','A4')]:
        rec=old[f'{left}_vs_{right}'];a=np.array([controls[q][left]['final_em'] for q in qids]);b=np.array([controls[q][right]['final_em'] for q in qids]);better=int(np.sum((a==1)&(b==0)));worse=int(np.sum((a==0)&(b==1)));p=ctl['exact_mcnemar_p'](better,worse)
        ok=p==rec['em']['mcnemar_exact_two_sided_p'] and better==rec['em']['left_better'] and worse==rec['em']['right_better']
        replay={}
        for metric,field,seed in [('EM','final_em',260817),('F1','final_f1',260818)]:
            res=ctl['paired_bootstrap_delta']([controls[q][left][field] for q in qids],[controls[q][right][field] for q in qids],seed)
            original=rec['em' if metric=='EM' else 'f1']['paired_bootstrap']
            match=comparable([res['observed_delta'],*res['ci95']],[original['observed_delta'],*original['ci95']]);ok &= match;replay[metric]={'computed':res,'stored':original,'matches':match}
        # Requested direction is reverse of saved original: delta -> -delta;
        # [low,high] -> [-high,-low], discordants swapped, p unchanged.
        emr=rec['em']['paired_bootstrap'];f1r=rec['f1']['paired_bootstrap']
        rows.append({'comparison':f'{right} - {left}','N':1096,'em_diff_pp':-emr['observed_delta_percentage_points'],'em_ci_low_pp':-emr['ci95_percentage_points'][1],'em_ci_high_pp':-emr['ci95_percentage_points'][0],'f1_diff_pp':-f1r['observed_delta_percentage_points'],'f1_ci_low_pp':-f1r['ci95_percentage_points'][1],'f1_ci_high_pp':-f1r['ci95_percentage_points'][0],'left_better_discordant':worse,'right_better_discordant':better,'mcnemar_exact_two_sided_p':p,'iterations':10000,'em_seed':260817,'f1_seed':260818,'em_ci_contains_zero':emr['ci95'][0]<=0<=emr['ci95'][1],'f1_ci_contains_zero':f1r['ci95'][0]<=0<=f1r['ci95'][1],'full_Dev1096':True,'fixed_generation':True,'original_direction':f'{left} - {right}','source_artifact':str(ROOT/(CONTROL+'selector_endpoint_pairwise_significance_v1.json')),'status':'PASS' if ok else 'MISMATCH — preserve original'})
        stored_replay.append({'comparison':f'{left} - {right}','metrics':replay,'matches':ok})
        print('Section 4.6',left,right,'matches',ok,flush=True)
    wc('section46_pairwise_endpoint_statistics.csv',rows)
    wj('section46_replay_details.json',{'summaries':summaries,'replay':stored_replay})
    # Locked1341: fixed per-QID scores, five-seed fractional comparator first.
    details=jl(LOCK+'mmqa_clean_risk_v2_one_time_official_eval_details_v1.jsonl')
    qids=sorted(details);assert len(qids)==1341
    mm=pure('audit_mmqa_v3_locked1341_frozen_boundary_v1_1.py',['stringify_answer','normalize_answer','exact_match','token_f1','best_score'])
    locked=pure('evaluate_mmqa_clean_risk_v2_one_time_official_v1.py',['paired_bootstrap_delta','exact_mcnemar_p'],constants=('BOOTSTRAP_SAMPLES',))
    others=pure('experiments/mmqa_g0_20_endpoint/run_g0_20_endpoint.py',['paired_bootstrap','exact_mcnemar_p'],constants=('BOOTSTRAP_REPS',))
    sig=j(LOCK+'mmqa_clean_risk_v2_significance_v1.json')
    gsig=j('experiments/mmqa_g0_20_endpoint/g0_20_statistics.json')
    isig=j('experiments/mmqa_instability20_endpoint/statistical_comparisons.json')
    x={m:{'EM':[],'F1':[]} for m in ('Clean-B3','Initial','Random mean','VC-20','G0-20','Instability-20')}
    rseeds={str(s):{'EM':[],'F1':[]} for s in range(20260822,20260827)}
    final=jl('outputs/mmqa_clean_risk_v2_final_prediction_merge_before_gold_v1/clean_v2_locked1341_final_predictions_frozen_before_gold_v1.jsonl')
    assert set(final)==set(details)
    gpred=jl('experiments/mmqa_g0_20_endpoint/g0_20_final_predictions.jsonl');ipred=jl('experiments/mmqa_instability20_endpoint/final_predictions.jsonl')
    assert set(gpred)==set(ipred)==set(details)
    score_mismatch=Counter()
    for q in qids:
        row=details[q]
        assert final[q]['final_answer']==row['clean_v2_answer']
        for m,prefix in [('Clean-B3','clean_v2'),('Initial','initial')]:
            e,f=mm['best_score'](row[prefix+'_answer'],row['gold_answers'])
            if e!=row[prefix+'_em'] or abs(f-row[prefix+'_f1'])>1e-12: score_mismatch[m]+=1
            x[m]['EM'].append(row[prefix+'_em']);x[m]['F1'].append(row[prefix+'_f1'])
        for metric,key in [('EM','em'),('F1','f1')]:
            x['Random mean'][metric].append(statistics.mean(row['random20'][s][key] for s in rseeds))
            x['VC-20'][metric].append(row['vc20']['vc20_qid_ascending'][key])
            for s in rseeds:rseeds[s][metric].append(row['random20'][s][key])
        for m,rows_ in [('G0-20',gpred),('Instability-20',ipred)]:
            e,f=mm['best_score'](rows_[q]['final_answer'],row['gold_answers']);x[m]['EM'].append(e);x[m]['F1'].append(f)
    reports=[]
    specifications=[
      ('Clean-B3','Initial',sig['primary_pairwise']['initial'],locked['paired_bootstrap_delta'],'f1',False),
      ('Clean-B3','Random mean',sig['random20_seed_mean_fractional_comparison'],locked['paired_bootstrap_delta'],'f1',False),
      ('G0-20','Clean-B3',gsig['g0_20_vs_clean_b3'],others['paired_bootstrap'],'token_f1',True),
      ('Instability-20','Clean-B3',isig['instability20_vs_clean_b3'],others['paired_bootstrap'],'token_f1',False),
      ('Clean-B3','VC-20',sig['primary_pairwise']['vc20_qid_ascending'],locked['paired_bootstrap_delta'],'f1',False),
    ]
    for left,right,rec,fun,f1name,reverse in specifications:
        ok=True; results={}; saved={}
        for metric,rkey,seed in [('EM','em',20260828),('F1',f1name,20260829)]:
            print('Locked1341 replay',left,right,metric,flush=True)
            res=fun(x[left][metric],x[right][metric],seed);orig=rec[rkey]['paired_bootstrap']
            match=comparable([res['delta_pp'],res['ci95_low_pp'],res['ci95_high_pp']],[orig['delta_pp'],orig['ci95_low_pp'],orig['ci95_high_pp']])
            ok &= match
            if reverse: results[metric]={'delta_pp':-orig['delta_pp'],'ci_low_pp':-orig['ci95_high_pp'],'ci_high_pp':-orig['ci95_low_pp']}
            else:results[metric]={'delta_pp':orig['delta_pp'],'ci_low_pp':orig['ci95_low_pp'],'ci_high_pp':orig['ci95_high_pp']}
            saved[metric]={'record':orig,'replay':res,'matches':match}
        discord=None
        if right!='Random mean':
            a=np.array(x[left]['EM']);b=np.array(x[right]['EM']);ab=int(np.sum((a==1)&(b==0)));ba=int(np.sum((a==0)&(b==1)));p=locked['exact_mcnemar_p'](ab,ba)
            pkey='mcnemar_exact_p' if 'mcnemar_exact_p' in rec['em'] else 'mcnemar_exact_two_sided_p'
            ok &= abs(p-rec['em'][pkey])<1e-14
            discord={'left_better':ba if reverse else ab,'right_better':ab if reverse else ba,'p':p,'stored_p':rec['em'][pkey]}
        reports.append({'comparison':f'{right} minus {left}' if reverse else f'{left} minus {right}','stored_direction':f'{left} minus {right}','metrics':results,'saved_replay':saved,'discordants':discord,'N':1341,'status':'PASS' if ok else 'MISMATCH — preserve original'})
        print('Locked comparison completed',left,right,ok,flush=True)
    wj('locked1341_replay_details.json',{'comparisons':reports,'means_percent':{m:{k:statistics.mean(v)*100 for k,v in d.items()} for m,d in x.items()},'five_random_seed_metrics':{s:{k:statistics.mean(v)*100 for k,v in d.items()} for s,d in rseeds.items()},'clean_initial_score_mismatches':dict(score_mismatch),'N':1341})

def finish(stage):
    for p,v in SOURCES.items():
        v['sha256_after']=sha(Path(p));v['unchanged']=v['sha256_before']==v['sha256_after']
    wj(f'source_hashes_{stage}.json',SOURCES)
    wj(f'source_code_positions_{stage}.json',CODE)
    if not all(v['unchanged'] for v in SOURCES.values()): raise RuntimeError('input changed during read-only audit')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['risk','m3','endpoints'],required=True);args=ap.parse_args()
    {'risk':risk_stage,'m3':m3_stage,'endpoints':endpoints_stage}[args.stage]()
    finish(args.stage)

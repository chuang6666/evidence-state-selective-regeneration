#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Clean Risk V2 Dev1096 Exact Top-6 held-out verifier inference only."""

from __future__ import annotations
import hashlib, json, os, random
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import joblib
import numpy as np
import torch
import run_mmqa_natural_frozen_verifier_v2_inference_v1 as v2
import run_mmqa_t4_operator_aware_verifier_v1_fixed as t4
import run_mmqa_verifier_v3_t1_t2_natural_adapter_corrected_v2 as v3

ROOT = Path(__file__).resolve().parent
TS = ["T1", "T2", "T3", "T4", "T5"]
EXPECTED_QIDS = 1096
EXPECTED_INST = {"T1":565,"T2":110,"T3":260,"T4":15,"T5":801}
EXPECTED_PRESENT = {"T1":396,"T2":84,"T3":122,"T4":15,"T5":801}
EXPECTED_MISSING = {"T1":296,"T2":418,"T3":340,"T4":447,"T5":295}
EXPECTED_INSTANCE_SHA = "61E7E1DE3846C0E8CB245E58F38D73CCC1374BCD81290C6C9DDCEF8D1483ED53"
EXPECTED_STATE_SHA = "6129C8B9D3612C74B70FC38C4C9D5337A9EFB6C207966579A4244786F561E665"
MAX_LENGTH, BATCH_SIZE, SEED = 512, 4, 42

IN_ROOT = ROOT/"outputs"/"mmqa_clean_risk_v2_dev1096_verifier_reconstruction_preflight_v1"
INSTANCE_FILE = IN_ROOT/"mmqa_clean_v2_dev1096_verifier_instances_v1.jsonl"
MANIFEST_FILE = IN_ROOT/"mmqa_clean_v2_dev1096_qid_state_manifest_v1.jsonl"
MISSING_FILE = IN_ROOT/"mmqa_clean_v2_dev1096_missing_transition_states_v1.jsonl"
STATE_FILE = IN_ROOT/"mmqa_clean_v2_dev1096_exact_top6_natural_states_frozen_v1.jsonl"
IN_HASH = IN_ROOT/"SHA256SUMS.json"
V1_THRESHOLD = ROOT/"outputs"/"mmqa_natural_failure_localization_v1"/"mmqa_natural_failure_thresholds_v1.json"
T4_ROOT = ROOT/"outputs"/"mmqa_clean_risk_v2_t4_rebuild_v1"
T4_SUMMARY = T4_ROOT/"t4_deterministic_rebuild_summary_v1.json"
T4_THRESHOLD = T4_ROOT/"t4_oof_threshold_reliability_v2.json"
T4_HASH = T4_ROOT/"SHA256SUMS.json"
OUT = ROOT/"outputs"/"mmqa_clean_risk_v2_dev1096_verifier_inference_v1"
PREFLIGHT = OUT/"dev1096_exact_top6_verifier_preflight_v1.json"
T12_RAW = OUT/"t1_t2_v3_adapter_instance_scores_v1.jsonl"
T35_RAW = OUT/"t3_t5_frozen_v2_instance_scores_v1.jsonl"
T4_RAW = OUT/"t4_rebuilt_operator_instance_scores_v1.jsonl"
INST_OUT = OUT/"dev1096_verifier_instance_scores_v1.jsonl"
TRANS_OUT = OUT/"dev1096_verifier_transition_scores_v1.jsonl"
QID_OUT = OUT/"dev1096_verifier_qid_states_v1.jsonl"
SUMMARY = OUT/"dev1096_exact_top6_verifier_inference_summary_v1.json"
HASH_OUT = OUT/"SHA256SUMS.json"
FIELDS = {
 "T1":{"question","visual_evidence_id","visual_evidence_text","candidate_grounding_claim"},
 "T2":{"question","upstream_claim","bridge_evidence_id","candidate_bridge_evidence_text","downstream_claim"},
 "T3":{"question","branch_evidence_id","branch_visual_evidence_text","candidate_branch_claim"},
 "T4":{"question","aggregation_operator","verified_branch_values","candidate_aggregation_result"},
 "T5":{"question","verified_upstream_claims","candidate_final_answer"},
}

def load_json(p: Path):
    if not p.is_file(): raise FileNotFoundError(p)
    x=json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(x,dict): raise RuntimeError(f"Expected object: {p}")
    return x

def load_jsonl(p: Path):
    if not p.is_file(): raise FileNotFoundError(p)
    out=[]
    for n,line in enumerate(p.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        x=json.loads(line)
        if not isinstance(x,dict): raise RuntimeError(f"Non-object {p}:{n}")
        out.append(x)
    return out

def write_json(p: Path,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def write_jsonl(p: Path,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8",newline="\n") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")

def sha(p: Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest().upper()

def finfo(p: Path): return {"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)}
def eq(a,b,name):
    if a!=b: raise RuntimeError(f"PREFLIGHT_MISMATCH {name}: {a!r} != {b!r}")

def preflight():
    req=[INSTANCE_FILE,MANIFEST_FILE,MISSING_FILE,STATE_FILE,IN_HASH,v2.OOF_PRED_FILE,
         v3.SUMMARY_FILE,V1_THRESHOLD,T4_SUMMARY,T4_THRESHOLD,T4_HASH]
    for p in req:
        if not p.is_file(): raise FileNotFoundError(p)
    eq(sha(INSTANCE_FILE),EXPECTED_INSTANCE_SHA,"Exact Top-6 instance SHA")
    eq(sha(STATE_FILE),EXPECTED_STATE_SHA,"Exact Top-6 state SHA")
    recorded=load_json(IN_HASH)["files"]
    for p in [INSTANCE_FILE,MANIFEST_FILE,MISSING_FILE,STATE_FILE]:
        eq(sha(p),recorded[p.name],f"recorded SHA {p.name}")

    t4sum=load_json(T4_SUMMARY)
    eq(t4sum["status"],"MMQA_CLEAN_RISK_V2_T4_DETERMINISTIC_REBUILD_V1_COMPLETE","T4 status")
    eq(t4sum["numerically_identical_to_frozen_predictions"],True,"T4 numerical gate")
    t4hash=load_json(T4_HASH)["files"]
    assets=[]
    for fold in range(5):
        base=v2.VERIFIER_ROOT/f"fold_{fold}"/"best_model.pt"
        tok=v2.VERIFIER_ROOT/f"fold_{fold}"/"tokenizer"
        adapter=v3.OUT_ROOT/f"fold_{fold}"/"best_adapter.pt"
        bundle=T4_ROOT/f"fold_{fold}"/"t4_operator_aware_bundle_v1.joblib"
        if not base.is_file() or not tok.is_dir() or not adapter.is_file() or not bundle.is_file():
            raise RuntimeError(f"Missing verifier asset fold {fold}")
        rel=str(bundle.relative_to(T4_ROOT))
        eq(sha(bundle),t4hash[rel],f"T4 bundle SHA fold {fold}")
        b=joblib.load(bundle); eq(int(b["fold"]),fold,f"T4 bundle fold {fold}")
        assets.append({"fold":fold,"v2_model":finfo(base),"v2_tokenizer":str(tok),
          "v3_adapter":finfo(adapter),"t4_bundle":finfo(bundle),"t4_mode":b["mode"],"t4_C":b["C"]})

    v3sum=load_json(v3.SUMMARY_FILE)
    deploy=v3sum["architecture"]["deployment_decisions"]
    eq(bool(deploy["T1"]["deploy_adapter"]),True,"T1 deploy")
    eq(bool(deploy["T2"]["deploy_adapter"]),True,"T2 deploy")

    inst=load_jsonl(INSTANCE_FILE); mans=load_jsonl(MANIFEST_FILE)
    missrows=load_jsonl(MISSING_FILE); states=load_jsonl(STATE_FILE)
    foldmap_all=v2.build_qid_fold_map(load_jsonl(v2.OOF_PRED_FILE))
    qids={str(r["qid"]) for r in mans}
    eq(len(qids),EXPECTED_QIDS,"QID count")
    eq({str(r["qid"]) for r in states},qids,"state QIDs")
    if not qids.issubset(foldmap_all): raise RuntimeError("Missing held-out QID fold")
    qfold={q:int(foldmap_all[q]) for q in qids}
    manmap={str(r["qid"]):r for r in mans}
    ids=[str(r["instance_id"]) for r in inst]; eq(len(ids),len(set(ids)),"unique instance IDs")
    counts=dict(sorted(Counter(str(r["transition_type"]) for r in inst).items()))
    eq(counts,EXPECTED_INST,"instance counts")
    schema=[]; folds=[]; membership=[]
    for r in inst:
        q=str(r["qid"]); t=str(r["transition_type"]); f=r.get("fields")
        if not isinstance(f,dict) or not FIELDS[t].issubset(f): schema.append(r["instance_id"])
        if int(r.get("heldout_verifier_fold",-1))!=qfold[q]: folds.append(r["instance_id"])
        if t not in manmap[q]["required_and_present"]: membership.append(r["instance_id"])
    eq(schema,[],"schema"); eq(folds,[],"folds"); eq(membership,[],"membership")
    present=Counter();
    for r in mans: present.update(str(t) for t in r["required_and_present"])
    missing=Counter(str(r["transition_type"]) for r in missrows)
    eq(dict(sorted(present.items())),EXPECTED_PRESENT,"required present")
    eq(dict(sorted(missing.items())),EXPECTED_MISSING,"missing")

    old=load_json(V1_THRESHOLD)["thresholds"]
    t4new=load_json(T4_THRESHOLD)["frozen_v2_values"]
    contract={}
    for t in TS:
        s=t4new if t=="T4" else old[t]
        contract[t]={"threshold":float(s["threshold"]),"reliability_weight":float(s["reliability_weight"]),
          "source":"Clean V2 rebuilt T4 OOF" if t=="T4" else str(s.get("source"))}
    eq(contract["T4"]["threshold"],float(old["T4"]["threshold"]),"T4 threshold identity")
    eq(contract["T4"]["reliability_weight"],float(old["T4"]["reliability_weight"]),"T4 reliability identity")
    payload={"status":"MMQA_CLEAN_RISK_V2_DEV1096_EXACT_TOP6_VERIFIER_PREFLIGHT_PASS",
      "qids":len(qids),"instances":len(inst),"per_transition_instances":counts,
      "per_transition_required_present_qids":dict(sorted(present.items())),
      "per_transition_missing_required_slots":dict(sorted(missing.items())),
      "heldout_fold_qid_counts":dict(sorted(Counter(qfold[q] for q in qids).items())),
      "schema_failures":0,"fold_failures":0,"membership_failures":0,
      "exact_top6_input_sha256":sha(INSTANCE_FILE),"exact_top6_state_sha256":sha(STATE_FILE),
      "threshold_contract":contract,"model_assets":assets,"v3_deployment_decisions":deploy,
      "transformers_offline":True,"verifier_training":False,"external_api_calls":0,
      "gold_opened":False,"locked_1341_opened_or_evaluated":False,
      "answer_risk_training":False,"localization":False,"v1_modified":False}
    write_json(PREFLIGHT,payload)
    return inst,mans,qfold,contract

def diagnostic(row,contract):
    t=row["transition"]; p=float(row["valid_prob"]); th=contract[t]["threshold"]; rel=contract[t]["reliability_weight"]
    invalid=p<th; margin=max(0.0,(th-p)/max(th,1e-6))
    return {**row,"score_available":True,"threshold":th,"reliability_weight":rel,
      "invalid":bool(invalid),"severity":float(margin*rel),"threshold_source":contract[t]["source"]}

def main():
    if SUMMARY.is_file() and load_json(SUMMARY).get("status")=="MMQA_CLEAN_RISK_V2_DEV1096_EXACT_TOP6_VERIFIER_INFERENCE_V1_COMPLETE":
        print("Existing completed verifier inference found; refusing duplicate inference.")
        print("Summary:",SUMMARY); print("STATUS = MMQA_CLEAN_RISK_V2_DEV1096_EXACT_TOP6_VERIFIER_INFERENCE_V1_COMPLETE"); return
    OUT.mkdir(parents=True,exist_ok=True)
    inst,mans,qfold,contract=preflight()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*120); print("MMQA CLEAN RISK V2 — DEV1096 EXACT TOP-6 VERIFIER INFERENCE V1")
    print("Preflight PASS | device",device,"| qids",len(mans),"| instances",len(inst))
    print("API=0 Gold=0 locked=0 training=0 localization=0"); print("-"*120)

    deploy=load_json(v3.SUMMARY_FILE)["architecture"]["deployment_decisions"]
    args=SimpleNamespace(adapter_dropout=.10,max_length=MAX_LENGTH,encode_batch_size=BATCH_SIZE)
    i12=[r for r in inst if r["transition_type"] in {"T1","T2"}]
    print("Scoring T1/T2:",len(i12))
    r12=v3.score_natural_with_v3(i12,qfold,deploy,args,device)
    eq(len(r12),len(i12),"T1/T2 score count")
    write_jsonl(T12_RAW,sorted(r12,key=lambda r:(r["qid"],r["transition"],int(r.get("component_index") or 0))))

    i35=[r for r in inst if r["transition_type"] in {"T3","T5"}]; byfold=defaultdict(list)
    for r in i35: byfold[qfold[str(r["qid"])]].append(r)
    r35=[]
    for fold in range(5):
        rows=sorted(byfold[fold],key=lambda r:(str(r["qid"]),v2.T2ID[r["transition_type"]],int(r.get("component_index") or 0)))
        print(f"Scoring T3/T5 fold {fold}: {len(rows)}")
        if rows: r35.extend(v2.score_fold(fold,rows,BATCH_SIZE,MAX_LENGTH,device,True))
    eq(len(r35),len(i35),"T3/T5 score count")
    write_jsonl(T35_RAW,sorted(r35,key=lambda r:(r["qid"],r["transition"],int(r.get("component_index") or 0))))

    i4=[r for r in inst if r["transition_type"]=="T4"]; r4=[]
    for fold in range(5):
        rows=[r for r in i4 if qfold[str(r["qid"])]==fold]
        if not rows: continue
        bp=T4_ROOT/f"fold_{fold}"/"t4_operator_aware_bundle_v1.joblib"; b=joblib.load(bp)
        recs=[t4.natural_row_to_record(r,qfold) for r in rows]
        raw,probs=t4.predict_model(b["feature_builder"],b["ranker"],b["calibrator"],recs)
        for rec,rs,p in zip(recs,raw.tolist(),probs.tolist()):
            r4.append({"instance_id":rec["instance_id"],"qid":rec["qid"],"fold":fold,"transition":"T4",
              "component_index":rec.get("component_index"),"rank_score":float(rs),"prob_positive":float(p),
              "selected_mode":b["mode"],"selected_C":b["C"],"bundle_sha256":sha(bp)})
    eq(len(r4),len(i4),"T4 score count"); write_jsonl(T4_RAW,sorted(r4,key=lambda r:r["qid"]))

    combined=[]
    for r in r12: combined.append({"instance_id":r["instance_id"],"qid":r["qid"],"fold":int(r["fold"]),
      "transition":r["transition"],"component_index":r.get("component_index"),"valid_prob":float(r["v3_prob_positive"]),
      "score_source":"V3_NATURAL_ADAPTER","base_v2_prob":float(r["v2_prob_positive_recomputed"]),
      "adapter_logit_delta":float(r["adapter_logit_delta_deployed"]),"adapter_deployed":bool(r["adapter_deployed"])})
    for r in r35: combined.append({"instance_id":r["instance_id"],"qid":r["qid"],"fold":int(r["fold"]),
      "transition":r["transition"],"component_index":r.get("component_index"),"valid_prob":float(r["prob_positive"]),
      "score_source":"FROZEN_V2","logit":float(r["logit"])})
    for r in r4: combined.append({"instance_id":r["instance_id"],"qid":r["qid"],"fold":int(r["fold"]),
      "transition":"T4","component_index":r.get("component_index"),"valid_prob":float(r["prob_positive"]),
      "score_source":"CLEAN_V2_T4_OPERATOR_AWARE_REBUILD","rank_score":float(r["rank_score"])})
    combined.sort(key=lambda r:(r["qid"],TS.index(r["transition"]),int(r.get("component_index") or 0)))
    eq(len(combined),len(inst),"combined count"); eq(len({r["instance_id"] for r in combined}),len(inst),"unique score IDs")
    for r in combined: eq(r["fold"],qfold[str(r["qid"])],f"score fold {r['instance_id']}")
    write_jsonl(INST_OUT,combined)

    grouped=defaultdict(list)
    for r in combined: grouped[(str(r["qid"]),r["transition"])].append(r)
    trans=[]
    for (qid,t),items in sorted(grouped.items(),key=lambda kv:(kv[0][0],TS.index(kv[0][1]))):
        probs=np.asarray([float(x["valid_prob"]) for x in items])
        trans.append(diagnostic({"qid":qid,"fold":qfold[qid],"transition":t,"n_components":len(items),
          "valid_prob":float(probs.min()),"mean_prob":float(probs.mean()),"max_prob":float(probs.max()),
          "aggregation":"min_component_probability","score_source":items[0]["score_source"]},contract))
    transmap={(r["qid"],r["transition"]):r for r in trans}; write_jsonl(TRANS_OUT,trans)

    qstates=[]; status_counts=Counter(); invalid_counts=Counter(); available=Counter()
    for m in sorted(mans,key=lambda r:str(r["qid"])):
        q=str(m["qid"]); present=set(m["required_and_present"]); missing=set(m["required_but_missing"]); extra=set(m["present_but_not_required"])
        states={}
        for t in TS:
            c=contract[t]
            if t in present:
                s=transmap.get((q,t))
                if s is None: raise RuntimeError(f"Missing score {q}/{t}")
                st={"required":True,"state_status":"PRESENT","score_available":True,"valid_prob":s["valid_prob"],
                  "mean_prob":s["mean_prob"],"max_prob":s["max_prob"],"n_components":s["n_components"],
                  "score_source":s["score_source"],"threshold":s["threshold"],"reliability_weight":s["reliability_weight"],
                  "invalid":s["invalid"],"severity":s["severity"]}; available[t]+=1
                if st["invalid"]: invalid_counts[t]+=1
            elif t in missing:
                st={"required":True,"state_status":"MISSING_REQUIRED_TRANSITION_STATE","score_available":False,
                  "valid_prob":None,"n_components":0,"score_source":None,"threshold":c["threshold"],
                  "reliability_weight":c["reliability_weight"],"invalid":None,"severity":None}
            else:
                st={"required":False,"state_status":"PRESENT_BUT_NOT_REQUIRED" if t in extra else "NOT_REQUIRED",
                  "score_available":False,"valid_prob":None,"n_components":None,"score_source":None,
                  "threshold":c["threshold"],"reliability_weight":c["reliability_weight"],"invalid":None,"severity":None}
            states[t]=st; status_counts[st["state_status"]]+=1
        qstates.append({"qid":q,"heldout_verifier_fold":qfold[q],"transitions":states,
          "localization_decision":None,"localization_executed":False,"gold_used":False})
    eq(len(qstates),EXPECTED_QIDS,"QID output coverage"); write_jsonl(QID_OUT,qstates)

    per={}
    for t in TS:
        ii=[r for r in combined if r["transition"]==t]; tt=[r for r in trans if r["transition"]==t]
        per[t]={"instances_input_and_scored":len(ii),"required_present_qids":len(tt),"score_available_qids":available[t],
          "invalid_qids":invalid_counts[t],"required_missing_slots":EXPECTED_MISSING[t],"threshold":contract[t]["threshold"],
          "reliability_weight":contract[t]["reliability_weight"],
          "mean_valid_prob":float(np.mean([r["valid_prob"] for r in tt])) if tt else None,
          "median_valid_prob":float(np.median([r["valid_prob"] for r in tt])) if tt else None,
          "mean_severity":float(np.mean([r["severity"] for r in tt])) if tt else None,
          "max_severity":float(np.max([r["severity"] for r in tt])) if tt else None}
    summary={"status":"MMQA_CLEAN_RISK_V2_DEV1096_EXACT_TOP6_VERIFIER_INFERENCE_V1_COMPLETE",
      "qid_coverage":len(qstates),"verifier_instances_input":len(inst),"verifier_instances_scored":len(combined),
      "qid_transition_scores":len(trans),"heldout_fold_qid_counts":dict(sorted(Counter(qfold[r["qid"]] for r in qstates).items())),
      "heldout_fold_instance_counts":dict(sorted(Counter(r["fold"] for r in combined).items())),"per_transition":per,
      "state_status_counts":dict(status_counts),"missing_and_empty_statistics":{
        "required_missing_slots_total":sum(EXPECTED_MISSING.values()),"required_missing_slots_per_transition":EXPECTED_MISSING,
        "qids_with_any_required_missing":sum(bool(r["required_but_missing"]) for r in mans),
        "qids_with_zero_required_present_states":sum(not r["required_and_present"] for r in mans),
        "qids_with_zero_present_natural_transitions":sum(not r["present_natural_transitions"] for r in mans),
        "empty_states_are_not_assigned_probability_zero":True},
      "input_integrity":{"exact_top6_instance_sha256":sha(INSTANCE_FILE),"exact_top6_state_sha256":sha(STATE_FILE),
        "schema_failures":0,"qid_fold_failures":0,"score_coverage_failures":0},
      "scoring_contract":{"T1_T2":"frozen V2 + held-out V3 natural adapter","T3_T5":"held-out frozen V2; CUDA FP16 when available",
        "T4":"held-out Clean V2 numerically-equivalent operator-aware bundle","aggregation":"minimum probability",
        "device":str(device),"max_length":MAX_LENGTH,"batch_size":BATCH_SIZE},
      "external_api_calls":0,"gold_opened":False,"locked_1341_opened_or_evaluated":False,"verifier_training":False,
      "answer_risk_training":False,"localization":False,"v1_modified":False,
      "next_step":"STOP_FOR_MANUAL_REVIEW_BEFORE_LOCALIZATION"}
    write_json(SUMMARY,summary)
    outs=[PREFLIGHT,T12_RAW,T35_RAW,T4_RAW,INST_OUT,TRANS_OUT,QID_OUT,SUMMARY]
    hashes={"namespace":OUT.name,"files":{p.name:sha(p) for p in outs},"source_files":{
      str(p.relative_to(ROOT)):sha(p) for p in [INSTANCE_FILE,MANIFEST_FILE,MISSING_FILE,STATE_FILE,v2.OOF_PRED_FILE,
      v3.SUMMARY_FILE,V1_THRESHOLD,T4_SUMMARY,T4_THRESHOLD]},"status":summary["status"],"external_api_calls":0,
      "gold_opened":False,"locked_1341_opened_or_evaluated":False,"answer_risk_training":False,
      "localization":False,"v1_modified":False}
    write_json(HASH_OUT,hashes)
    print("-"*120); print("FINAL qids",len(qstates),"/1096 instances",len(combined),"/",len(inst),"transitions",len(trans))
    print("Missing slots",sum(EXPECTED_MISSING.values()),"invalid per T",dict(invalid_counts))
    print("API=0 Gold=0 locked=0 training=0 localization=0")
    print("Summary:",SUMMARY); print("SHA:",HASH_OUT); print("NEXT STEP = STOP FOR MANUAL REVIEW BEFORE LOCALIZATION")
    print("STATUS =",summary["status"])

if __name__=="__main__": main()

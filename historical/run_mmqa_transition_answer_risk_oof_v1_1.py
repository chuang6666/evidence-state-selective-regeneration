#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import importlib.util, json, math, re, sys, unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
HELPER_FILE = ROOT / "run_mmqa_v6_1_topology_routed_raw_visual.py"
OUT_DIR = ROOT / "outputs" / "mmqa_transition_answer_risk_oof_v1"
INV_FILE = OUT_DIR / "risk_feature_inventory_v1.json"
OOF_FILE = OUT_DIR / "risk_oof_predictions_v1.jsonl"
SUM_FILE = OUT_DIR / "risk_oof_summary_v1.json"

EXPECTED_N = 1096
TRANSITIONS = ("T1", "T2", "T3", "T4", "T5")
LEAK = ("gold","label","correct","incorrect","exact_match","em_score","token_f1","eval","oracle","ground_truth","groundtruth")
RAW_ANSWER = ("draft_answer","final_answer","candidate_answer","gold_answer","reference_answer")
SEED = 260817

def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def st(x): return "" if x is None else str(x).strip()

def norm(x):
    s = unicodedata.normalize("NFKC", st(x)).lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def leaky(path):
    p = path.lower()
    return any(x in p for x in LEAK) or any(x in p for x in RAW_ANSWER) or p.endswith(".qid") or p=="qid"

def tmentions(path, val=None):
    blob = path.upper() + (" " + str(val).upper() if isinstance(val,str) else "")
    return [t for t in TRANSITIONS if re.search(rf"(?<![A-Z0-9]){t}(?![A-Z0-9])", blob)]

def scalar(v):
    if isinstance(v,bool): return int(v)
    if isinstance(v,(int,float)) and not isinstance(v,bool):
        try:
            if math.isnan(float(v)) or math.isinf(float(v)): return None
        except Exception: pass
        return float(v)
    if isinstance(v,str) and 0 < len(v.strip()) <= 120: return v.strip()
    return None

def flatten(obj, prefix, general, per_t):
    if isinstance(obj,dict):
        for k,v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if not leaky(p): flatten(v,p,general,per_t)
        return
    if isinstance(obj,list):
        if leaky(prefix): return
        general[f"{prefix}.__len__"] = len(obj)
        if len(obj) <= 30:
            for item in obj:
                if isinstance(item,(str,int,float,bool)):
                    s = st(item)
                    ts = tmentions(prefix,s)
                    if ts:
                        for t in ts: per_t[t][f"{prefix}.__contains__{t}"] = 1
                    elif s and len(s) <= 60:
                        general[f"{prefix}.__contains__{s}"] = 1
        return
    if leaky(prefix): return
    v = scalar(obj)
    if v is None: return
    ts = tmentions(prefix,obj)
    if ts:
        for t in ts: per_t[t][prefix] = v
    else:
        general[prefix] = v

def required_transitions(row):
    for k in ("predicted_required_transitions_schema_constrained","predicted_required_transitions","required_transitions"):
        xs = row.get(k)
        if isinstance(xs,list):
            out=[st(x).upper() for x in xs if st(x).upper() in TRANSITIONS]
            if out: return list(dict.fromkeys(out))
    return []

def general_features(base):
    q, a = st(base.get("question")), st(base.get("draft_answer"))
    evs = base.get("retrieved_evidence")
    if not isinstance(evs,list): evs=[]
    mods=Counter(); lens=[]
    for e in evs:
        if not isinstance(e,dict): continue
        mods[st(e.get("modality")).lower() or "unknown"] += 1
        txt=(st(e.get("title"))+" "+st(e.get("content"))).strip()
        if txt: lens.append(len(txt))
    feats={
        "GEN.question_chars":len(q),
        "GEN.question_tokens":len(norm(q).split()),
        "GEN.draft_chars":len(a),
        "GEN.draft_tokens":len(norm(a).split()),
        "GEN.draft_single_token":int(len(norm(a).split())==1),
        "GEN.draft_numeric":int(bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?",norm(a)))),
        "GEN.evidence_count":len(evs),
        "GEN.evidence_mean_chars":float(np.mean(lens)) if lens else 0.0,
        "GEN.evidence_max_chars":max(lens) if lens else 0,
    }
    for m,n in mods.items(): feats[f"GEN.modality::{m}"]=n
    return feats

def build_parts(base, topo, loc):
    g=general_features(base)
    pt={t:{} for t in TRANSITIONS}
    req=set(required_transitions(topo))
    decision=st(loc.get("decision")).upper()
    localized=st(loc.get("localized_transition")).upper()
    g["DIAG.decision"]=decision or "UNKNOWN"
    g["DIAG.structural_missing"]=int(decision=="STRUCTURAL_MISSING")
    g["DIAG.verifier_invalid"]=int(decision=="VERIFIER_INVALID")
    g["DIAG.clean_or_uncertain"]=int(decision=="CLEAN_OR_UNCERTAIN")
    for t in TRANSITIONS:
        pt[t][f"{t}.required"]=int(t in req)
        pt[t][f"{t}.localized"]=int(localized==t)
        pt[t][f"{t}.localized_structural"]=int(localized==t and decision=="STRUCTURAL_MISSING")
        pt[t][f"{t}.localized_invalid"]=int(localized==t and decision=="VERIFIER_INVALID")
    flatten(topo,"TOPO",g,pt)
    flatten(loc,"LOC",g,pt)
    dummy={}
    base_pt={t:{} for t in TRANSITIONS}
    flatten(base,"BASE",dummy,base_pt)
    for t in TRANSITIONS: pt[t].update(base_pt[t])
    return g,pt

def merge(*ds):
    out={}
    for d in ds: out.update(d)
    return out

def variants(parts):
    g,pt=parts
    return {
        "B0_GENERAL": merge(g),
        "B1_GENERAL_T1": merge(g,pt["T1"]),
        "B2_GENERAL_T1_T4": merge(g,pt["T1"],pt["T2"],pt["T3"],pt["T4"]),
        "B3_FULL_T1_T5": merge(g,pt["T1"],pt["T2"],pt["T3"],pt["T4"],pt["T5"]),
        "B4_TRANSITION_ONLY": merge(pt["T1"],pt["T2"],pt["T3"],pt["T4"],pt["T5"]),
    }

def model():
    return Pipeline([
        ("vec",DictVectorizer(sparse=True)),
        ("clf",LogisticRegression(solver="liblinear",C=1.0,class_weight="balanced",max_iter=2000,random_state=SEED)),
    ])

def topstat(y,p,f):
    k=max(1,int(math.ceil(len(y)*f)))
    idx=np.argsort(-p)[:k]
    found=int(y[idx].sum())
    return {"fraction":f,"k":k,"precision":found/k,"recall":found/int(y.sum())}

def metrics(y,p):
    return {
        "roc_auc":float(roc_auc_score(y,p)),
        "average_precision":float(average_precision_score(y,p)),
        "population_error_rate":float(y.mean()),
        "top10":topstat(y,p,.10),
        "top20":topstat(y,p,.20),
        "top30":topstat(y,p,.30),
        "top40":topstat(y,p,.40),
    }

def main():
    h=imp("risk_helper",HELPER_FILE)
    for attr in ("BASE_FILE","TOPOLOGY_FILE","LOCALIZATION_FILE"):
        if not hasattr(h,attr): raise RuntimeError(f"Helper missing {attr}")

    base={h.safe_text(r.get("qid")):r for r in h.load_jsonl(h.BASE_FILE) if h.safe_text(r.get("qid"))}
    topo={h.safe_text(r.get("qid")):r for r in h.load_jsonl(h.TOPOLOGY_FILE) if h.safe_text(r.get("qid"))}
    loc={h.safe_text(r.get("qid")):r for r in h.load_jsonl(h.LOCALIZATION_FILE) if h.safe_text(r.get("qid"))}
    if len(base)!=EXPECTED_N or set(base)!=set(topo) or set(base)!=set(loc):
        raise RuntimeError(f"QID mismatch base={len(base)} topo={len(topo)} loc={len(loc)}")

    gold=h.load_gold()
    qids=sorted(base)
    parts={q:build_parts(base[q],topo[q],loc[q]) for q in qids}
    y=[]
    for q in qids:
        em,_=h.score_one(h.safe_text(base[q].get("draft_answer")),gold[q])
        y.append(int(em==0))
    y=np.asarray(y,dtype=int)

    group_names={}
    gen=set()
    for q in qids:
        gen.update(parts[q][0].keys())
    for t in TRANSITIONS:
        s=set()
        for q in qids: s.update(parts[q][1][t].keys())
        group_names[t]=sorted(s)

    OUT_DIR.mkdir(parents=True,exist_ok=True)
    INV_FILE.write_text(json.dumps({
        "n":len(qids),
        "initial_wrong":int(y.sum()),
        "initial_error_rate":float(y.mean()),
        "general_feature_count":len(gen),
        "transition_feature_counts":{t:len(group_names[t]) for t in TRANSITIONS},
        "transition_feature_examples":{t:group_names[t][:80] for t in TRANSITIONS},
        "sources":{"base":str(h.BASE_FILE),"topology":str(h.TOPOLOGY_FILE),"localization":str(h.LOCALIZATION_FILE)},
        "gold_or_correctness_input_features":"EXCLUDED",
    },ensure_ascii=False,indent=2),encoding="utf-8")

    byq={q:variants(parts[q]) for q in qids}
    names=list(byq[qids[0]].keys())
    oof={n:np.zeros(len(qids)) for n in names}
    foldid=np.zeros(len(qids),dtype=int)
    dims={n:[] for n in names}

    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    for fold,(tr,va) in enumerate(skf.split(np.zeros(len(y)),y),1):
        foldid[va]=fold
        for n in names:
            m=model()
            m.fit([byq[qids[i]][n] for i in tr],y[tr])
            oof[n][va]=m.predict_proba([byq[qids[i]][n] for i in va])[:,1]
            dims[n].append(len(m.named_steps["vec"].feature_names_))

    sm={n:{**metrics(y,oof[n]),"feature_dim_per_fold":dims[n]} for n in names}
    b1,b2,b3=sm["B1_GENERAL_T1"],sm["B2_GENERAL_T1_T4"],sm["B3_FULL_T1_T5"]
    role={
        "T2_T4_over_T1":{
            "roc_auc_delta":b2["roc_auc"]-b1["roc_auc"],
            "average_precision_delta":b2["average_precision"]-b1["average_precision"],
            "top20_precision_delta":b2["top20"]["precision"]-b1["top20"]["precision"],
            "top30_precision_delta":b2["top30"]["precision"]-b1["top30"]["precision"],
        },
        "T5_over_T1_T4":{
            "roc_auc_delta":b3["roc_auc"]-b2["roc_auc"],
            "average_precision_delta":b3["average_precision"]-b2["average_precision"],
            "top20_precision_delta":b3["top20"]["precision"]-b2["top20"]["precision"],
            "top30_precision_delta":b3["top30"]["precision"]-b2["top30"]["precision"],
        }
    }

    rows=[]
    for i,q in enumerate(qids):
        r={"qid":q,"fold":int(foldid[i]),"initial_wrong_label":int(y[i])}
        for n in names:r[f"risk_{n}"]=float(oof[n][i])
        rows.append(r)
    write_jsonl(OOF_FILE,rows)

    summary={
        "status":"MMQA_TRANSITION_TO_ANSWER_RISK_OOF_V1_COMPLETE",
        "api_calls":0,
        "m3vqa_accessed":False,
        "mmqa_heldout_1341_accessed":False,
        "mmqa_development_gold_used_as_label_only":True,
        "n":len(qids),
        "initial_wrong":int(y.sum()),
        "initial_correct":int(len(y)-y.sum()),
        "initial_error_rate":float(y.mean()),
        "cv":"5-fold stratified out-of-fold",
        "model":"DictVectorizer + fixed LogisticRegression",
        "variants":sm,
        "transition_role_analysis":role,
        "feature_inventory":str(INV_FILE),
        "oof_predictions":str(OOF_FILE),
    }
    SUM_FILE.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

    print("="*128)
    print("MMQA — TRANSITION-TO-ANSWER RISK OOF V1")
    print("="*128)
    print("API calls: 0")
    print("M3-VQA accessed: NO")
    print("MMQA held-out 1341 accessed: NO")
    print("MMQA development gold used: YES, LABEL ONLY")
    print(f"N={len(y)} | wrong={int(y.sum())} ({y.mean():.2%}) | correct={int(len(y)-y.sum())}")
    print("-"*128)
    print("FEATURE COUNTS:", {"GENERAL":len(gen), **{t:len(group_names[t]) for t in TRANSITIONS}})
    print("-"*128)
    print(f"{'VARIANT':24s} {'ROC-AUC':>8s} {'AP':>8s} {'Top20P':>8s} {'Top20R':>8s} {'Top30P':>8s} {'Top30R':>8s}")
    for n in names:
        s=sm[n]
        print(f"{n:24s} {s['roc_auc']:8.4f} {s['average_precision']:8.4f} {s['top20']['precision']:8.2%} {s['top20']['recall']:8.2%} {s['top30']['precision']:8.2%} {s['top30']['recall']:8.2%}")
    print("-"*128)
    d=role["T2_T4_over_T1"]
    print(f"T2-T4 contribution over T1: AUC {d['roc_auc_delta']:+.4f} | AP {d['average_precision_delta']:+.4f} | Top20P {d['top20_precision_delta']:+.2%} | Top30P {d['top30_precision_delta']:+.2%}")
    d=role["T5_over_T1_T4"]
    print(f"T5 contribution over T1-T4: AUC {d['roc_auc_delta']:+.4f} | AP {d['average_precision_delta']:+.4f} | Top20P {d['top20_precision_delta']:+.2%} | Top30P {d['top30_precision_delta']:+.2%}")
    print("-"*128)
    d=role["T2_T4_over_T1"]
    if d["roc_auc_delta"]>0 or d["average_precision_delta"]>0 or d["top20_precision_delta"]>0:
        print("KEY VERDICT: T2-T4 add measurable answer-error risk signal beyond T1.")
    else:
        print("KEY VERDICT: Current T2-T4 diagnostics do not yet add measurable risk signal beyond T1.")
    print("Feature inventory:",INV_FILE)
    print("OOF predictions:",OOF_FILE)
    print("Summary:",SUM_FILE)
    print("STATUS = MMQA_TRANSITION_TO_ANSWER_RISK_OOF_V1_COMPLETE")

if __name__=="__main__":
    main()

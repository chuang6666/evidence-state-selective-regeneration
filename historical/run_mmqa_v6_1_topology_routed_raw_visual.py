from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import mimetypes
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent

BASE_FILE = (
    ROOT / "outputs" / "mmqa_natural_end2end_base_v2_corrected"
    / "mmqa_dev1100_natural_end2end_base_v2_corrected.jsonl"
)
LOCALIZATION_FILE = (
    ROOT / "outputs" / "mmqa_natural_failure_localization_v1"
    / "mmqa_natural_failure_localization_v1.jsonl"
)
TOPOLOGY_FILE = (
    ROOT / "outputs" / "mmqa_transition_topology_tfidf_oof_v1"
    / "mmqa_transition_topology_tfidf_oof_predictions_v1.jsonl"
)
IMAGE_DIR = (
    ROOT / "data" / "public_benchmarks" / "multimodalqa" / "images_dev_full"
)
GOLD_FILE = ROOT / "data" / "raw" / "mmqa_official" / "MMQA_dev.jsonl.gz"

BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

EXPECTED_QIDS = 1096
EXPECTED_T1_LOCALIZED = 296
SEED = 42
BOOTSTRAPS = 10000

STAGE_A_CONF = 0.90
STAGE_B_CONF = 0.90
CONCRETE_ORIGINAL_CONF = 0.95

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty", 21: "twenty-one", 22: "twenty-two",
    23: "twenty-three", 24: "twenty-four", 25: "twenty-five",
    26: "twenty-six", 27: "twenty-seven", 28: "twenty-eight",
    29: "twenty-nine", 30: "thirty",
}
WORD_NUMBERS = {v.replace("-", " "): k for k, v in NUMBER_WORDS.items()}


def safe_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def norm(x: Any) -> str:
    s = unicodedata.normalize("NFKC", safe_text(x)).casefold()
    s = s.replace("-", " ")
    s = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", s)
    s = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff\s\.\-%']", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(x: Any) -> list[str]:
    return [t for t in norm(x).split() if t]


def token_f1(a: str, b: str) -> float:
    aa, bb = tokens(a), tokens(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    ac, bc = Counter(aa), Counter(bb)
    common = sum((ac & bc).values())
    if common == 0:
        return 0.0
    p = common / len(aa)
    r = common / len(bb)
    return 2 * p * r / (p + r)


def load_jsonl(path: Path, tolerant: bool = False) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                x = json.loads(s)
            except Exception as e:
                if tolerant:
                    continue
                raise RuntimeError(f"Malformed JSONL {path} line {line_no}: {e}") from e
            if isinstance(x, dict):
                rows.append(x)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_env_key() -> str:
    for env_file in (ROOT / ".env", ROOT.parent / ".env"):
        if env_file.is_file():
            for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
                s = raw.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
    for k in ("ZHIPUAI_API_KEY", "ZHIPU_API_KEY", "BIGMODEL_API_KEY"):
        v = os.getenv(k, "").strip()
        if v and "YOUR" not in v.upper() and "你的" not in v:
            return v
    raise RuntimeError("No valid Zhipu/BigModel API key found")


def image_index() -> dict[str, Path]:
    out: dict[str, Path] = {}
    dup = set()
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(IMAGE_DIR)
    for p in IMAGE_DIR.iterdir():
        if not p.is_file() or p.suffix.lower() not in ALLOWED_EXTS:
            continue
        stem = p.stem.lower()
        if stem in out:
            dup.add(stem)
        else:
            out[stem] = p
    if dup:
        raise RuntimeError(f"Duplicate image stems: {sorted(dup)[:10]}")
    return out


def to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def robust_json(text: str) -> dict[str, Any]:
    s = safe_text(text)
    if not s:
        raise ValueError("empty response")
    try:
        x = json.loads(s)
        if isinstance(x, dict):
            return x
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            x, _ = decoder.raw_decode(s[i:])
        except Exception:
            continue
        if isinstance(x, dict):
            return x
    raise ValueError("no JSON object found")


def bool_value(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    return safe_text(x).casefold() in {"true", "yes", "1", "pass"}


def conf_value(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v > 1.0:
        v /= 100.0
    return max(0.0, min(1.0, v))


def parse_int_like(x: Any) -> int | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and float(x).is_integer():
        return int(x)
    s = norm(x)
    if re.fullmatch(r"\d+", s):
        return int(s)
    return WORD_NUMBERS.get(s)


def is_count_question(question: str) -> bool:
    q = norm(question)
    patterns = (
        r"\bhow many\b",
        r"\bnumber of\b",
        r"\bwhat number\b",
    )
    return any(re.search(p, q) for p in patterns)


def infer_question_operation(question: str) -> str:
    q = norm(question)
    if is_count_question(question):
        return "COUNT"
    if re.search(r"\bwhat colo(u)?r\b|\bwhich colo(u)?r\b", q):
        return "COLOR"
    if re.search(r"\bwhat (word|text|letter|number)\b|\bwhat does .* say\b", q):
        return "TEXT"
    if re.search(r"\bwhat position\b|\bwhat pose\b|\bhow .* positioned\b", q):
        return "STATE"
    if re.search(r"\bwhat .* holding\b|\bholds what\b|\bwearing\b|\bwhat kind of\b", q):
        return "ATTRIBUTE"
    return "OTHER"


def normalize_operation(x: Any, question: str) -> str:
    deterministic = infer_question_operation(question)
    if deterministic != "OTHER":
        return deterministic
    s = safe_text(x).upper()
    if s in {"COUNT", "COLOR", "TEXT", "OBJECT", "STATE", "ATTRIBUTE", "OTHER"}:
        return s
    return "OTHER"


def parse_observation(obj: dict[str, Any], question: str) -> dict[str, Any]:
    instances_raw = obj.get("instances")
    instances = []
    if isinstance(instances_raw, list):
        for x in instances_raw:
            s = safe_text(x)
            if s:
                instances.append(s[:240])

    op = normalize_operation(obj.get("operation"), question)
    answer = safe_text(obj.get("answer")).strip().strip("\"'").strip()
    count = parse_int_like(obj.get("count"))
    answer_count = parse_int_like(answer)

    count_consistent = True
    if is_count_question(question):
        count_consistent = (
            count is not None
            and answer_count is not None
            and count == answer_count
            and len(instances) == count
        )

    return {
        "target_match": bool_value(obj.get("target_match")),
        "direct_answerable": bool_value(obj.get("direct_answerable")),
        "operation": op,
        "answer": answer,
        "count": count,
        "instances": instances,
        "visible_text": safe_text(obj.get("visible_text"))[:1200],
        "confidence": conf_value(obj.get("confidence")),
        "reason": safe_text(obj.get("reason"))[:1600],
        "count_consistent": bool(count_consistent),
    }


def observation_is_eligible(obs: dict[str, Any], question: str, min_conf: float) -> bool:
    if not (
        obs.get("target_match")
        and obs.get("direct_answerable")
        and safe_text(obs.get("answer"))
        and float(obs.get("confidence", 0.0)) >= min_conf
    ):
        return False
    if is_count_question(question) and not obs.get("count_consistent"):
        return False
    return True


def agreement_key(question: str, answer: str, count: int | None = None) -> str:
    if is_count_question(question):
        n = count if count is not None else parse_int_like(answer)
        return f"COUNT::{n}" if n is not None else "COUNT::INVALID"
    return "TEXT::" + norm(answer)


def canonical_output(question: str, answer: str, count: int | None = None) -> str:
    if is_count_question(question):
        n = count if count is not None else parse_int_like(answer)
        if n is not None and n in NUMBER_WORDS:
            return NUMBER_WORDS[n]
    return safe_text(answer).strip().strip("\"'").strip()


def is_insufficient(x: str) -> bool:
    n = norm(x).replace("_", " ")
    return n in {
        "insufficient evidence", "unknown", "cannot determine",
        "not enough information", "not enough evidence", "n a",
    }


def candidate_key(job: dict[str, Any]) -> str:
    return f"{job['qid']}::{job['evidence_id']}"


def latest_success_by_key(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    if not path.is_file():
        return out
    for r in load_jsonl(path, tolerant=True):
        if r.get("status") != "success":
            continue
        key = safe_text(r.get("cache_key"))
        if key:
            out[key] = r
    return out


def model_tag(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model)


def api_call(
    api_key: str,
    model: str,
    image_path: Path,
    prompt: str,
    attempts: int,
    request_interval: float,
    last_request: float,
) -> tuple[dict[str, Any] | None, float, str, str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": to_data_url(image_path)}},
                {"type": "text", "text": prompt},
            ],
        }],
        "thinking": {"type": "disabled"},
        "temperature": 0.0,
        "max_tokens": 1000,
    }

    last_error = ""
    raw = ""
    error_kind = "unknown"

    for attempt in range(1, attempts + 1):
        try:
            wait = request_interval - (time.monotonic() - last_request)
            if wait > 0:
                time.sleep(wait)

            resp = requests.post(
                BASE_URL,
                headers=headers,
                json=payload,
                timeout=(20, 120),
            )
            last_request = time.monotonic()

            if resp.status_code != 200:
                code = ""
                msg = ""
                try:
                    e = resp.json().get("error", {})
                    code = safe_text(e.get("code"))
                    msg = safe_text(e.get("message"))
                except Exception:
                    pass

                last_error = (
                    f"HTTP {resp.status_code} code={code!r} "
                    f"message={msg!r} body={resp.text[:1200]!r}"
                )

                if resp.status_code == 429 and code == "1305":
                    error_kind = "provider_overload_1305"
                    cooldown = [20, 45, 75][min(attempt - 1, 2)]
                elif resp.status_code == 429 and code == "1302":
                    error_kind = "account_rate_limit_1302"
                    cooldown = [30, 60, 90][min(attempt - 1, 2)]
                elif resp.status_code >= 500:
                    error_kind = "provider_5xx"
                    cooldown = [10, 20, 40][min(attempt - 1, 2)]
                else:
                    error_kind = "http_error"
                    cooldown = [5, 10, 20][min(attempt - 1, 2)]

                if attempt < attempts:
                    print(
                        f"    [{error_kind}] attempt={attempt}/{attempts}; "
                        f"cooldown={cooldown}s; defer-friendly retry...",
                        flush=True,
                    )
                    time.sleep(cooldown)
                    continue
                break

            data = resp.json()
            raw = safe_text(data["choices"][0]["message"]["content"])
            obj = robust_json(raw)
            return obj, last_request, "", raw, ""

        except Exception as e:
            last_error = repr(e)
            error_kind = "request_or_parse_transient"
            if attempt < attempts:
                cooldown = [5, 10, 20][min(attempt - 1, 2)]
                print(
                    f"    [transient] attempt={attempt}/{attempts}; "
                    f"cooldown={cooldown}s; retrying... error={last_error[:300]}",
                    flush=True,
                )
                time.sleep(cooldown)

    return None, last_request, last_error, raw, error_kind


def stage_a_prompt(question: str) -> str:
    op = infer_question_operation(question)
    count_rule = ""
    if op == "COUNT":
        count_rule = """
COUNT-SPECIFIC RULES:
- Enumerate EVERY counted visible instance in `instances`.
- Each instance must have a short distinguishing location/description.
- `count` MUST equal len(instances).
- `answer` MUST be the same number as `count`.
- If you cannot enumerate consistently, set direct_answerable=false.
""".strip()

    return f"""
You are performing BLIND RAW-IMAGE TARGET SELECTION AND VISUAL EXTRACTION.

QUESTION:
{question}

You receive ONE raw candidate image from a frozen retrieval result.
You are NOT given the image filename, title, evidence ID, rank, caption, prior repair,
original answer, candidate answer, gold answer, or gold evidence.

Your job:
1. Decide whether THIS raw image itself matches the visual entity/asset referred to by the question.
2. Decide whether the FINAL requested answer slot can be answered directly from THIS image alone.
3. If and only if both are true, extract the shortest complete answer from visible pixels.
4. Do not use outside knowledge.
5. If the question requires an unseen relation, comparison, chronology, table/text lookup,
   identity not established by the pixels, or another reasoning hop, set direct_answerable=false.
6. Be conservative about target identity. A visually plausible but unidentifiable image is not enough.
7. Preserve all visually necessary attributes; do not drop a color/material/state modifier that is visible.
{count_rule}

Return EXACTLY one JSON object:
{{
  "target_match": true,
  "direct_answerable": true,
  "operation": "{op if op != 'OTHER' else 'COUNT|COLOR|TEXT|OBJECT|STATE|ATTRIBUTE|OTHER'}",
  "answer": "short answer or empty",
  "count": null,
  "instances": [],
  "visible_text": "only question-relevant text actually visible, or empty",
  "confidence": 0,
  "reason": "brief pixel-grounded reason"
}}

confidence is 0-100.
For non-count questions, use count=null and instances=[].
""".strip()


def stage_b_prompt(question: str) -> str:
    op = infer_question_operation(question)
    count_rule = ""
    if op == "COUNT":
        count_rule = """
COUNT-SPECIFIC RULES:
- Independently enumerate EVERY counted visible instance in `instances`.
- Give each a short distinguishing location/description.
- `count` MUST equal len(instances).
- `answer` MUST equal `count`.
- If the enumeration is not internally consistent, set direct_answerable=false.
""".strip()

    return f"""
You are the SECOND INDEPENDENT RAW-IMAGE RECONSTRUCTION CHECK.

QUESTION:
{question}

Observe the raw image from scratch.
You are NOT shown any first-pass answer/rationale, original draft answer, image title,
evidence ID, filename, rank, caption, gold answer, or gold evidence.

Independently determine:
1. Whether this exact image matches the visual target referred to by the question.
2. Whether the FINAL requested answer is directly answerable from this image alone.
3. The shortest complete visual answer if both are true.
4. Use only visible pixels; no outside knowledge.
5. If any extra nonvisual hop is required, set direct_answerable=false.
6. Preserve necessary visible modifiers rather than shortening away answer-critical information.
{count_rule}

Return EXACTLY one JSON object:
{{
  "target_match": true,
  "direct_answerable": true,
  "operation": "{op if op != 'OTHER' else 'COUNT|COLOR|TEXT|OBJECT|STATE|ATTRIBUTE|OTHER'}",
  "answer": "short answer or empty",
  "count": null,
  "instances": [],
  "visible_text": "only question-relevant text actually visible, or empty",
  "confidence": 0,
  "reason": "brief independent pixel-grounded reason"
}}

confidence is 0-100.
For non-count questions, use count=null and instances=[].
""".strip()


def rank_value(e: dict[str, Any], fallback: int) -> float:
    x = e.get("rank", fallback)
    try:
        return float(x)
    except Exception:
        return float(fallback)


def build_stage_a_jobs(
    base_by: dict[str, dict[str, Any]],
    topology_direct_qids: set[str],
    t1_localized: set[str],
    img_by_stem: dict[str, Path],
    topk_images: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    jobs = []
    no_image = []
    stats = Counter()

    target_qids = sorted(topology_direct_qids & t1_localized)
    for qid in target_qids:
        base = base_by[qid]
        imgs = []
        seen = set()
        for idx, e in enumerate(base.get("retrieved_evidence", []) or [], 1):
            if not isinstance(e, dict):
                continue
            modality = safe_text(e.get("modality")).casefold()
            if "image" not in modality and "visual" not in modality:
                continue
            eid = safe_text(e.get("evidence_id") or e.get("id"))
            if not eid or eid.casefold() in seen:
                continue
            seen.add(eid.casefold())
            p = img_by_stem.get(eid.casefold())
            if p is None:
                stats["retrieved_image_missing_raw_file"] += 1
                continue
            imgs.append((rank_value(e, idx), idx, eid, p))

        imgs.sort(key=lambda x: (x[0], x[1]))
        imgs = imgs[:topk_images]

        if not imgs:
            no_image.append({
                "qid": qid,
                "question": safe_text(base.get("question")),
                "reason": "no_retrieved_raw_image_available",
            })
            continue

        for local_pos, (rank, _, eid, p) in enumerate(imgs, 1):
            jobs.append({
                "cache_key": f"{qid}::{eid}",
                "qid": qid,
                "question": safe_text(base.get("question")),
                "original_answer": safe_text(base.get("draft_answer")),
                "evidence_id": eid,
                "retrieval_rank": rank,
                "candidate_image_position": local_pos,
                "image_path": str(p),
            })

    stats["topology_direct_qids"] = len(topology_direct_qids)
    stats["localized_t1_qids"] = len(t1_localized)
    stats["intersection_qids"] = len(target_qids)
    stats["intersection_without_raw_image"] = len(no_image)
    stats["stage_a_image_jobs"] = len(jobs)
    return jobs, no_image, dict(stats)


def run_cached_jobs(
    jobs: list[dict[str, Any]],
    cache_path: Path,
    api_key: str,
    model: str,
    prompt_fn,
    parser_fn,
    stage_name: str,
    attempts_per_item: int,
    recovery_rounds: int,
    request_interval: float,
    round_cooldown: int,
) -> dict[str, dict[str, Any]]:
    cache = latest_success_by_key(cache_path)
    last_request = 0.0

    for round_idx in range(1, recovery_rounds + 1):
        pending = [j for j in jobs if j["cache_key"] not in cache]
        if not pending:
            break

        print(
            f"{stage_name} recovery round {round_idx}/{recovery_rounds}: "
            f"pending={len(pending)}",
            flush=True,
        )

        round_success = 0
        for i, job in enumerate(pending, 1):
            obj, last_request, err, raw, error_kind = api_call(
                api_key=api_key,
                model=model,
                image_path=Path(job["image_path"]),
                prompt=prompt_fn(job["question"]),
                attempts=attempts_per_item,
                request_interval=request_interval,
                last_request=last_request,
            )
            if obj is None:
                append_jsonl(cache_path, {
                    "cache_key": job["cache_key"],
                    "qid": job["qid"],
                    "evidence_id": job["evidence_id"],
                    "status": "failed",
                    "stage": stage_name,
                    "model": model,
                    "error_kind": error_kind,
                    "error": err,
                })
                print(
                    f"  {stage_name} [{i}/{len(pending)}] "
                    f"{job['cache_key']} -> PENDING ({error_kind})",
                    flush=True,
                )
                continue

            try:
                parsed = parser_fn(obj, job["question"])
            except Exception as e:
                append_jsonl(cache_path, {
                    "cache_key": job["cache_key"],
                    "qid": job["qid"],
                    "evidence_id": job["evidence_id"],
                    "status": "failed",
                    "stage": stage_name,
                    "model": model,
                    "error_kind": "schema_validation",
                    "error": repr(e),
                    "raw_response": raw,
                })
                print(
                    f"  {stage_name} [{i}/{len(pending)}] "
                    f"{job['cache_key']} -> PENDING (schema_validation)",
                    flush=True,
                )
                continue

            row = {
                "cache_key": job["cache_key"],
                "qid": job["qid"],
                "evidence_id": job["evidence_id"],
                "status": "success",
                "stage": stage_name,
                "model": model,
                "observation": parsed,
                "raw_response": raw,
            }
            append_jsonl(cache_path, row)
            cache[job["cache_key"]] = row
            round_success += 1

            print(
                f"  {stage_name} [{i}/{len(pending)}] {job['cache_key']} "
                f"target={parsed['target_match']} "
                f"direct={parsed['direct_answerable']} "
                f"conf={parsed['confidence']:.2f} "
                f"count_ok={parsed['count_consistent']} "
                f"answer={parsed['answer']!r}",
                flush=True,
            )

        remaining = sum(j["cache_key"] not in cache for j in jobs)
        print(
            f"{stage_name} round {round_idx}: "
            f"success={round_success}, remaining={remaining}",
            flush=True,
        )
        if remaining and round_idx < recovery_rounds:
            print(
                f"{stage_name}: deferring failed items; "
                f"cooldown={round_cooldown}s before next recovery round.",
                flush=True,
            )
            time.sleep(round_cooldown)

    return latest_success_by_key(cache_path)


def select_stage_a(
    jobs: list[dict[str, Any]],
    stage_a_cache: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for j in jobs:
        by_qid.setdefault(j["qid"], []).append(j)

    selections = []
    stage_b_jobs = []

    for qid in sorted(by_qid):
        options = []
        for j in sorted(
            by_qid[qid],
            key=lambda x: (x["retrieval_rank"], x["candidate_image_position"]),
        ):
            cached = stage_a_cache.get(j["cache_key"])
            if not cached:
                continue
            obs = cached["observation"]
            eligible = observation_is_eligible(obs, j["question"], STAGE_A_CONF)
            options.append({
                "cache_key": j["cache_key"],
                "evidence_id": j["evidence_id"],
                "retrieval_rank": j["retrieval_rank"],
                "candidate_image_position": j["candidate_image_position"],
                "image_path": j["image_path"],
                "eligible": bool(eligible),
                "observation": obs,
                "agreement_key": (
                    agreement_key(j["question"], obs["answer"], obs.get("count"))
                    if eligible else None
                ),
            })

        eligible_opts = [x for x in options if x["eligible"]]
        reason = ""
        selected = None

        if not eligible_opts:
            reason = "no_stage_a_eligible_image"
        else:
            keys = {x["agreement_key"] for x in eligible_opts}
            if len(keys) != 1:
                reason = "conflicting_high_conf_raw_images"
            else:
                # Same-answer consensus across all eligible target images.
                selected = sorted(
                    eligible_opts,
                    key=lambda x: (
                        -float(x["observation"]["confidence"]),
                        float(x["retrieval_rank"]),
                        int(x["candidate_image_position"]),
                    ),
                )[0]
                reason = "stage_a_target_selected_with_consensus"

        selection = {
            "qid": qid,
            "question": by_qid[qid][0]["question"],
            "original_answer": by_qid[qid][0]["original_answer"],
            "num_raw_candidates": len(options),
            "num_stage_a_eligible": len(eligible_opts),
            "selection_reason": reason,
            "selected": selected,
            "options": options,
        }
        selections.append(selection)

        if selected is not None:
            stage_b_jobs.append({
                "cache_key": selected["cache_key"],
                "qid": qid,
                "question": selection["question"],
                "original_answer": selection["original_answer"],
                "evidence_id": selected["evidence_id"],
                "retrieval_rank": selected["retrieval_rank"],
                "candidate_image_position": selected["candidate_image_position"],
                "image_path": selected["image_path"],
            })

    return selections, stage_b_jobs


def projection_policy(
    question: str,
    original: str,
    a: dict[str, Any],
    b: dict[str, Any],
) -> tuple[str, bool, str]:
    """
    Minimal answer-preserving projection.
    A and B are independent raw-image observations of the same selected image.
    """
    if not observation_is_eligible(b, question, STAGE_B_CONF):
        return original, False, "preserve_stage_b_not_eligible"

    key_a = agreement_key(question, a["answer"], a.get("count"))
    key_b = agreement_key(question, b["answer"], b.get("count"))
    if key_a != key_b:
        return original, False, "preserve_independent_answer_disagreement"

    candidate = canonical_output(question, b["answer"], b.get("count"))
    if not candidate:
        return original, False, "preserve_empty_candidate"

    # Count questions: semantic numeric agreement is explicit and independently checked.
    if is_count_question(question):
        cand_n = parse_int_like(candidate)
        orig_n = parse_int_like(original)

        if cand_n is None:
            return original, False, "preserve_invalid_count_candidate"

        # General deterministic canonicalization: digit -> English number word.
        if orig_n is not None and orig_n == cand_n:
            if norm(original) != norm(candidate):
                return candidate, True, "update_count_format_canonicalization"
            return original, False, "preserve_same_count_semantics"

        if is_insufficient(original):
            return candidate, True, "update_insufficient_to_double_verified_count"

        # Concrete answer changes require stronger confidence.
        if (
            float(a.get("confidence", 0.0)) >= CONCRETE_ORIGINAL_CONF
            and float(b.get("confidence", 0.0)) >= CONCRETE_ORIGINAL_CONF
        ):
            return candidate, norm(candidate) != norm(original), "update_double_verified_count_correction"

        return original, False, "preserve_concrete_count_low_conf"

    # Non-count: never rewrite an already equivalent answer.
    if norm(candidate) == norm(original):
        return original, False, "preserve_semantically_same"

    # If the old answer is insufficient, a double-independent raw-image answer may fill it.
    if is_insufficient(original):
        return candidate, True, "update_insufficient_to_double_verified_visual"

    # Prevent partial/near-synonymous rewrites such as:
    # "red hoodie" -> "hoodie" or "hands clasped together" -> paraphrase.
    overlap = token_f1(original, candidate)
    if overlap >= 0.50:
        return original, False, "preserve_overlap_or_modifier_loss_risk"

    # Keep answer projection atomic and conservative.
    if len(candidate) > 100 or len(tokens(candidate)) > 12:
        return original, False, "preserve_non_atomic_candidate"

    # A concrete original may be overwritten only by two high-confidence
    # independent observations that agree exactly after normalization.
    if (
        float(a.get("confidence", 0.0)) >= CONCRETE_ORIGINAL_CONF
        and float(b.get("confidence", 0.0)) >= CONCRETE_ORIGINAL_CONF
    ):
        return candidate, True, "update_concrete_double_verified_visual_correction"

    return original, False, "preserve_concrete_original_high_bar"


def extract_answers(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, (int, float, bool)):
        return [str(x)]
    out = []
    if isinstance(x, list):
        for v in x:
            out.extend(extract_answers(v))
    elif isinstance(x, dict):
        for k in ("answers", "answer", "gold_answers", "gold_answer", "text", "value"):
            if k in x:
                out.extend(extract_answers(x[k]))
    return out


def load_gold() -> dict[str, list[str]]:
    out = {}
    with gzip.open(GOLD_FILE, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not isinstance(r, dict):
                continue
            qid = safe_text(r.get("qid") or r.get("question_id") or r.get("id"))
            vals = []
            for k in ("answers", "gold_answers", "gold_answer", "answer"):
                if k in r:
                    vals.extend(extract_answers(r[k]))
            vals = [v for v in vals if safe_text(v)]
            if qid and vals:
                out[qid] = vals
    return out


def score_one(pred: str, golds: list[str]) -> tuple[int, float]:
    em = int(any(norm(pred) == norm(g) for g in golds))
    f1 = max(token_f1(pred, g) for g in golds)
    return em, f1


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def bootstrap_delta(base: np.ndarray, new: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n = len(base)
    ds = np.empty(BOOTSTRAPS, dtype=float)
    for i in range(BOOTSTRAPS):
        idx = rng.integers(0, n, size=n)
        ds[i] = new[idx].mean() - base[idx].mean()
    lo, hi = np.quantile(ds, [0.025, 0.975])
    return {
        "delta": float(new.mean() - base.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "samples": BOOTSTRAPS,
    }


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    bem = np.asarray([r["base_em"] for r in details], dtype=float)
    nem = np.asarray([r["v6_em"] for r in details], dtype=float)
    bf = np.asarray([r["base_f1"] for r in details], dtype=float)
    nf = np.asarray([r["v6_f1"] for r in details], dtype=float)

    w2r = sum(r["base_em"] == 0 and r["v6_em"] == 1 for r in details)
    r2w = sum(r["base_em"] == 1 and r["v6_em"] == 0 for r in details)

    return {
        "n": len(details),
        "base_em": float(bem.mean()),
        "v6_em": float(nem.mean()),
        "delta_em": float(nem.mean() - bem.mean()),
        "base_f1": float(bf.mean()),
        "v6_f1": float(nf.mean()),
        "delta_f1": float(nf.mean() - bf.mean()),
        "wrong_to_right": int(w2r),
        "right_to_wrong": int(r2w),
        "net_em_corrections": int(w2r - r2w),
        "mcnemar_exact_p": float(exact_mcnemar_p(w2r, r2w)),
        "bootstrap_em": bootstrap_delta(bem, nem),
        "bootstrap_f1": bootstrap_delta(bf, nf),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glm-4.6v-flash")
    ap.add_argument("--topk-images", type=int, default=3)
    ap.add_argument("--attempts-per-item", type=int, default=3)
    ap.add_argument("--recovery-rounds", type=int, default=3)
    ap.add_argument("--request-interval", type=float, default=4.0)
    ap.add_argument("--round-cooldown", type=int, default=60)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    if args.topk_images < 1:
        raise ValueError("--topk-images must be >= 1")
    if args.attempts_per_item < 1:
        raise ValueError("--attempts-per-item must be >= 1")
    if args.recovery_rounds < 1:
        raise ValueError("--recovery-rounds must be >= 1")

    required_paths = [
        BASE_FILE, LOCALIZATION_FILE, TOPOLOGY_FILE, IMAGE_DIR,
    ]
    for p in required_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    tag = model_tag(args.model)
    OUT = ROOT / "outputs" / f"mmqa_v6_1_topology_routed_raw_visual_{tag}"
    STAGE_A_CACHE = OUT / "stage_a_raw_image_candidates_cache.jsonl"
    STAGE_A_SELECTION = OUT / "stage_a_target_selection.jsonl"
    STAGE_B_CACHE = OUT / "stage_b_independent_reconstruction_cache.jsonl"
    NO_IMAGE_FILE = OUT / "preflight_no_raw_image_qids.jsonl"
    PREFLIGHT_FILE = OUT / "preflight_summary.json"
    PREDICTIONS = OUT / "mmqa_v6_1_predictions.jsonl"
    DETAILS = OUT / "mmqa_v6_1_eval_details.jsonl"
    SUMMARY = OUT / "mmqa_v6_1_summary.json"
    OUT.mkdir(parents=True, exist_ok=True)

    base_rows = load_jsonl(BASE_FILE)
    if len(base_rows) != EXPECTED_QIDS:
        raise RuntimeError(
            f"Expected {EXPECTED_QIDS} natural base rows, found {len(base_rows)}"
        )
    base_by = {safe_text(r.get("qid")): r for r in base_rows if safe_text(r.get("qid"))}
    if len(base_by) != EXPECTED_QIDS:
        raise RuntimeError("Base QIDs are missing/duplicated")

    loc_rows = load_jsonl(LOCALIZATION_FILE)
    loc_by = {safe_text(r.get("qid")): r for r in loc_rows if safe_text(r.get("qid"))}
    if set(loc_by) != set(base_by):
        raise RuntimeError("Localization/base QID sets differ")

    t1_localized = {
        qid
        for qid, r in loc_by.items()
        if safe_text(r.get("decision")) != "CLEAN_OR_UNCERTAIN"
        and safe_text(r.get("localized_transition")).upper() == "T1"
    }
    if len(t1_localized) != EXPECTED_T1_LOCALIZED:
        raise RuntimeError(
            f"Frozen T1 localized count changed: "
            f"{len(t1_localized)} != {EXPECTED_T1_LOCALIZED}"
        )

    topology_rows = load_jsonl(TOPOLOGY_FILE)
    topology_by = {
        safe_text(r.get("qid")): r
        for r in topology_rows
        if safe_text(r.get("qid"))
    }
    if set(topology_by) != set(base_by):
        raise RuntimeError("Topology/base QID sets differ")

    topology_direct_qids = set()
    for qid, r in topology_by.items():
        pred = r.get("predicted_required_transitions_schema_constrained")
        if not isinstance(pred, list):
            continue
        s = {safe_text(x) for x in pred if safe_text(x)}
        if s == {"T1", "T5"}:
            topology_direct_qids.add(qid)

    if not topology_direct_qids:
        raise RuntimeError("Question-only OOF topology router produced zero T1+T5 qids")

    img_by_stem = image_index()

    stage_a_jobs, no_image_rows, preflight_stats = build_stage_a_jobs(
        base_by=base_by,
        topology_direct_qids=topology_direct_qids,
        t1_localized=t1_localized,
        img_by_stem=img_by_stem,
        topk_images=args.topk_images,
    )
    write_jsonl(NO_IMAGE_FILE, no_image_rows)

    if preflight_stats["intersection_qids"] == 0:
        raise RuntimeError("No qids survived T1-localization x OOF T1+T5 topology routing")
    if not (20 <= preflight_stats["intersection_qids"] <= 100):
        raise RuntimeError(
            "Preflight routing cardinality is outside the expected safety range "
            f"[20,100]: got {preflight_stats['intersection_qids']}. "
            "Stopping BEFORE any API calls."
        )
    if not stage_a_jobs:
        raise RuntimeError("No raw-image jobs available after preflight")

    preflight = {
        "status": "MMQA_V6_1_RAW_VISUAL_PREFLIGHT_PASS",
        "model": args.model,
        "routing": "frozen qid-held-out question-only OOF topology == {T1,T5}",
        "topk_images_per_qid": args.topk_images,
        "counts": preflight_stats,
        "gold_loaded": False,
        "final_test_used": False,
        "evaluation_scorer": "exact V5 canonical norm/token_f1 protocol",
        "inference_inputs_exposed_to_vlm": [
            "question_text",
            "raw_image_pixels",
        ],
        "explicitly_not_exposed_to_vlm": [
            "image_title",
            "evidence_title",
            "evidence_id",
            "retrieval_rank",
            "caption",
            "guided_caption",
            "old_local_repair",
            "old_candidate_answer",
            "original_draft_answer",
            "gold_answer",
            "gold_evidence",
            "official_question_type",
        ],
        "stage_a_policy": (
            "Only frozen natural T1-localized qids whose frozen qid-held-out "
            "question-only OOF topology prediction is exactly {T1,T5} are eligible. "
            "For each qid, inspect up to top-k retrieved raw images, metadata-blind."
        ),
        "stage_b_policy": (
            "Second independent reconstruction sees only question + selected raw image; "
            "it does not see Stage-A answer/rationale or original answer."
        ),
    }
    PREFLIGHT_FILE.write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 132)
    print("MMQA — V6.1 TOPOLOGY-ROUTED HIGH-PRECISION RAW-VISUAL LOCAL REPAIR")
    print("=" * 132)
    print("Model:", args.model)
    print("Base qids:", len(base_by))
    print("Frozen localized T1 qids:", len(t1_localized))
    print("OOF topology-direct (T1+T5 only) qids:", len(topology_direct_qids))
    print("Intersection (T1 failure x OOF T1+T5 topology):", preflight_stats["intersection_qids"])
    print("Intersection with no raw image:", preflight_stats["intersection_without_raw_image"])
    print("Stage-A raw-image jobs:", len(stage_a_jobs))
    print("Top-k retrieved raw images per qid:", args.topk_images)
    print("VLM sees evidence title/ID/rank/caption: NO")
    print("Gold loaded before prediction freeze: NO")
    print("Final-test used: NO")
    print("Pending-queue policy: bounded per-item retry + recovery rounds")
    print("Preflight:", PREFLIGHT_FILE)
    print("-" * 132)

    if args.preflight_only:
        print("API/model calls = 0")
        print("STATUS = MMQA_V6_1_RAW_VISUAL_PREFLIGHT_ONLY_COMPLETE")
        return

    api_key = load_env_key()

    # Stage A: metadata-blind target selection + extraction over top-k retrieved raw images.
    stage_a_cache = run_cached_jobs(
        jobs=stage_a_jobs,
        cache_path=STAGE_A_CACHE,
        api_key=api_key,
        model=args.model,
        prompt_fn=stage_a_prompt,
        parser_fn=parse_observation,
        stage_name="STAGE_A",
        attempts_per_item=args.attempts_per_item,
        recovery_rounds=args.recovery_rounds,
        request_interval=args.request_interval,
        round_cooldown=args.round_cooldown,
    )

    missing_a = [
        j for j in stage_a_jobs
        if j["cache_key"] not in stage_a_cache
    ]
    if missing_a:
        print("-" * 132)
        print("Stage-A unresolved after bounded recovery:", len(missing_a))
        print("Successful Stage-A items are cached. Rerun the SAME command later.")
        print("STATUS = MMQA_V6_1_RAW_VISUAL_STAGE_A_PARTIAL")
        return

    selections, stage_b_jobs = select_stage_a(stage_a_jobs, stage_a_cache)
    write_jsonl(STAGE_A_SELECTION, selections)

    select_counts = Counter(s["selection_reason"] for s in selections)
    print("-" * 132)
    print("Stage-A selection qids:", len(selections))
    print("Stage-A selection reasons:", dict(select_counts))
    print("Stage-B selected qids:", len(stage_b_jobs))

    # Stage B: independent reconstruction on only the chosen raw image.
    stage_b_cache = run_cached_jobs(
        jobs=stage_b_jobs,
        cache_path=STAGE_B_CACHE,
        api_key=api_key,
        model=args.model,
        prompt_fn=stage_b_prompt,
        parser_fn=parse_observation,
        stage_name="STAGE_B",
        attempts_per_item=args.attempts_per_item,
        recovery_rounds=args.recovery_rounds,
        request_interval=args.request_interval,
        round_cooldown=args.round_cooldown,
    )

    missing_b = [
        j for j in stage_b_jobs
        if j["cache_key"] not in stage_b_cache
    ]
    if missing_b:
        print("-" * 132)
        print("Stage-B unresolved after bounded recovery:", len(missing_b))
        print("Successful Stage-B items are cached. Rerun the SAME command later.")
        print("STATUS = MMQA_V6_1_RAW_VISUAL_STAGE_B_PARTIAL")
        return

    # Freeze predictions BEFORE opening gold.
    selection_by = {s["qid"]: s for s in selections}
    stage_b_job_by = {j["qid"]: j for j in stage_b_jobs}

    prediction_rows = []
    policy_counts = Counter()

    for qid in sorted(base_by):
        base = base_by[qid]
        original = safe_text(base.get("draft_answer"))
        final = original
        changed = False
        policy = "preserve_non_candidate"
        selected_eid = None

        sel = selection_by.get(qid)
        if sel is not None:
            if sel.get("selected") is None:
                policy = "preserve_stage_a_no_safe_target"
            else:
                selected = sel["selected"]
                selected_eid = selected["evidence_id"]
                key = selected["cache_key"]
                a = stage_a_cache[key]["observation"]
                b = stage_b_cache[key]["observation"]
                final, changed, policy = projection_policy(
                    question=safe_text(base.get("question")),
                    original=original,
                    a=a,
                    b=b,
                )

        policy_counts[policy] += 1
        prediction_rows.append({
            "qid": qid,
            "question": base.get("question"),
            "original_answer": original,
            "final_answer": final,
            "answer_changed": int(changed),
            "policy_reason": policy,
            "selected_raw_image_evidence_id": selected_eid,
            "model": args.model if sel is not None else None,
            "inference_gold_used": False,
            "inference_official_type_used": False,
        })

    write_jsonl(PREDICTIONS, prediction_rows)

    print("-" * 132)
    print("PREDICTIONS FROZEN.")
    print("Gold has NOT been loaded by inference.")
    print("Changed answers:", sum(r["answer_changed"] for r in prediction_rows))
    print("Policy reasons:", dict(policy_counts))

    # Evaluation only now.
    gold = load_gold()
    if sum(qid in gold for qid in base_by) != EXPECTED_QIDS:
        raise RuntimeError("Gold coverage incomplete after prediction freeze")

    pred_by = {r["qid"]: r for r in prediction_rows}
    details = []

    for qid in sorted(base_by):
        original = safe_text(base_by[qid].get("draft_answer"))
        final = safe_text(pred_by[qid].get("final_answer"))

        bem, bf = score_one(original, gold[qid])
        vem, vf = score_one(final, gold[qid])

        sel = selection_by.get(qid)
        selected = sel.get("selected") if sel else None
        key = selected.get("cache_key") if selected else None

        details.append({
            **pred_by[qid],
            "base_em": bem,
            "base_f1": bf,
            "v6_em": vem,
            "v6_f1": vf,
            "stage_a_selection_reason": sel.get("selection_reason") if sel else None,
            "stage_a_selected_observation": (
                stage_a_cache[key]["observation"] if key else None
            ),
            "stage_b_independent_observation": (
                stage_b_cache[key]["observation"] if key else None
            ),
            "topology_prediction": (
                topology_by.get(qid, {}).get(
                    "predicted_required_transitions_schema_constrained"
                )
            ),
        })

    write_jsonl(DETAILS, details)
    overall = summarize(details)

    changed_rows = [r for r in details if r["answer_changed"] == 1]
    outcomes = Counter()
    for r in changed_rows:
        if r["base_em"] == 0 and r["v6_em"] == 1:
            outcomes["W2R"] += 1
        elif r["base_em"] == 1 and r["v6_em"] == 0:
            outcomes["R2W"] += 1
        elif r["base_em"] == 1 and r["v6_em"] == 1:
            outcomes["R2R"] += 1
        else:
            outcomes["W2W"] += 1

    dev_positive = (
        overall["delta_em"] > 0
        and overall["delta_f1"] > 0
        and overall["net_em_corrections"] > 0
    )
    dev_strong = (
        dev_positive
        and overall["mcnemar_exact_p"] < 0.05
        and overall["bootstrap_em"]["ci95_low"] > 0
        and overall["bootstrap_f1"]["ci95_low"] > 0
    )

    summary = {
        "status": "MMQA_V6_1_TOPOLOGY_ROUTED_RAW_VISUAL_DEV_COMPLETE",
        "development_set_only": True,
        "confirmatory_significance_claim_allowed": False,
        "raw_visual_model": args.model,
        "routing": {
            "source": str(TOPOLOGY_FILE),
            "rule": "qid-held-out question-only OOF topology exactly {T1,T5}",
            "development_selected": False,
        },
        "candidate_policy": {
            "requires_frozen_T1_localization": True,
            "requires_question_only_oof_topology_T1_T5": True,
            "topk_retrieved_raw_images": args.topk_images,
            "image_metadata_exposed_to_vlm": False,
            "stage_a_independent_from_original_answer": True,
            "stage_b_independent_from_stage_a_answer": True,
            "conflicting_stage_a_answers_default_keep": True,
            "count_requires_enumeration_consistency": True,
            "default_action": "KEEP_ORIGINAL",
        },
        "counts": {
            **preflight_stats,
            "stage_a_selected_qids": len(stage_b_jobs),
            "changed_answers": len(changed_rows),
            "changed_outcomes": dict(outcomes),
        },
        "overall": overall,
        "dev_positive_signal": bool(dev_positive),
        "dev_strong_signal": bool(dev_strong),
        "gold_loaded_only_after_predictions_frozen": True,
        "evaluation_scorer": "exact V5 canonical norm/token_f1 protocol",
        "final_test_used": False,
        "outputs": {
            "predictions": str(PREDICTIONS),
            "eval_details": str(DETAILS),
            "stage_a_selection": str(STAGE_A_SELECTION),
        },
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 132)
    print("FINAL V6.1 TOPOLOGY-ROUTED RAW-VISUAL DEV RESULT")
    print("=" * 132)
    print(f"No Repair: EM={overall['base_em']:.6f} F1={overall['base_f1']:.6f}")
    print(f"V6.1:      EM={overall['v6_em']:.6f} F1={overall['v6_f1']:.6f}")
    print(f"Delta:     EM={overall['delta_em']:+.6f} F1={overall['delta_f1']:+.6f}")
    print(
        f"Changed={len(changed_rows)}/{EXPECTED_QIDS} "
        f"W2R={overall['wrong_to_right']} "
        f"R2W={overall['right_to_wrong']} "
        f"Net={overall['net_em_corrections']}"
    )
    print("Changed outcomes:", dict(outcomes))
    print(f"McNemar p={overall['mcnemar_exact_p']:.8g}")
    print(
        "Delta EM 95% CI: "
        f"[{overall['bootstrap_em']['ci95_low']:+.6f}, "
        f"{overall['bootstrap_em']['ci95_high']:+.6f}]"
    )
    print(
        "Delta F1 95% CI: "
        f"[{overall['bootstrap_f1']['ci95_low']:+.6f}, "
        f"{overall['bootstrap_f1']['ci95_high']:+.6f}]"
    )
    print("DEV_POSITIVE_SIGNAL =", bool(dev_positive))
    print("DEV_STRONG_SIGNAL =", bool(dev_strong))
    print("Predictions:", PREDICTIONS)
    print("Summary:", SUMMARY)
    print("STATUS = MMQA_V6_1_TOPOLOGY_ROUTED_RAW_VISUAL_DEV_COMPLETE")


if __name__ == "__main__":
    main()

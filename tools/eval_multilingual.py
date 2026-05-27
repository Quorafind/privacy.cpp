#!/usr/bin/env python3
"""Evaluate multilingual PII detection: model-only vs model+regex.

Generates a multilingual test set, runs the privacy binary in both modes
(--no-regex for model-only, default for model+regex), and produces
per-language and per-label comparison tables.

Usage:
    python tools/eval_multilingual.py --binary build/Release/privacy.exe --model model [--count 500]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Import the generator
sys.path.insert(0, "tools")
from generate_test_set import generate_dataset, LANG_DISPLAY


def iou(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def run_predict(binary, model_dir, text, no_regex=False):
    cmd = [os.path.normpath(binary), "--model", os.path.normpath(model_dir)]
    if no_regex:
        cmd.append("--no-regex")
    cmd.extend(["--", text])
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    stdout = result.stdout.decode("utf-8", errors="replace")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def match_spans(truth, preds, iou_threshold=0.5):
    matched_truth = set()
    matched_pred = set()
    for i, t in enumerate(truth):
        for j, p in enumerate(preds):
            if j in matched_pred:
                continue
            if t["label"] != p["label"]:
                continue
            if iou(t["start"], t["end"], p["start"], p["end"]) >= iou_threshold:
                matched_truth.add(i)
                matched_pred.add(j)
                break
    tp = len(matched_truth)
    fn = len(truth) - tp
    fp = len(preds) - len(matched_pred)
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def evaluate_samples(binary, model_dir, samples, no_regex=False):
    per_label = {}
    per_lang = {}
    overall_tp, overall_fp, overall_fn = 0, 0, 0
    errors = 0
    total_ms = 0.0

    for idx, sample in enumerate(samples):
        text = sample["text"]
        truth = sample["spans"]
        lang = LANG_DISPLAY.get(sample.get("lang", "en"), sample.get("lang", "English"))

        t0 = time.perf_counter()
        preds = run_predict(binary, model_dir, text, no_regex=no_regex)
        t1 = time.perf_counter()
        total_ms += (t1 - t0) * 1000.0

        if preds is None:
            errors += 1
            overall_fn += len(truth)
            for t_span in truth:
                lbl = t_span["label"]
                per_label.setdefault(lbl, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
                per_lang.setdefault(lang, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
            continue

        truth_matched = set()
        pred_matched = set()
        for i, t_span in enumerate(truth):
            for j, p_span in enumerate(preds):
                if j in pred_matched:
                    continue
                if t_span["label"] != p_span["label"]:
                    continue
                if iou(t_span["start"], t_span["end"], p_span["start"], p_span["end"]) >= 0.5:
                    truth_matched.add(i)
                    pred_matched.add(j)
                    break

        tp = len(truth_matched)
        fp = len(preds) - len(pred_matched)
        fn = len(truth) - tp
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        for i, t_span in enumerate(truth):
            lbl = t_span["label"]
            per_label.setdefault(lbl, {"tp": 0, "fp": 0, "fn": 0})
            per_lang.setdefault(lang, {"tp": 0, "fp": 0, "fn": 0})
            if i in truth_matched:
                per_label[lbl]["tp"] += 1
                per_lang[lang]["tp"] += 1
            else:
                per_label[lbl]["fn"] += 1
                per_lang[lang]["fn"] += 1
        for j, p_span in enumerate(preds):
            lbl = p_span["label"]
            per_label.setdefault(lbl, {"tp": 0, "fp": 0, "fn": 0})
            if j not in pred_matched:
                per_label[lbl]["fp"] += 1
                per_lang.setdefault(lang, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1

        if (idx + 1) % 100 == 0:
            print(f"  ... {idx + 1}/{len(samples)}", flush=True)

    return {
        "per_label": per_label,
        "per_lang": per_lang,
        "overall": {"tp": overall_tp, "fp": overall_fp, "fn": overall_fn},
        "errors": errors,
        "total_ms": total_ms,
    }


def print_table(title, data_dict, key_label="Category"):
    print(f"\n### {title}\n")
    print(f"| {key_label:<20} | {'Prec':>8} | {'Recall':>8} | {'F1':>8} | {'TP':>5} | {'FP':>5} | {'FN':>5} |")
    print(f"|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*7}|{'-'*7}|{'-'*7}|")
    for key in sorted(data_dict.keys()):
        s = data_dict[key]
        p, r, f1 = prf(s["tp"], s["fp"], s["fn"])
        print(f"| {key:<20} | {p:>7.1%} | {r:>7.1%} | {f1:>7.1%} | {s['tp']:>5} | {s['fp']:>5} | {s['fn']:>5} |")


def print_comparison(title, raw, full, key_label="Category"):
    all_keys = sorted(set(list(raw.keys()) + list(full.keys())))
    print(f"\n### {title}\n")
    print(f"| {key_label:<20} | {'Model F1':>10} | {'Model+Regex F1':>14} | {'Delta':>8} |")
    print(f"|{'-'*22}|{'-'*12}|{'-'*16}|{'-'*10}|")
    for key in all_keys:
        r = raw.get(key, {"tp": 0, "fp": 0, "fn": 0})
        f = full.get(key, {"tp": 0, "fp": 0, "fn": 0})
        _, _, rf1 = prf(r["tp"], r["fp"], r["fn"])
        _, _, ff1 = prf(f["tp"], f["fp"], f["fn"])
        delta = ff1 - rf1
        d_str = f"+{delta:.1%}" if delta > 0 else f"{delta:.1%}" if delta < 0 else "="
        print(f"| {key:<20} | {rf1:>9.1%} | {ff1:>13.1%} | {d_str:>8} |")


def main():
    parser = argparse.ArgumentParser(description="Multilingual PII eval: model vs model+regex")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", default="model")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.count} multilingual test samples (seed={args.seed})...")
    samples = generate_dataset(args.count, args.seed)
    total_spans = sum(len(s["spans"]) for s in samples)
    print(f"  {len(samples)} samples, {total_spans} ground-truth spans\n")

    # Language distribution
    lang_dist = {}
    for s in samples:
        lang = LANG_DISPLAY.get(s.get("lang", "en"), s.get("lang", "English"))
        lang_dist[lang] = lang_dist.get(lang, 0) + 1
    print("Language distribution:")
    for lang, cnt in sorted(lang_dist.items()):
        print(f"  {lang}: {cnt}")

    print(f"\n{'='*70}")
    print("Phase 1: Model only (--no-regex)")
    print(f"{'='*70}")
    raw_results = evaluate_samples(args.binary, args.model, samples, no_regex=True)
    rp, rr, rf1 = prf(raw_results["overall"]["tp"], raw_results["overall"]["fp"], raw_results["overall"]["fn"])
    print(f"\nModel only — Overall: P={rp:.1%}  R={rr:.1%}  F1={rf1:.1%}")
    print_table("Model only — Per label", raw_results["per_label"], "Label")
    print_table("Model only — Per language", raw_results["per_lang"], "Language")

    print(f"\n{'='*70}")
    print("Phase 2: Model + Regex backstop")
    print(f"{'='*70}")
    full_results = evaluate_samples(args.binary, args.model, samples, no_regex=False)
    fp, fr, ff1 = prf(full_results["overall"]["tp"], full_results["overall"]["fp"], full_results["overall"]["fn"])
    print(f"\nModel+Regex — Overall: P={fp:.1%}  R={fr:.1%}  F1={ff1:.1%}")
    print_table("Model+Regex — Per label", full_results["per_label"], "Label")
    print_table("Model+Regex — Per language", full_results["per_lang"], "Language")

    print(f"\n{'='*70}")
    print("Comparison: Model vs Model+Regex")
    print(f"{'='*70}")

    overall_delta = ff1 - rf1
    print(f"\nOverall F1: {rf1:.1%} -> {ff1:.1%}  ({'+' if overall_delta >= 0 else ''}{overall_delta:.1%})")
    print_comparison("F1 by label", raw_results["per_label"], full_results["per_label"], "Label")
    print_comparison("F1 by language", raw_results["per_lang"], full_results["per_lang"], "Language")

    avg_raw = raw_results["total_ms"] / len(samples) if samples else 0
    avg_full = full_results["total_ms"] / len(samples) if samples else 0
    print(f"\nAvg latency: model={avg_raw:.0f}ms  model+regex={avg_full:.0f}ms")
    print(f"Errors: model={raw_results['errors']}  model+regex={full_results['errors']}")

    # Save JSON results
    out = {
        "samples": len(samples),
        "spans": total_spans,
        "model_only": {
            "overall": {"precision": round(rp, 4), "recall": round(rr, 4), "f1": round(rf1, 4)},
            "per_label": {k: {"f1": round(prf(v["tp"], v["fp"], v["fn"])[2], 4)} for k, v in raw_results["per_label"].items()},
            "per_lang": {k: {"f1": round(prf(v["tp"], v["fp"], v["fn"])[2], 4)} for k, v in raw_results["per_lang"].items()},
        },
        "model_regex": {
            "overall": {"precision": round(fp, 4), "recall": round(fr, 4), "f1": round(ff1, 4)},
            "per_label": {k: {"f1": round(prf(v["tp"], v["fp"], v["fn"])[2], 4)} for k, v in full_results["per_label"].items()},
            "per_lang": {k: {"f1": round(prf(v["tp"], v["fp"], v["fn"])[2], 4)} for k, v in full_results["per_lang"].items()},
        },
    }
    with open("tests/eval_multilingual_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nResults saved to tests/eval_multilingual_results.json")


if __name__ == "__main__":
    main()

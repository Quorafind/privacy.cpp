#!/usr/bin/env python3
"""Evaluate privacy detection accuracy against a ground-truth JSONL dataset.

Reads the JSONL file produced by generate_test_set.py, runs the `privacy` CLI
on each sample, and computes precision, recall, and F1 per label and overall.

Matching: a predicted span matches a ground-truth span if they share the same
label AND their character ranges overlap by at least 50% (IoU >= 0.5).

Usage:
    python tools/eval_accuracy.py --dataset tests/bench_dataset.jsonl --binary build/Release/privacy.exe [--model model]
"""

import argparse
import json
import subprocess
import sys
import time


def iou(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def run_predict(binary, model_dir, text):
    cmd = [binary, "--model", model_dir, "--", text]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate PII detection accuracy")
    parser.add_argument("--dataset", required=True, help="Path to bench_dataset.jsonl")
    parser.add_argument("--binary", required=True, help="Path to privacy binary")
    parser.add_argument("--model", default="model", help="Model directory")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0=all)")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if args.limit > 0:
        samples = samples[:args.limit]

    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    per_label = {}
    errors = 0
    total_predict_ms = 0.0

    print(f"Evaluating {len(samples)} samples...", flush=True)

    for idx, sample in enumerate(samples):
        text = sample["text"]
        truth = sample["spans"]

        t0 = time.perf_counter()
        preds = run_predict(args.binary, args.model, text)
        t1 = time.perf_counter()
        total_predict_ms += (t1 - t0) * 1000.0

        if preds is None:
            errors += 1
            overall_fn += len(truth)
            continue

        tp, fp, fn = match_spans(truth, preds, args.iou)
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        for t_span in truth:
            lbl = t_span["label"]
            if lbl not in per_label:
                per_label[lbl] = {"tp": 0, "fp": 0, "fn": 0}
        for p_span in preds:
            lbl = p_span["label"]
            if lbl not in per_label:
                per_label[lbl] = {"tp": 0, "fp": 0, "fn": 0}

        truth_matched = set()
        pred_matched = set()
        for i, t_span in enumerate(truth):
            for j, p_span in enumerate(preds):
                if j in pred_matched:
                    continue
                if t_span["label"] != p_span["label"]:
                    continue
                if iou(t_span["start"], t_span["end"], p_span["start"], p_span["end"]) >= args.iou:
                    per_label[t_span["label"]]["tp"] += 1
                    truth_matched.add(i)
                    pred_matched.add(j)
                    break

        for i, t_span in enumerate(truth):
            if i not in truth_matched:
                per_label[t_span["label"]]["fn"] += 1
        for j, p_span in enumerate(preds):
            if j not in pred_matched:
                per_label[p_span["label"]]["fp"] += 1

        if (idx + 1) % 100 == 0:
            print(f"  ... {idx + 1}/{len(samples)}", flush=True)

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    print()
    print("=" * 72)
    print(f"{'Label':<20} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 72)

    for lbl in sorted(per_label.keys()):
        s = per_label[lbl]
        p, r, f1 = prf(s["tp"], s["fp"], s["fn"])
        print(f"{lbl:<20} {p:>8.1%} {r:>8.1%} {f1:>8.1%} {s['tp']:>6} {s['fp']:>6} {s['fn']:>6}")

    print("-" * 72)
    p, r, f1 = prf(overall_tp, overall_fp, overall_fn)
    print(f"{'OVERALL':<20} {p:>8.1%} {r:>8.1%} {f1:>8.1%} {overall_tp:>6} {overall_fp:>6} {overall_fn:>6}")
    print("=" * 72)

    avg_ms = total_predict_ms / len(samples) if samples else 0
    print(f"\nSamples: {len(samples)}  |  Errors: {errors}  |  Avg latency: {avg_ms:.1f} ms/sample")
    print(f"Total wall time: {total_predict_ms / 1000.0:.1f} s")

    results = {
        "samples": len(samples),
        "errors": errors,
        "overall": {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                     "tp": overall_tp, "fp": overall_fp, "fn": overall_fn},
        "per_label": {},
        "avg_latency_ms": round(avg_ms, 1),
        "total_seconds": round(total_predict_ms / 1000.0, 1),
    }
    for lbl in sorted(per_label.keys()):
        s = per_label[lbl]
        lp, lr, lf = prf(s["tp"], s["fp"], s["fn"])
        results["per_label"][lbl] = {
            "precision": round(lp, 4), "recall": round(lr, 4), "f1": round(lf, 4),
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
        }

    json_out = args.dataset.replace(".jsonl", "_results.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_out}")


if __name__ == "__main__":
    main()

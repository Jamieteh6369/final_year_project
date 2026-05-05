"""
TTP Pipeline Evaluation
------------------------
Reconstructs the exact validation splits used during training so the
evaluation only tests on sentences the models have NEVER seen before.

Stage 1 val split : 20% of full dataset  (random_state=42, stratify=tactic)
Stage 2 val split : 20% of rare-filtered dataset (random_state=42, stratify=technique)
Evaluation pool   : intersection of both val sets

Usage:
    python evaluate.py [--samples 300] [--threshold 0.80] [--seed 42]

Examples:
    python evaluate.py
    python evaluate.py --samples 500 --threshold 0.75
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import TTPPipeline

DATASET_PATH = (
    Path(__file__).parent.parent / "model" / "datasets" / "complete_clean_training_dataset.csv"
)


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the TTP two-stage pipeline.")
    parser.add_argument("--samples",   type=int,   default=300,  help="Number of rows to sample (default: 300)")
    parser.add_argument("--threshold", type=float, default=0.80, help="Confidence threshold (default: 0.80)")
    parser.add_argument("--seed",      type=int,   default=42,   help="Random seed (default: 42)")
    return parser.parse_args()


def bar(value: float, width: int = 30) -> str:
    """Simple ASCII progress bar for a 0-1 float."""
    filled = int(round(value * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value:.1%}"


def threshold_sweep(results: list[dict]) -> None:
    """Show pass rate and accuracy at every 0.05 threshold step."""
    thresholds = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]
    total = len(results)

    print("\n  Thresh │ Pass S1 │  S1 Acc  │ Pass S2 │  S2 Acc  │ Both Correct")
    print("  " + "─" * 65)

    for t in thresholds:
        # Stage 1 at this threshold
        s1_pass   = [r for r in results if r["tactic_confidence"] >= t]
        s1_acc    = (
            sum(1 for r in s1_pass if r["tactic_correct"]) / len(s1_pass)
            if s1_pass else 0
        )

        # Stage 2 at this threshold (only rows that also passed S1)
        s2_pass   = [r for r in s1_pass if r["technique_confidence"] is not None
                     and r["technique_confidence"] >= t]
        s2_acc    = (
            sum(1 for r in s2_pass if r["technique_correct"]) / len(s2_pass)
            if s2_pass else 0
        )

        both_correct = sum(
            1 for r in s2_pass
            if r["tactic_correct"] and r["technique_correct"]
        )
        both_pct = both_correct / len(s2_pass) if s2_pass else 0

        print(
            f"  {t:.2f}  │ {len(s1_pass):>3}/{total} "
            f"│ {s1_acc:>7.1%}  "
            f"│ {len(s2_pass):>3}/{total} "
            f"│ {s2_acc:>7.1%}  "
            f"│ {both_pct:.1%} ({both_correct}/{len(s2_pass) if s2_pass else 0})"
        )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Load dataset and reconstruct true validation splits ───────────────
    print(f"\nLoading dataset from {DATASET_PATH.name}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Full dataset : {len(df):,} rows  |  {df['tactic'].nunique()} tactics  |  {df['labels'].nunique()} techniques")

    # Reconstruct Stage 1 val split (same params used during training)
    y_s1 = LabelEncoder().fit_transform(df["tactic"])
    s1_train_idx, s1_val_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42, stratify=y_s1
    )
    s1_val_indices = set(s1_val_idx)

    # Reconstruct Stage 2 val split (filter rare classes first, same as training)
    df_s2 = df.copy()
    valid_labels = df_s2["labels"].value_counts()
    valid_labels = valid_labels[valid_labels >= 2].index
    df_s2 = df_s2[df_s2["labels"].isin(valid_labels)]
    y_s2 = LabelEncoder().fit_transform(df_s2["labels"])
    s2_train_idx, s2_val_idx = train_test_split(
        range(len(df_s2)), test_size=0.2, random_state=42, stratify=y_s2
    )
    # Map Stage 2 val indices back to original df positions
    df_s2_original_indices = df_s2.index.tolist()
    s2_val_original = {df_s2_original_indices[i] for i in s2_val_idx}

    # Intersection: rows unseen by BOTH models
    unseen_indices = sorted(s1_val_indices & s2_val_original)
    val_pool = df.iloc[unseen_indices].reset_index(drop=True)

    print(f"  Stage 1 val  : {len(s1_val_idx):,} rows  (never seen by Stage 1)")
    print(f"  Stage 2 val  : {len(s2_val_idx):,} rows  (never seen by Stage 2)")
    print(f"  Intersection : {len(val_pool):,} rows  (never seen by either model)")

    n = min(args.samples, len(val_pool))
    sample = val_pool.sample(n=n, random_state=args.seed).reset_index(drop=True)
    print(f"  Sampled      : {n} rows  (seed={args.seed})")

    # ── 2. Load pipeline ──────────────────────────────────────────────────────
    print("\nLoading TTP pipeline...")
    pipeline = TTPPipeline()

    # ── 3. Run predictions ────────────────────────────────────────────────────
    print(f"\nRunning predictions (threshold={args.threshold:.0%})...\n")
    results = []

    for i, row in sample.iterrows():
        print(f"  [{i+1:>3}/{len(sample)}]", end="\r")

        pred = pipeline.predict(row["sentence"], threshold=args.threshold)
        pred["true_tactic"]     = row["tactic"]
        pred["true_technique"]  = row["labels"]
        pred["tactic_correct"]  = pred["tactic"] == row["tactic"]
        pred["technique_correct"] = (
            pred["technique"] == row["labels"]
            if pred["technique"] is not None else False
        )
        results.append(pred)

    print(f"  Done. {len(results)} sentences processed.")

    # ── 4. Compute metrics ────────────────────────────────────────────────────
    total = len(results)

    # Stage 1
    s1_pass    = [r for r in results if r["status"] != "flagged_stage1"]
    s1_flagged = [r for r in results if r["status"] == "flagged_stage1"]
    s1_correct = sum(1 for r in results if r["tactic_correct"])
    s1_correct_passed = sum(1 for r in s1_pass if r["tactic_correct"])

    # Stage 2
    s2_pass    = [r for r in results if r["status"] == "passed"]
    s2_flagged = [r for r in results if r["status"] == "flagged_stage2"]
    s2_correct = sum(1 for r in s2_pass if r["technique_correct"])

    # End-to-end
    both_correct = sum(1 for r in s2_pass if r["tactic_correct"] and r["technique_correct"])

    # ── 5. Print report ───────────────────────────────────────────────────────
    sep = "=" * 70

    print(f"\n{sep}")
    print("  TTP PIPELINE EVALUATION REPORT")
    print(sep)
    print(f"  Dataset   : {DATASET_PATH.name}")
    print(f"  Samples   : {total}  (seed={args.seed})")
    print(f"  Threshold : {args.threshold:.0%}")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STAGE 1 — Tactic Prediction")
    print(f"{'─'*70}")

    s1_acc_all    = s1_correct / total
    s1_acc_passed = s1_correct_passed / len(s1_pass) if s1_pass else 0
    s1_pass_rate  = len(s1_pass) / total
    avg_s1_conf   = np.mean([r["tactic_confidence"] for r in results])
    avg_s1_conf_correct   = np.mean([r["tactic_confidence"] for r in results if r["tactic_correct"]]) if s1_correct else 0
    avg_s1_conf_incorrect = np.mean([r["tactic_confidence"] for r in results if not r["tactic_correct"]]) if (total - s1_correct) else 0

    print(f"\n  Overall tactic accuracy  : {bar(s1_acc_all)}")
    print(f"  Accuracy on passed only  : {bar(s1_acc_passed)}")
    print(f"\n  Pass rate (≥{args.threshold:.0%} conf)  : {bar(s1_pass_rate)}  ({len(s1_pass)}/{total} sentences)")
    print(f"  Flagged at Stage 1       : {len(s1_flagged)} sentences ({len(s1_flagged)/total:.1%})")
    print(f"\n  Avg confidence (all)     : {avg_s1_conf:.1%}")
    print(f"  Avg confidence (correct) : {avg_s1_conf_correct:.1%}")
    print(f"  Avg confidence (wrong)   : {avg_s1_conf_incorrect:.1%}")

    # Tactic breakdown
    print(f"\n  {'Tactic':<28} {'Total':>5}  {'Correct':>7}  {'Accuracy':>8}  {'Avg Conf':>8}")
    print(f"  {'─'*28} {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}")
    for tactic in sorted(df["tactic"].unique()):
        rows = [r for r in results if r["true_tactic"] == tactic]
        if not rows:
            continue
        correct = sum(1 for r in rows if r["tactic_correct"])
        avg_conf = np.mean([r["tactic_confidence"] for r in rows])
        acc = correct / len(rows)
        print(f"  {tactic:<28} {len(rows):>5}  {correct:>7}  {acc:>7.1%}   {avg_conf:>7.1%}")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STAGE 2 — Technique Prediction")
    print(f"{'─'*70}")

    s2_pass_rate   = len(s2_pass) / total
    s2_acc         = s2_correct / len(s2_pass) if s2_pass else 0
    avg_s2_conf_correct   = np.mean([r["technique_confidence"] for r in s2_pass if r["technique_correct"]]) if s2_correct else 0
    avg_s2_conf_incorrect = np.mean([r["technique_confidence"] for r in s2_pass if not r["technique_correct"]]) if (len(s2_pass) - s2_correct) else 0

    print(f"\n  Sentences reaching Stage 2 : {len(s1_pass)}/{total} ({len(s1_pass)/total:.1%})")
    print(f"  Pass rate (≥{args.threshold:.0%} conf)   : {bar(s2_pass_rate)}  ({len(s2_pass)}/{total} sentences)")
    print(f"  Flagged at Stage 2         : {len(s2_flagged)} sentences ({len(s2_flagged)/total:.1%})")
    print(f"\n  Technique accuracy (passed): {bar(s2_acc)}")
    print(f"  Avg confidence (correct)   : {avg_s2_conf_correct:.1%}")
    print(f"  Avg confidence (wrong)     : {avg_s2_conf_incorrect:.1%}")

    # ── End-to-end ────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  END-TO-END SUMMARY")
    print(f"{'─'*70}")

    e2e_rate = both_correct / len(s2_pass) if s2_pass else 0

    print(f"\n  Of {total} input sentences:")
    print(f"    Passed both stages (confident)   : {len(s2_pass):>3}  ({len(s2_pass)/total:.1%})")
    print(f"    Flagged at Stage 1               : {len(s1_flagged):>3}  ({len(s1_flagged)/total:.1%})")
    print(f"    Flagged at Stage 2               : {len(s2_flagged):>3}  ({len(s2_flagged)/total:.1%})")
    print(f"\n  Of {len(s2_pass)} confident predictions:")
    print(f"    Both tactic + technique correct  : {both_correct:>3}  {bar(e2e_rate)}")
    print(f"    Tactic correct, technique wrong  : {sum(1 for r in s2_pass if r['tactic_correct'] and not r['technique_correct']):>3}")
    print(f"    Tactic wrong, technique wrong    : {sum(1 for r in s2_pass if not r['tactic_correct']):>3}")

    # ── Threshold sweep ───────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  THRESHOLD SWEEP  (pass rate vs accuracy at each confidence level)")
    print(f"{'─'*70}")
    threshold_sweep(results)

    # ── Flagged at Stage 1 ────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  SENTENCES FLAGGED AT STAGE 1  ({len(s1_flagged)} total — tactic confidence below {args.threshold:.0%})")
    print(f"{'─'*70}")
    if s1_flagged:
        s1_correct_flagged   = sum(1 for r in s1_flagged if r["tactic_correct"])
        s1_incorrect_flagged = len(s1_flagged) - s1_correct_flagged
        print(f"\n  Of the {len(s1_flagged)} flagged sentences:")
        print(f"    Tactic was actually correct : {s1_correct_flagged}  (model was right but not confident enough)")
        print(f"    Tactic was actually wrong   : {s1_incorrect_flagged}  (model was genuinely confused)")
        print()
        for i, r in enumerate(s1_flagged, 1):
            mark = "correct" if r["tactic_correct"] else "wrong  "
            print(f"  {i:>3}. [{mark}]  conf={r['tactic_confidence']:.0%}  predicted={r['tactic']:<25} actual={r['true_tactic']}")
            print(f"       {r['sentence'][:110]}{'...' if len(r['sentence']) > 110 else ''}")
            print()
    else:
        print("\n  No sentences were flagged at Stage 1.")

    # ── Sample predictions ────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  SAMPLE PREDICTIONS (first 50 confident results)")
    print(f"{'─'*70}")
    for r in s2_pass[:50]:
        t_mark = "✓" if r["tactic_correct"]    else "✗"
        l_mark = "✓" if r["technique_correct"] else "✗"
        print(f"\n  [{t_mark}] Tactic    : predicted={r['tactic']:<25} actual={r['true_tactic']}")
        print(f"  [{l_mark}] Technique : predicted={r['technique']:<25} actual={r['true_technique']}")
        print(f"      Conf   : Stage1={r['tactic_confidence']:.0%}  Stage2={r['technique_confidence']:.0%}")
        print(f"      Input  : {r['sentence'][:100]}{'...' if len(r['sentence']) > 100 else ''}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()

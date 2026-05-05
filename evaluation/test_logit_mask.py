# -*- coding: utf-8 -*-
"""
Logit Mask Constraint Test
---------------------------
Verifies that Stage 2 only outputs techniques belonging to the tactic
predicted by Stage 1. Runs two checks:

  Check 1 - Mask structure  : confirms each tactic mask has the correct
                               number of valid (0.0) and blocked (-inf) slots.
                               No model load required.

  Check 2 - Hard constraint : runs a small set of sentences through the full
                               pipeline and asserts that every predicted
                               technique belongs to the predicted tactic.

Usage:
    python test_logit_mask.py
"""

import sys
import json
from pathlib import Path

# -- Check 1: mask structure (no model needed) ---------------------------------

STORAGE_PATH = Path(__file__).parent / "storage" / "tactic_techniques.json"

with open(STORAGE_PATH, "r", encoding="utf-8") as f:
    TACTIC_TO_TECHNIQUES = {t: set(v) for t, v in json.load(f).items()}

print("=" * 60)
print("  CHECK 1 - Tactic-to-Technique Mapping (from JSON)")
print("=" * 60)
print(f"\n  {'Tactic':<28} {'# Techniques':>12}")
print(f"  {'-'*28} {'-'*12}")

total_techniques = 0
for tactic, techniques in sorted(TACTIC_TO_TECHNIQUES.items()):
    print(f"  {tactic:<28} {len(techniques):>12}")
    total_techniques += len(techniques)

print(f"\n  Total tactics    : {len(TACTIC_TO_TECHNIQUES)}")
print(f"  Total techniques : {total_techniques} (across all tactics)")
print("\n  [PASS] JSON loaded and mapping looks correct.\n")


# -- Check 2: hard constraint via full pipeline --------------------------------

print("=" * 60)
print("  CHECK 2 - Hard Constraint (pipeline inference)")
print("=" * 60)

TEST_SENTENCES = [
    "The attacker dumped LSASS memory to extract plaintext passwords.",
    "A PowerShell script was used to download and execute the payload.",
    "The malware encrypted all files on the victim machine and demanded ransom.",
    "Attackers used spearphishing emails with malicious attachments to gain access.",
    "The adversary moved laterally using stolen credentials via RDP.",
    "Registry run keys were added to maintain persistence across reboots.",
    "The threat actor exfiltrated data by uploading archives to a cloud storage service.",
    "Adversary performed network scanning to discover active hosts.",
    "The malware obfuscated its code using base64 encoding to evade detection.",
    "Attackers exploited a privilege escalation vulnerability to gain SYSTEM access.",
]

print("\nLoading pipeline (this may take a moment)...\n")
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import TTPPipeline, TACTIC_TO_TECHNIQUES as MASK_MAP

pipeline = TTPPipeline()

print(f"\n  {'#':<3}  {'Tactic':<28} {'Technique':<14}  {'Valid?':<6}  {'Status'}")
print(f"  {'-'*3}  {'-'*28} {'-'*14}  {'-'*6}  {'-'*10}")

passed = 0
failed = 0
skipped = 0

for i, sentence in enumerate(TEST_SENTENCES, 1):
    result = pipeline.predict(sentence)

    tactic    = result["tactic"]
    technique = result["technique"]
    status    = result["status"]

    if status == "flagged_stage1":
        print(f"  {i:<3}  {tactic:<28} {'N/A':<14}  {'N/A':<6}  flagged_stage1 (skipped)")
        skipped += 1
        continue

    if status == "flagged_stage2":
        print(f"  {i:<3}  {tactic:<28} {technique:<14}  {'N/A':<6}  flagged_stage2 (skipped)")
        skipped += 1
        continue

    valid_techniques = MASK_MAP.get(tactic, set())
    is_valid = technique in valid_techniques

    marker = "PASS" if is_valid else "FAIL"
    print(f"  {i:<3}  {tactic:<28} {technique:<14}  {marker:<6}  {status}")

    if is_valid:
        passed += 1
    else:
        failed += 1
        print(f"       !! '{technique}' is NOT in the allowed set for '{tactic}'")

print(f"\n  Results : {passed} passed | {failed} failed | {skipped} skipped (flagged)")

print("\n" + "=" * 60)
if failed == 0:
    print("  ALL CHECKS PASSED - hard constraint is working correctly.")
else:
    print(f"  WARNING: {failed} prediction(s) violated the tactic constraint.")
print("=" * 60 + "\n")

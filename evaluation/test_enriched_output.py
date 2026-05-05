# -*- coding: utf-8 -*-
"""
Enriched TTP Output Test
-------------------------
Runs the pipeline on sample sentences and prints the full enriched output:

  - Sentence
  - Tactic + definition
  - Technique + definition (from MITRE ATT&CK STIX)
  - Overall confidence score
  - Severity level
  - Reference link

The MITRE ATT&CK STIX bundle is downloaded once and cached in
backend/storage/enterprise-attack.json (~30 MB).

Usage:
    python test_enriched_output.py
"""

import sys
import json
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BACKEND_DIR  = Path(__file__).parent
STORAGE_DIR  = BACKEND_DIR / "storage"
STIX_CACHE   = STORAGE_DIR / "enterprise-attack.json"
TACTIC_META  = STORAGE_DIR / "tactic_metadata.json"

STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)

# ---------------------------------------------------------------------------
# Load tactic metadata (definitions + severity)
# ---------------------------------------------------------------------------

with open(TACTIC_META, "r", encoding="utf-8") as f:
    TACTIC_METADATA = json.load(f)

# ---------------------------------------------------------------------------
# Download STIX bundle once, cache locally
# ---------------------------------------------------------------------------

def ensure_stix() -> bool:
    if STIX_CACHE.exists():
        return True
    print("  Downloading MITRE ATT&CK STIX bundle (one-time, ~30 MB)...")
    try:
        r = requests.get(STIX_URL, timeout=60)
        r.raise_for_status()
        STIX_CACHE.write_bytes(r.content)
        print("  Download complete.\n")
        return True
    except Exception as e:
        print(f"  WARNING: Could not download STIX bundle: {e}")
        print("  Technique definitions will not be available.\n")
        return False

# ---------------------------------------------------------------------------
# Load MITRE ATT&CK technique lookups via mitreattack-python
# ---------------------------------------------------------------------------

def load_mitre():
    try:
        from mitreattack.stix20 import MitreAttackData
        return MitreAttackData(str(STIX_CACHE))
    except Exception as e:
        print(f"  WARNING: Could not load MITRE data: {e}")
        return None


def get_technique_definition(mitre, technique_id: str) -> str:
    if mitre is None:
        return "Definition unavailable (STIX bundle not loaded)."
    try:
        obj = mitre.get_object_by_attack_id(technique_id, "attack-pattern")
        if obj and hasattr(obj, "description"):
            desc = obj.description.replace("\n", " ").strip()
            return desc[:300] + "..." if len(desc) > 300 else desc
    except Exception:
        pass
    return "Definition not found in MITRE ATT&CK."


def get_technique_name(mitre, technique_id: str) -> str:
    if mitre is None:
        return technique_id
    try:
        obj = mitre.get_object_by_attack_id(technique_id, "attack-pattern")
        if obj and hasattr(obj, "name"):
            return obj.name
    except Exception:
        pass
    return technique_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reference_link(technique_id: str) -> str:
    parts = technique_id.split(".")
    if len(parts) == 1:
        return f"https://attack.mitre.org/techniques/{parts[0]}/"
    return f"https://attack.mitre.org/techniques/{parts[0]}/{parts[1]}/"


def overall_confidence(tactic_conf: float, technique_conf: float) -> float:
    return round((tactic_conf + technique_conf) / 2, 4)


def severity(tactic: str) -> str:
    return TACTIC_METADATA.get(tactic, {}).get("severity", "Unknown")


def tactic_definition(tactic: str) -> str:
    return TACTIC_METADATA.get(tactic, {}).get("definition", "Definition not available.")

# ---------------------------------------------------------------------------
# Test sentences
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  ENRICHED TTP OUTPUT TEST")
    print("=" * 70)

    # Step 1: ensure STIX data
    stix_ok = ensure_stix()
    mitre   = load_mitre() if stix_ok else None

    # Step 2: load pipeline
    print("  Loading pipeline...\n")
    sys.path.insert(0, str(BACKEND_DIR))
    from pipeline import TTPPipeline
    pipeline = TTPPipeline()
    print()

    # Step 3: run and print enriched output
    for i, sentence in enumerate(TEST_SENTENCES, 1):
        result = pipeline.predict(sentence)
        status = result["status"]

        print("=" * 70)
        print(f"  [{i}] INPUT SENTENCE")
        print(f"      {sentence}")
        print()

        tactic = result["tactic"]
        tactic_conf = result["tactic_confidence"]

        print(f"  TACTIC             : {tactic}")
        print(f"  Tactic Definition  : {tactic_definition(tactic)}")
        print(f"  Tactic Confidence  : {tactic_conf:.0%}")

        if status == "flagged_stage1":
            print(f"\n  STATUS             : FLAGGED (Stage 1 confidence below threshold)")
            print(f"  Severity           : {severity(tactic)}")
            print()
            continue

        technique    = result["technique"]
        tech_conf    = result["technique_confidence"]
        tech_name    = get_technique_name(mitre, technique)
        tech_def     = get_technique_definition(mitre, technique)
        overall_conf = overall_confidence(tactic_conf, tech_conf)
        ref_link     = reference_link(technique)
        sev          = severity(tactic)

        print()
        print(f"  TECHNIQUE          : {technique} - {tech_name}")
        print(f"  Technique Def      : {tech_def}")
        print()
        print(f"  Overall Confidence : {overall_conf:.0%}")
        print(f"  Severity Level     : {sev}")
        print(f"  Reference Link     : {ref_link}")

        if status == "flagged_stage2":
            print(f"  STATUS             : FLAGGED (Stage 2 confidence below threshold)")
        else:
            print(f"  STATUS             : PASSED")

        print()

    print("=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()

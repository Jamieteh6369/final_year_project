"""
TTP Report Enricher
-------------------
Enriches raw pipeline predictions with MITRE ATT&CK metadata:
  - Tactic and technique definitions
  - Severity level (from tactic_metadata.json)
  - Overall confidence score
  - MITRE ATT&CK reference URL

The STIX bundle (~30 MB) is downloaded once on first use and cached at
backend/storage/enterprise-attack.json.
"""

import json
import requests
from pathlib import Path

_STORAGE_DIR = Path(__file__).parent / "storage"
_STIX_CACHE  = _STORAGE_DIR / "enterprise-attack.json"
_TACTIC_META = _STORAGE_DIR / "tactic_metadata.json"
_STIX_URL    = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)

with open(_TACTIC_META, "r", encoding="utf-8") as _f:
    _TACTIC_METADATA: dict = json.load(_f)


# ── STIX bundle management ────────────────────────────────────────────────────

def _ensure_stix() -> bool:
    if _STIX_CACHE.exists():
        return True
    print("[Reporter] Downloading MITRE ATT&CK STIX bundle (one-time, ~30 MB)...")
    try:
        r = requests.get(_STIX_URL, timeout=120)
        r.raise_for_status()
        _STIX_CACHE.write_bytes(r.content)
        print("[Reporter] Download complete.")
        return True
    except Exception as e:
        print(f"[Reporter] WARNING: Could not download STIX bundle — {e}")
        return False


def _load_mitre():
    try:
        from mitreattack.stix20 import MitreAttackData
        return MitreAttackData(str(_STIX_CACHE))
    except Exception as e:
        print(f"[Reporter] WARNING: Could not load MITRE ATT&CK data — {e}")
        return None


# ── Enricher ──────────────────────────────────────────────────────────────────

class MitreEnricher:
    """
    Enriches a raw TTPPipeline prediction dict with MITRE ATT&CK metadata
    and formats it as a human-readable text report.
    """

    def __init__(self):
        stix_ok     = _ensure_stix()
        self._mitre = _load_mitre() if stix_ok else None

    # ── private helpers ───────────────────────────────────────────────────────

    def _technique_info(self, technique_id: str) -> tuple[str, str]:
        if self._mitre is None:
            return technique_id, "Definition unavailable (STIX bundle not loaded)."
        try:
            obj = self._mitre.get_object_by_attack_id(technique_id, "attack-pattern")
            if obj:
                name = getattr(obj, "name", technique_id)
                raw  = getattr(obj, "description", "").replace("\n", " ").strip()
                defn = (raw[:400] + "...") if len(raw) > 400 else raw
                return name, defn
        except Exception:
            pass
        return technique_id, "Definition not found in MITRE ATT&CK."

    @staticmethod
    def _reference_link(technique_id: str) -> str:
        parts = technique_id.split(".")
        if len(parts) == 1:
            return f"https://attack.mitre.org/techniques/{parts[0]}/"
        # sub-technique: T1003.001 → /techniques/T1003/001/
        return f"https://attack.mitre.org/techniques/{parts[0]}/{parts[1].zfill(3)}/"

    @staticmethod
    def _tactic_definition(tactic: str) -> str:
        return _TACTIC_METADATA.get(tactic, {}).get("definition", "Definition not available.")

    @staticmethod
    def _severity(tactic: str) -> str:
        return _TACTIC_METADATA.get(tactic, {}).get("severity", "Unknown")

    # ── public API ────────────────────────────────────────────────────────────

    def enrich(self, prediction: dict) -> dict:
        """
        Return a copy of the prediction dict with these extra keys added:
          tactic_definition, severity,
          technique_name, technique_definition,
          reference_link, overall_confidence
        """
        tactic      = prediction["tactic"]
        tactic_conf = prediction["tactic_confidence"]
        technique   = prediction.get("technique")
        tech_conf   = prediction.get("technique_confidence")

        enriched = {
            **prediction,
            "tactic_definition":    self._tactic_definition(tactic),
            "severity":             self._severity(tactic),
            "technique_name":       None,
            "technique_definition": None,
            "reference_link":       None,
            "overall_confidence":   None,
        }

        if technique:
            tech_name, tech_def = self._technique_info(technique)
            enriched["technique_name"]       = tech_name
            enriched["technique_definition"] = tech_def
            enriched["reference_link"]       = self._reference_link(technique)
            if tech_conf is not None:
                enriched["overall_confidence"] = round(
                    (tactic_conf + tech_conf) / 2, 4
                )

        return enriched

    def format_text(self, enriched: dict) -> str:
        """Return the enriched prediction as a formatted text block."""
        status_label = enriched["status"].upper().replace("_", " ")

        lines = [
            f"Sentence           : {enriched['sentence']}",
            "",
            f"Tactic             : {enriched['tactic']}",
            f"Tactic Definition  : {enriched['tactic_definition']}",
        ]

        technique = enriched.get("technique")
        if technique:
            lines += [
                "",
                f"Technique          : {technique} — {enriched['technique_name']}",
                f"Technique Def      : {enriched['technique_definition']}",
            ]
        else:
            lines += [
                "",
                "Technique          : N/A  (flagged at Stage 1 — tactic confidence below threshold)",
            ]

        lines += ["", f"Status             : {status_label}"]

        if enriched["overall_confidence"] is not None:
            lines.append(
                f"Overall Confidence : {enriched['overall_confidence']:.0%}"
                f"  (Stage 1 {enriched['tactic_confidence']:.0%}"
                f" · Stage 2 {enriched['technique_confidence']:.0%})"
            )
        else:
            lines.append(f"Tactic Confidence  : {enriched['tactic_confidence']:.0%}")

        lines.append(f"Severity Level     : {enriched['severity']}")

        if enriched.get("reference_link"):
            lines.append(f"Reference Link     : {enriched['reference_link']}")

        return "\n".join(lines)

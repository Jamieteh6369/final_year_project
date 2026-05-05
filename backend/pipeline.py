import json
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).parent.parent  # Final-Year-Project/

STAGE1_PATH = BASE_DIR / "model" / "First_stage_classification" / "tactic_classification_model"
STAGE2_PATH = BASE_DIR / "model" / "second_stage_classification" / "second_stage_classification_model"

CONFIDENCE_THRESHOLD = 0.80

_STORAGE_PATH = Path(__file__).parent / "storage" / "tactic_techniques.json"
with open(_STORAGE_PATH, "r", encoding="utf-8") as _f:
    TACTIC_TO_TECHNIQUES = {tactic: set(techniques) for tactic, techniques in json.load(_f).items()}


class TTPPipeline:
    """
    Two-stage TTP classifier.
    Stage 1: sentence          -> tactic   (14 MITRE tactics)
    Stage 2: tactic + sentence -> technique (255 MITRE techniques, tactic-constrained)
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TTPPipeline] Using device: {self.device}")

        print("[TTPPipeline] Loading Stage 1 model...")
        self.stage1_tokenizer = AutoTokenizer.from_pretrained(str(STAGE1_PATH))
        self.stage1_model = AutoModelForSequenceClassification.from_pretrained(str(STAGE1_PATH))
        self.stage1_model.to(self.device).eval()

        print("[TTPPipeline] Loading Stage 2 model...")
        self.stage2_tokenizer = AutoTokenizer.from_pretrained(str(STAGE2_PATH))
        self.stage2_model = AutoModelForSequenceClassification.from_pretrained(str(STAGE2_PATH))
        self.stage2_model.to(self.device).eval()

        self._tactic_masks = self._build_tactic_masks()
        print("[TTPPipeline] Both models ready.")

    # ── private helpers ──────────────────────────────────────────────────────

    def _build_tactic_masks(self) -> dict:
        """Pre-compute a -inf mask tensor per tactic so Stage 2 can only output
        techniques that belong to the predicted tactic."""
        id2label = self.stage2_model.config.id2label
        num_labels = self.stage2_model.config.num_labels
        masks = {}
        for tactic, valid_techniques in TACTIC_TO_TECHNIQUES.items():
            mask = torch.full((num_labels,), float("-inf"))
            for idx, label in id2label.items():
                if label in valid_techniques:
                    mask[idx] = 0.0
            masks[tactic] = mask.to(self.device)
        return masks

    def _run_stage1(self, sentence: str):
        inputs = self.stage1_tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = F.softmax(self.stage1_model(**inputs).logits, dim=-1)
        confidence, predicted_id = torch.max(probs, dim=-1)
        tactic = self.stage1_model.config.id2label[predicted_id.item()]
        return tactic, round(confidence.item(), 4)

    def _run_stage2(self, sentence: str, tactic: str):
        enriched = f"{tactic} {sentence}"
        inputs = self.stage2_tokenizer(
            enriched, return_tensors="pt", truncation=True, max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.stage2_model(**inputs).logits
            if tactic in self._tactic_masks:
                logits = logits + self._tactic_masks[tactic]
            probs = F.softmax(logits, dim=-1)
        confidence, predicted_id = torch.max(probs, dim=-1)
        technique = self.stage2_model.config.id2label[predicted_id.item()]
        return technique, round(confidence.item(), 4)

    # ── public API ───────────────────────────────────────────────────────────

    def predict(self, sentence: str, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
        """
        Run the full two-stage pipeline for a single sentence.

        Returns a dict with keys:
          sentence, tactic, tactic_confidence,
          technique, technique_confidence, status

        status values:
          "passed"            – both stages confident, full TTP mapping returned
          "flagged_stage1"    – Stage 1 confidence below threshold, Stage 2 skipped
          "flagged_stage2"    – Stage 2 confidence below threshold
        """
        tactic, tactic_conf = self._run_stage1(sentence)

        result = {
            "sentence": sentence,
            "stage1_input": sentence,
            "tactic": tactic,
            "tactic_confidence": tactic_conf,
            "stage2_input": None,
            "technique": None,
            "technique_confidence": None,
            "status": None,
        }

        if tactic_conf < threshold:
            result["status"] = "flagged_stage1"
            return result

        stage2_input = f"{tactic} {sentence}"
        technique, tech_conf = self._run_stage2(sentence, tactic)
        result["stage2_input"] = stage2_input
        result["technique"] = technique
        result["technique_confidence"] = tech_conf
        result["status"] = "passed" if tech_conf >= threshold else "flagged_stage2"

        return result

    def predict_batch(self, sentences: list[str], threshold: float = CONFIDENCE_THRESHOLD) -> list[dict]:
        """Run the two-stage pipeline on a list of sentences in batched forward passes."""
        # Stage 1 — one forward pass for all sentences
        inputs = self.stage1_tokenizer(
            sentences, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = F.softmax(self.stage1_model(**inputs).logits, dim=-1)
        s1_confidences, s1_predicted_ids = torch.max(probs, dim=-1)

        # Build initial results and group Stage-1-passing sentences by tactic
        results = []
        tactic_groups: dict[str, list[tuple[int, str]]] = {}

        for i, sentence in enumerate(sentences):
            tactic = self.stage1_model.config.id2label[s1_predicted_ids[i].item()]
            tactic_conf = round(s1_confidences[i].item(), 4)
            result = {
                "sentence": sentence,
                "stage1_input": sentence,
                "tactic": tactic,
                "tactic_confidence": tactic_conf,
                "stage2_input": None,
                "technique": None,
                "technique_confidence": None,
                "status": None,
            }
            if tactic_conf < threshold:
                result["status"] = "flagged_stage1"
            else:
                tactic_groups.setdefault(tactic, []).append((i, sentence))
            results.append(result)

        # Stage 2 — one forward pass per tactic group
        for tactic, group in tactic_groups.items():
            indices, sents = zip(*group)
            enriched = [f"{tactic} {s}" for s in sents]
            inputs = self.stage2_tokenizer(
                list(enriched), padding=True, truncation=True, max_length=256, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.stage2_model(**inputs).logits
                if tactic in self._tactic_masks:
                    logits = logits + self._tactic_masks[tactic]
                probs = F.softmax(logits, dim=-1)
            s2_confidences, s2_predicted_ids = torch.max(probs, dim=-1)

            for j, orig_idx in enumerate(indices):
                technique = self.stage2_model.config.id2label[s2_predicted_ids[j].item()]
                tech_conf = round(s2_confidences[j].item(), 4)
                results[orig_idx]["stage2_input"] = enriched[j]
                results[orig_idx]["technique"] = technique
                results[orig_idx]["technique_confidence"] = tech_conf
                results[orig_idx]["status"] = "passed" if tech_conf >= threshold else "flagged_stage2"

        return results
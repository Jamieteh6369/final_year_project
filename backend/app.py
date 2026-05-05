from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import TTPPipeline
from reporter import MitreEnricher

# ── Models loaded once at startup ────────────────────────────────────────────

pipeline: TTPPipeline | None = None
enricher: MitreEnricher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, enricher
    pipeline = TTPPipeline()
    enricher = MitreEnricher()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TTP Classification API",
    description=(
        "Two-stage MITRE ATT&CK TTP classifier. "
        "Parse a sentence to get its Tactic (Stage 1) and Technique (Stage 2), "
        "enriched with MITRE ATT&CK definitions, severity, and reference links."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    sentence: str
    threshold: float = 0.80


class BatchPredictRequest(BaseModel):
    sentences: list[str]
    threshold: float = 0.80


class PredictResponse(BaseModel):
    sentence: str
    stage1_input: str
    # Stage 1
    tactic: str
    tactic_confidence: float
    tactic_definition: str
    # Stage 2
    stage2_input: str | None
    technique: str | None
    technique_name: str | None
    technique_confidence: float | None
    technique_definition: str | None
    # Summary
    overall_confidence: float | None
    status: str       # "passed" | "flagged_stage1" | "flagged_stage2"
    severity: str
    reference_link: str | None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": pipeline is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.sentence.strip():
        raise HTTPException(status_code=400, detail="sentence cannot be empty")
    raw = pipeline.predict(req.sentence, threshold=req.threshold)
    return enricher.enrich(raw)


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(req: BatchPredictRequest):
    sentences = [s for s in req.sentences if s.strip()]
    if not sentences:
        raise HTTPException(status_code=400, detail="sentences list cannot be empty")
    raws = pipeline.predict_batch(sentences, threshold=req.threshold)
    return [enricher.enrich(r) for r in raws]

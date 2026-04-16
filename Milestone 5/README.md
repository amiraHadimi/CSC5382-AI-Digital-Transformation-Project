# Milestone 5 – ML Productionization
> **Project:** AI-Based Story Point Estimation for Agile Software Development
> **Course:** CSC5382 – AI for Digital Transformation
> **Repository:** github.com/amiraHadimi/CSC5382-AI-Digital-Transformation-Project

![CI/CD](https://github.com/amiraHadimi/CSC5382-AI-Digital-Transformation-Project/actions/workflows/ci.yml/badge.svg)

---

## Table of Contents
1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Requirement 1 · ML System Architecture](#requirement-1--ml-system-architecture-3-pts)
4. [Requirement 2 · Model Serving Modes](#requirement-2--model-serving-modes-3-pts)
5. [Requirement 3 · Model Service Development](#requirement-3--model-service-development-5-pts)
6. [Requirement 4 · Front-end Client](#requirement-4--front-end-client-2-pts)
7. [Requirement 5 · Packaging and Containerization](#requirement-5--packaging-and-containerization-3-pts)
8. [Requirement 6 · CI/CD Pipeline](#requirement-6--cicd-pipeline-5-pts)
9. [Requirement 7 · ML Service Deployment](#requirement-7--ml-service-deployment-3-pts)
10. [Requirement 8 · Model Serving Runtime](#requirement-8--model-serving-runtime-3-pts)
11. [How to Run](#how-to-run)
12. [API Reference](#api-reference)
13. [Grading Coverage Summary](#grading-coverage-summary)

---

## Overview

Milestone 5 wraps the Llama3SP model from Milestones 3–4 into a **production-ready ML service**. The system provides:

- A **FastAPI REST service** (`POST /predict`) for on-demand machine story point estimation
- A **Streamlit web UI** for human-facing interaction
- A **batch prediction script** for processing CSV files of user stories
- **Two Dockerfiles** and a `docker-compose.yml` packaging both services
- A **GitHub Actions CI/CD pipeline** (test → build → deploy on push to main)
- Deployment on **HuggingFace Spaces** (Docker SDK)

---

## Repository Structure

```
Milestone 5/
├── app/
│   ├── __init__.py
│   ├── main.py              ← FastAPI app: POST /predict, GET /health
│   ├── model.py             ← Llama3SP (PEFT/LoRA) loader + word-count fallback
│   └── schemas.py           ← Pydantic PredictionRequest / PredictionResponse
├── frontend/
│   └── app.py               ← Streamlit UI (calls FastAPI via API_URL env var)
├── hf_space/                ← HuggingFace Spaces deployment package
│   ├── README.md            ← HF Space metadata (sdk: docker, app_port: 7860)
│   ├── Dockerfile           ← Streamlit on port 7860, direct model inference
│   ├── app.py               ← Streamlit app (calls model.py directly)
│   ├── model.py             ← Llama3SP loader adapted for single-container HF Space
│   └── requirements.txt
├── tests/
│   └── test_api.py          ← pytest: /health and /predict endpoint tests
├── .github/
│   └── workflows/
│       └── ci.yml           ← GitHub Actions: test → docker build → HF deploy
├── batch_predict.py         ← Batch serving: reads sample_input.csv, calls API
├── sample_input.csv         ← Example batch input (3 issues)
├── sample_output.csv        ← Example batch output with predictions
├── assets/
│   └── architecture.png     ← ML system architecture diagram
├── Dockerfile               ← FastAPI API container (port 8000)
├── Dockerfile.streamlit     ← Streamlit frontend container (port 8501)
├── docker-compose.yml       ← Runs API + Streamlit as linked containers
├── .env                     ← HF_TOKEN (NOT committed — in .gitignore)
├── .gitignore               ← Excludes .env, __pycache__, sample_output.csv
├── .dockerignore
├── requirements.txt
└── README.md                ← This file
```

---

## Requirement 1 · ML System Architecture (3 pts)
**Tool:** Draw.io | **Diagram:** `assets/architecture.png`

The architecture diagram shows the complete production system with three serving modes,
the FastAPI backend, model layer, Docker packaging, CI/CD pipeline, and HuggingFace
Spaces deployment. See `assets/architecture.png`.

**Key design decisions:**
- Two-container local setup (API + Streamlit) vs. single-container HuggingFace Space
- Model loaded once at first request and cached globally (singleton pattern)
- Heuristic fallback ensures the service never crashes even without GPU or HF_TOKEN

---

## Requirement 2 · Model Serving Modes (3 pts)

Three serving modes are implemented:

### Mode 1 — On-demand to a human (Streamlit UI)
`frontend/app.py` — the user fills in a title and description, clicks "Predict", and
sees the estimated story points and which model served the request.

```bash
streamlit run frontend/app.py
# Open http://localhost:8501
```

### Mode 2 — On-demand to a machine (REST API)
`app/main.py` — any system can call `POST /predict` with JSON and receive a structured
JSON response synchronously.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Add OAuth2 login", "description": "Support Google login flow."}'
```

Response:
```json
{"story_points": 3.5, "model_used": "DEVCamiloSepulveda/2-LLAMA3SP-talendesb"}
```

### Mode 3 — Batch prediction
`batch_predict.py` — reads `sample_input.csv`, calls the API for each row, and writes
all predictions to `sample_output.csv`.

```bash
# With API running:
python batch_predict.py
```

Input (`sample_input.csv`):
```
title,description
Fix login bug,Users cannot login after password reset
Add dark mode,Implement a toggle for dark mode in settings
```

---

## Requirement 3 · Model Service Development (5 pts)
**Tool:** FastAPI + Pydantic | **Code:** `app/`

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns `{"status": "ok"}` — used by Docker health checks |
| POST | `/predict` | Predict story points from a user story title + description |

### Request schema (`PredictionRequest`)
```json
{"title": "string (required)", "description": "string (required)"}
```

### Response schema (`PredictionResponse`)
```json
{"story_points": 3.5, "model_used": "DEVCamiloSepulveda/2-LLAMA3SP-talendesb"}
```

### Model loading logic (`app/model.py`)
1. Reads `HF_TOKEN` from environment (via `.env` locally, GitHub/HF secret in CI/CD)
2. Loads `PeftConfig` from `DEVCamiloSepulveda/2-LLAMA3SP-talendesb` to resolve base model
3. Loads `meta-llama/Llama-3.2-1B` tokenizer and `AutoModelForSequenceClassification`
4. Wraps with `PeftModel.from_pretrained` for the LoRA adapter
5. Runs on CUDA if available, CPU otherwise
6. Falls back to word-count heuristic if any step fails (no token, no memory, network error)
7. Model is cached globally — loaded once, reused for all requests

Auto-generated Swagger UI: `http://localhost:8000/docs`

---

## Requirement 4 · Front-end Client (2 pts)
**Tool:** Streamlit | **Code:** `frontend/app.py`

The Streamlit UI provides:
- Text input for issue title
- Text area for issue description
- "Predict" button
- Result display: story points + model name

The `API_URL` is read from the environment variable `API_URL` (defaults to
`http://localhost:8000`), so it works both locally and inside Docker Compose.

Screenshot: `assets/streamlit_screenshot.png`

---

## Requirement 5 · Packaging and Containerization (3 pts)
**Tool:** Docker + Docker Compose | **Code:** `Dockerfile`, `Dockerfile.streamlit`, `docker-compose.yml`

Two Docker images are built from the same codebase:

**`Dockerfile`** (FastAPI API, port 8000):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`Dockerfile.streamlit`** (Streamlit UI, port 8501):
```dockerfile
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

**`docker-compose.yml`** links both:
- `llama3sp_api` — FastAPI on port 8000, loads `.env` for HF_TOKEN
- `llama3sp_frontend` — Streamlit on port 8501, depends on `api`

```bash
docker-compose up --build    # start
docker-compose down          # stop
```

---

## Requirement 6 · CI/CD Pipeline (5 pts)
**Tool:** GitHub Actions | **Code:** `.github/workflows/ci.yml`
**Evidence:** 2 successful workflow runs visible in GitHub Actions tab

The pipeline runs on every push to `main`, `milestone5`, and `milestone4-final`:

| Job | Trigger | Steps |
|---|---|---|
| `test` | every push | Install deps → `pytest tests/ -v` |
| `docker-build` | after tests pass | Build API + Streamlit images |
| `deploy` | push to `main` only | Upload `hf_space/` to HuggingFace Spaces |

### How to set up the secret
1. GitHub repo → **Settings → Secrets and variables → Actions**
2. Click **New repository secret** → Name: `HF_TOKEN` → Value: your token

---

## Requirement 7 · ML Service Deployment (3 pts)
**Tool:** HuggingFace Spaces (Docker SDK)

**Live URL:** https://huggingface.co/spaces/ameeera/llama3sp-story-point-estimator

### Architecture of the HF Space
HuggingFace Spaces only supports one container per Space. The `hf_space/` folder
contains a self-contained version that runs Streamlit directly on top of the model
(no separate FastAPI layer needed):

```
hf_space/
├── Dockerfile     ← FROM python:3.11-slim, runs streamlit on port 7860
├── README.md      ← HF metadata: sdk: docker, app_port: 7860
├── app.py         ← Streamlit UI, imports model.py directly
├── model.py       ← Llama3SP loader + fallback (same logic as app/model.py)
└── requirements.txt
```

### Manual deployment steps
```bash
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi(token="your_hf_token_here")
api.upload_folder(
    folder_path="Milestone 5/hf_space",
    repo_id="amiraHadimi/story-point-estimator",
    repo_type="space",
    commit_message="Deploy Milestone 5"
)
print("Done!")
EOF
```

After the first manual deployment, every push to `main` on GitHub will
auto-deploy via the GitHub Actions `deploy` job.

### HF Space secret
In your Space → **Settings → Repository secrets** → add `HF_TOKEN`.

---

## Requirement 8 · Model Serving Runtime (3 pts)
**Tool:** Transformers + PEFT (LoRA) | **Code:** `app/model.py`

| Component | Value |
|---|---|
| Base model | `meta-llama/Llama-3.2-1B` |
| LoRA adapter | `DEVCamiloSepulveda/2-LLAMA3SP-talendesb` |
| Task | Sequence classification (regression, `num_labels=1`) |
| Device | CUDA if available, CPU otherwise |
| Tokenizer | `AutoTokenizer`, `pad_token = eos_token` |
| Inference | `torch.no_grad()` → `outputs.logits.squeeze().item()` |
| Fallback | Word-count heuristic: `max(1, min(13, words // 8 + 1))` |

---

## How to Run

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- HuggingFace account + token with access to `meta-llama/Llama-3.2-1B`

### Option A — Local (no Docker)

```bash
cd "Milestone 5"

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt

# .env already contains HF_TOKEN — do NOT commit this file

# Terminal 1 — API
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run frontend/app.py
```

- Swagger docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

### Option B — Docker Compose (recommended)

```bash
cd "Milestone 5"
docker-compose up --build
```

- API: http://localhost:8000
- Streamlit: http://localhost:8501

### Run tests

```bash
cd "Milestone 5"
pytest tests/ -v
```

Expected:
```
tests/test_api.py::test_health  PASSED
tests/test_api.py::test_predict PASSED
2 passed
```

### Batch prediction

```bash
# With API running:
python batch_predict.py
# Reads: sample_input.csv → Writes: sample_output.csv
```

---

## API Reference

### GET /health
```json
{"status": "ok"}
```

### POST /predict

**Request:**
```json
{"title": "string (required)", "description": "string (required)"}
```

**Response (200):**
```json
{"story_points": 3.5, "model_used": "DEVCamiloSepulveda/2-LLAMA3SP-talendesb"}
```

**Error (500):**
```json
{"detail": "error message string"}
```

Interactive API docs: http://localhost:8000/docs

---

## Grading Coverage Summary

| Requirement | Tool | Points | Status |
|---|---|---|---|
| ML system architecture drawing | Draw.io | 3 | ✅ `assets/architecture.png` |
| Serving mode — on-demand (human) | Streamlit | 3 | ✅ `frontend/app.py` |
| Serving mode — on-demand (machine) | FastAPI | | ✅ `POST /predict` |
| Serving mode — batch | `batch_predict.py` | | ✅ CSV in → CSV out |
| Model service development | FastAPI + Pydantic | 5 | ✅ `app/main.py`, `app/schemas.py`, `app/model.py` |
| Front-end client | Streamlit | 2 | ✅ `frontend/app.py` |
| Packaging and containerization | Docker + Compose | 3 | ✅ `Dockerfile`, `Dockerfile.streamlit`, `docker-compose.yml` |
| CI/CD pipeline | GitHub Actions | 5 | ✅ `.github/workflows/ci.yml` — 2 green runs |
| Hosting the application | HuggingFace Spaces | 3 | ✅ https://huggingface.co/spaces/amiraHadimi/story-point-estimator |
| Model serving runtime | Llama3SP (PEFT) | 3 | ✅ `app/model.py` |
| **Total** | | **27** | ✅ |

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

- A **FastAPI REST service** (`POST /predict`) for on-demand machine-to-machine story point estimation
- A **Streamlit web UI** for human-facing interaction
- A **batch prediction script** for processing CSV files of user stories
- **Two Dockerfiles** and a `docker-compose.yml` packaging both services
- A **GitHub Actions CI/CD pipeline** (test → build → deploy)
- Deployment on **HuggingFace Spaces**

---

## Repository Structure

```
Milestone 5/
├── app/
│   ├── __init__.py
│   ├── main.py              ← FastAPI app: POST /predict, GET /health
│   ├── model.py             ← Llama3SP (PEFT/LoRA) loader + heuristic fallback
│   └── schemas.py           ← Pydantic PredictionRequest / PredictionResponse
├── frontend/
│   └── app.py               ← Streamlit UI (calls FastAPI backend)
├── hf_space/                ← HuggingFace Space deployment package
│   ├── README.md            ← HF Space metadata (sdk: docker, app_port: 8501)
│   ├── Dockerfile           ← Single container: supervisord runs API + Streamlit
│   ├── supervisord.conf     ← Runs uvicorn (port 8000) + streamlit (port 8501)
│   ├── requirements.txt
│   └── frontend_app.py      ← Streamlit app adapted for HF Space
├── tests/
│   └── test_api.py          ← pytest: /health and /predict endpoint tests
├── .github/
│   └── workflows/
│       └── ci.yml           ← GitHub Actions: test → docker build → HF deploy
├── batch_predict.py         ← Batch serving: reads CSV, calls API, saves output
├── sample_input.csv         ← Example batch input
├── sample_output.csv        ← Example batch output
├── Dockerfile               ← FastAPI API container
├── Dockerfile.streamlit     ← Streamlit frontend container
├── docker-compose.yml       ← Runs API + Streamlit as linked containers
├── .env                     ← HF_TOKEN (NOT committed to Git — in .gitignore)
├── .gitignore
├── .dockerignore
├── requirements.txt
└── README.md                ← This file
```

---

## Requirement 1 · ML System Architecture (3 pts)
**Tool:** Draw.io

The architecture diagram (`assets/architecture.png`) shows the complete production system:

```
 Human User                   External System
     │                              │
     ▼                              ▼
┌──────────────────┐    ┌───────────────────────┐
│  Streamlit UI    │    │   batch_predict.py     │
│  (port 8501)     │    │   (CSV batch mode)     │
└────────┬─────────┘    └──────────┬────────────┘
         │ HTTP POST /predict      │ HTTP POST /predict
         └─────────────┬───────────┘
                       ▼
         ┌─────────────────────────┐
         │     FastAPI Service     │
         │      (port 8000)        │
         │  POST /predict          │
         │  GET  /health           │
         └──────────┬──────────────┘
                    │
         ┌──────────▼──────────────┐
         │      app/model.py       │
         │  ┌─────────────────┐    │
         │  │   Llama3SP      │    │
         │  │ (PEFT/LoRA)     │◄───┼── HuggingFace Hub
         │  └────────┬────────┘    │   DEVCamiloSepulveda/
         │           │ fallback    │   2-LLAMA3SP-talendesb
         │  ┌────────▼────────┐    │
         │  │  Heuristic      │    │
         │  │  (word count)   │    │
         │  └─────────────────┘    │
         └─────────────────────────┘

 ┌──────────────────────────────────────────┐
 │  Docker Compose (local)                  │
 │  llama3sp_api (Dockerfile)               │
 │  llama3sp_frontend (Dockerfile.streamlit)│
 └──────────────────────────────────────────┘

 ┌──────────────────────────────────────────┐
 │  HuggingFace Spaces (production)         │
 │  Single Docker container                 │
 │  supervisord → API (8000) + UI (8501)    │
 └──────────────────────────────────────────┘

 GitHub → GitHub Actions → HuggingFace Spaces
          (test → build → deploy on push to main)
```

---

## Requirement 2 · Model Serving Modes (3 pts)

Three serving modes are implemented:

### Mode 1 — On-demand to a human (Streamlit)
The Streamlit frontend (`frontend/app.py`) provides a web form. The user types a title and description, clicks "Estimate Story Points", and receives a prediction with a Fibonacci scale visualization.

### Mode 2 — On-demand to a machine (REST API)
The FastAPI service exposes `POST /predict`. Any system (CI pipeline, JIRA plugin, another microservice) can call it with JSON and receive a structured JSON response.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Add OAuth2 login", "description": "Support Google login flow."}'
```

### Mode 3 — Batch (batch_predict.py)
`batch_predict.py` reads a CSV file of issues, calls the API for each row, and writes predictions to an output CSV. Example:

```bash
# With the API running:
python batch_predict.py
# Reads sample_input.csv → writes sample_output.csv
```

Input format (`sample_input.csv`):
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
| GET | `/health` | Health check — returns `{"status": "ok"}` |
| POST | `/predict` | Predict story points for a user story |

### POST /predict — Request schema (`PredictionRequest`)

```json
{
  "title": "Add OAuth2 login",
  "description": "Support Google and GitHub login flows with token refresh."
}
```

### POST /predict — Response schema (`PredictionResponse`)

```json
{
  "story_points": 5.23,
  "model_used": "DEVCamiloSepulveda/2-LLAMA3SP-talendesb"
}
```

### Model loading (`app/model.py`)

1. Reads `HF_TOKEN` from environment (`.env` file locally, GitHub Secret in CI/CD, HF Space secret in production)
2. Loads `PeftConfig` from `DEVCamiloSepulveda/2-LLAMA3SP-talendesb` to resolve the base model ID
3. Loads base `meta-llama/Llama-3.2-1B` tokenizer and `AutoModelForSequenceClassification`
4. Loads LoRA adapter via `PeftModel.from_pretrained`
5. Runs on CUDA if available, CPU otherwise
6. If any step fails (no token, no GPU memory, network error): falls back to a word-count heuristic

---

## Requirement 4 · Front-end Client (2 pts)
**Tool:** Streamlit | **Code:** `frontend/app.py`

The Streamlit frontend:
- Text input for issue title
- Text area for description
- "Estimate Story Points" button
- Displays predicted story points and model used
- Shows a Fibonacci scale with the nearest value highlighted
- Connects to FastAPI via `API_URL` environment variable (defaults to `http://api:8000` in Docker Compose, `http://localhost:8000` in HF Space)

Run standalone:
```bash
streamlit run frontend/app.py
```

Screenshot: *(see assets/streamlit_screenshot.png)*

---

## Requirement 5 · Packaging and Containerization (3 pts)
**Tool:** Docker + Docker Compose | **Code:** `Dockerfile`, `Dockerfile.streamlit`, `docker-compose.yml`

Two separate images are built:

**`Dockerfile`** — FastAPI API (port 8000):
- Base: `python:3.11-slim`
- Installs `requirements.txt`
- Runs: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

**`Dockerfile.streamlit`** — Streamlit UI (port 8501):
- Same base image
- Runs: `streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=8501`

**`docker-compose.yml`** — Runs both:
- `llama3sp_api` on port 8000
- `llama3sp_frontend` on port 8501, depends on `api`
- `HF_TOKEN` injected via `.env` file

```bash
# Start everything
docker-compose up --build

# Stop
docker-compose down
```

---

## Requirement 6 · CI/CD Pipeline (5 pts)
**Tool:** GitHub Actions | **Code:** `.github/workflows/ci.yml`

The pipeline runs on every push to `main`, `milestone5`, or `milestone5-final`:

| Job | Trigger | Actions |
|---|---|---|
| `test` | every push | Install dependencies → `pytest tests/ -v` |
| `docker-build` | after tests pass | Build API image + Streamlit image (validates no build errors) |
| `deploy` | push to `main` only | Upload `hf_space/` folder to HuggingFace Spaces via `huggingface_hub` |

### Setup steps for CI/CD
1. Go to your GitHub repo → **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Name: `HF_TOKEN`, Value: your HuggingFace token
4. Push to `main` — the pipeline runs automatically

---

## Requirement 7 · ML Service Deployment (3 pts)
**Tool:** HuggingFace Spaces (Docker SDK)

**Live URL:** https://huggingface.co/spaces/amiraHadimi/story-point-estimator

### Manual deployment steps (one-time setup)

**Step 1 — Create the Space**
1. Go to https://huggingface.co/new-space
2. Space name: `story-point-estimator`
3. SDK: **Docker**
4. Visibility: Public
5. Click **Create Space**

**Step 2 — Add your HF_TOKEN as a Space secret**
1. In your Space → **Settings → Repository secrets**
2. Add secret: `HF_TOKEN` = your HuggingFace token

**Step 3 — Push the `hf_space/` folder**
```bash
# Install huggingface_hub
pip install huggingface_hub

# Push (replace with your username)
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi(token="your_hf_token")
api.upload_folder(
    folder_path="Milestone 5/hf_space",
    repo_id="amiraHadimi/story-point-estimator",
    repo_type="space",
    commit_message="Initial deployment"
)
print("Deployed!")
EOF
```

**Step 4 — Verify**
Visit https://huggingface.co/spaces/amiraHadimi/story-point-estimator — the Space builds the Docker image and starts automatically (takes ~5 minutes on first build).

After this, every push to `main` on GitHub will **auto-deploy** via the GitHub Actions `deploy` job.

---

## Requirement 8 · Model Serving Runtime (3 pts)
**Tool:** Llama3SP (PEFT/LoRA via Transformers)

The model serving runtime is implemented in `app/model.py`:

- **Base model:** `meta-llama/Llama-3.2-1B` (loaded via `AutoModelForSequenceClassification`)
- **Adapter:** `DEVCamiloSepulveda/2-LLAMA3SP-talendesb` (loaded via `PeftModel.from_pretrained`)
- **Tokenizer:** `AutoTokenizer` with `pad_token = eos_token`
- **Device:** CUDA if available, CPU otherwise
- **Inference:** `torch.no_grad()` → `model(**inputs)` → `.logits.squeeze().item()`
- **Fallback:** word-count heuristic when model cannot be loaded

---

## How to Run

### Prerequisites
- Python 3.11+
- Docker and Docker Compose (for containerized run)
- HuggingFace account + token with access to `meta-llama/Llama-3.2-1B`

### Option A — Local without Docker

```bash
cd "Milestone 5"

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt

# Add your HF token to .env (already created, do NOT commit this file)
# .env contains: HF_TOKEN="hf_..."

# Terminal 1 — API
uvicorn app.main:app --reload --port 8000

# Terminal 2 — UI
streamlit run frontend/app.py
```

- API docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

### Option B — Docker Compose

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

Expected output:
```
tests/test_api.py::test_health PASSED
tests/test_api.py::test_predict PASSED
2 passed in 0.XXs
```

### Batch prediction

```bash
# With API running (Option A or B)
python batch_predict.py
# Reads: sample_input.csv
# Writes: sample_output.csv
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
{
  "title": "string (required)",
  "description": "string (required)"
}
```

**Response:**
```json
{
  "story_points": 5.23,
  "model_used": "DEVCamiloSepulveda/2-LLAMA3SP-talendesb"
}
```

**Error (500):**
```json
{"detail": "error message"}
```

Interactive Swagger UI: http://localhost:8000/docs

---

## Grading Coverage Summary

| Requirement | Tool | Points | Status |
|---|---|---|---|
| ML system architecture drawing | Draw.io | 3 | ✅ Diagram in `assets/architecture.png` |
| Serving mode — on-demand (human) | Streamlit | 3 | ✅ `frontend/app.py` |
| Serving mode — on-demand (machine) | FastAPI | | ✅ `POST /predict` |
| Serving mode — batch | `batch_predict.py` | | ✅ CSV in → CSV out |
| Model service development | FastAPI + Pydantic | 5 | ✅ `app/main.py`, `app/model.py`, `app/schemas.py` |
| Front-end client | Streamlit | 2 | ✅ `frontend/app.py` |
| Packaging and containerization | Docker + Compose | 3 | ✅ `Dockerfile`, `Dockerfile.streamlit`, `docker-compose.yml` |
| CI/CD pipeline | GitHub Actions | 5 | ✅ `.github/workflows/ci.yml` |
| Hosting the application | HuggingFace Spaces | 3 | ✅ https://huggingface.co/spaces/amiraHadimi/story-point-estimator |
| Model serving runtime | Llama3SP (PEFT) | 3 | ✅ `app/model.py` |
| **Total** | | **27** | ✅ |

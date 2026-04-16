---
title: LLAMA3SP Story Point Estimator
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
---

# LLAMA3SP Story Point Estimator

This Space predicts Agile story points from Jira issues.

## Notes

- Uses LLAMA3SP model when possible
- Falls back to lightweight estimator if resources are limited

## Project context

This is a simplified deployment version of a full system built with:
- FastAPI (model serving)
- Streamlit (frontend)
- Docker Compose (multi-service architecture)
- CI/CD with GitHub Actions

Due to Hugging Face limitations (single container), this version merges everything into one app.
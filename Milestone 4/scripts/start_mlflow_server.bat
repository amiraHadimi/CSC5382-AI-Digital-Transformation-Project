@echo off
cd ..

mkdir mlartifacts
mlflow db upgrade sqlite:///mlflow.db

mlflow server ^
  --backend-store-uri sqlite:///mlflow.db ^
  --default-artifact-root ./mlartifacts ^
  --host 127.0.0.1 ^
  --port 5000
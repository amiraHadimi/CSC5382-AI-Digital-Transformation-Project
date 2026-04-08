import argparse
from pathlib import Path

import pandas as pd
import mlflow.pyfunc
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_model(model_uri: str):
    return mlflow.pyfunc.load_model(model_uri)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="storypoints_tfidf_lr")
    parser.add_argument("--version", default=None)
    parser.add_argument("--alias", default=None)
    args = parser.parse_args()

    m4_root = Path(__file__).resolve().parents[2]
    test_path = m4_root / ".." / "Milestone 3" / "data" / "processed" / "test.parquet"

    test_df = pd.read_parquet(test_path)

    X_test = test_df[["input_text"]]
    y_test = test_df["storypoints"].astype(float)

    if args.alias:
        model_uri = f"models:/{args.model_name}@{args.alias}"
    elif args.version:
        model_uri = f"models:/{args.model_name}/{args.version}"
    else:
        model_uri = f"models:/{args.model_name}/latest"

    model = load_model(model_uri)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    print("MODEL_URI:", model_uri)
    print("TEST_MAE:", mae)
    print("TEST_RMSE:", rmse)


if __name__ == "__main__":
    main()
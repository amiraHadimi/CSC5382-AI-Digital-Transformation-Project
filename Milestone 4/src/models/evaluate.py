import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow

def evaluate_model(model, vectorizer):
    test_df = pd.read_parquet("../Milestone 3/data/processed/test.parquet")

    X_test = test_df["input_text"]
    y_test = test_df["storypoints"]

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    mlflow.log_metric("test_MAE", mae)
    mlflow.log_metric("test_RMSE", rmse)

    print(f"Test MAE: {mae}")
    print(f"Test RMSE: {rmse}")

    return mae, rmse
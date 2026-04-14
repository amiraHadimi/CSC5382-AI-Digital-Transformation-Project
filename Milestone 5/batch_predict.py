import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"

INPUT_FILE = "sample_input.csv"
OUTPUT_FILE = "sample_output.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    predictions = []

    for _, row in df.iterrows():
        payload = {
            "title": str(row["title"]),
            "description": str(row["description"]),
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()

            predictions.append({
                "title": row["title"],
                "description": row["description"],
                "story_points": result["story_points"],
                "model_used": result["model_used"],
            })

        except Exception as e:
            predictions.append({
                "title": row["title"],
                "description": row["description"],
                "story_points": None,
                "model_used": f"error: {e}",
            })

    out_df = pd.DataFrame(predictions)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
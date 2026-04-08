import argparse
from mlflow import MlflowClient

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="storypoints_tfidf_lr")
    p.add_argument("--version", required=True)
    p.add_argument("--alias", required=True, help="candidate or champion")
    args = p.parse_args()

    client = MlflowClient()
    client.set_registered_model_alias(args.model_name, args.alias, args.version)
    print(f"Alias '{args.alias}' -> {args.model_name} v{args.version}")

if __name__ == "__main__":
    main()

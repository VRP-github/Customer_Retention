from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

load_dotenv()

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME   = os.getenv("MODEL_NAME", "Best_Churn_Predictor")
EXPERIMENT   = "Customer_Churn_Prediction"
METRIC       = "f1_score"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

print(f"\n{'='*55}")
print(f"  MLflow Tracking URI : {TRACKING_URI}")
print(f"  Experiment          : {EXPERIMENT}")
print(f"  Ranking metric      : {METRIC}")
print(f"{'='*55}\n")

experiment = client.get_experiment_by_name(EXPERIMENT)
if experiment is None:
    raise SystemExit(f"Experiment '{EXPERIMENT}' not found in {TRACKING_URI}")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="",
    order_by=[f"metrics.{METRIC} DESC"],
    max_results=5,
)

if not runs:
    raise SystemExit("No runs found in this experiment.")

print(f"{'Rank':<5} {'Run Name':<35} {'Model':<22} {METRIC.upper():<10} {'Run ID'}")
print("-" * 100)
for i, run in enumerate(runs, 1):
    name   = run.info.run_name or "—"
    model  = run.data.params.get("model_type", "—")
    metric = run.data.metrics.get(METRIC, 0)
    run_id = run.info.run_id
    marker = "  BEST" if i == 1 else ""
    print(f"{i:<5} {name:<35} {model:<22} {metric:<10.4f} {run_id}{marker}")

best_run    = runs[0]
best_run_id = best_run.info.run_id
best_metric = best_run.data.metrics.get(METRIC, 0)
best_name   = best_run.info.run_name or "—"

print(f"\nBest run  : {best_name}")
print(f"    Run ID   : {best_run_id}")
print(f"    {METRIC.upper():<10}: {best_metric:.4f}")

print(f"\n{'='*55}")
print(f"  Registered versions of '{MODEL_NAME}'")
print(f"{'='*55}")

try:
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if versions:
        for v in versions:
            print(f"  Version {v.version:<4} | Stage: {v.current_stage:<12} | Run: {v.run_id[:8]}…")
    else:
        print("  No versions registered yet.")
except Exception:
    print("  Model not registered yet.")

parser = argparse.ArgumentParser()
parser.add_argument("--promote", action="store_true",
                    help="Register the best run and promote it to Production")
args = parser.parse_args()

if args.promote:
    print(f"\nRegistering best run as '{MODEL_NAME}' ...")
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    print(f"Promoting version {mv.version} to Production ...")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"'{MODEL_NAME}' v{mv.version} is now in Production.")
    print(f"\n    Update your .env:")
    print(f"    MODEL_STAGE=Production")
else:
    print("\nDry run complete. Pass --promote to register and promote the best model.")
# src/analysis/save_training_logs.py
"""
Save training logs and results for the essential models used in the project.
Only saves results for models actually used: MLP, XGBoost Tuned, Baseline GBM.
"""
import os
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Models to train and log
ESSENTIAL_MODELS = {
    "baseline": {
        "command": [sys.executable, "-m", "src.models.baselines"],
        "description": "Baseline GBM (Champion Model)",
        "log_file": "training_logs/baseline_training.log"
    },
    "xgboost_tuned": {
        "command": [sys.executable, "-m", "src.models.tune_xgboost"],
        "description": "XGBoost with Optuna Optimization",
        "log_file": "training_logs/xgboost_tuned_training.log"
    },
    "mlp": {
        "command": [sys.executable, "-m", "src.models.mlp"],
        "description": "MLP with Feature Selection & Class Weights",
        "log_file": "training_logs/mlp_training.log"
    }
}

def ensure_dirs():
    """Ensure output directories exist."""
    os.makedirs("training_logs", exist_ok=True)
    os.makedirs("reports/training_summary", exist_ok=True)

def run_training(model_key: str) -> bool:
    """Run training for a specific model and capture output."""
    if model_key not in ESSENTIAL_MODELS:
        print(f"[ERROR] Unknown model: {model_key}")
        return False

    model_info = ESSENTIAL_MODELS[model_key]
    log_file = model_info["log_file"]
    description = model_info["description"]
    command = model_info["command"]

    print(f"\n{'='*80}")
    print(f"TRAINING: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(command)}")
    print(f"Log file: {log_file}")

    try:
        # Ensure log directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        # Add timestamp header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"\n{'='*80}\nTRAINING SESSION: {timestamp}\nModel: {description}\nCommand: {' '.join(command)}\n{'='*80}\n\n"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(header)

        # Run the training command and capture output
        print(f"[INFO] Starting training... (output will be saved to {log_file})")

        with open(log_file, "a", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=os.getcwd(),
                stdout=log,
                stderr=subprocess.STDOUT,  # Merge stderr with stdout
                text=True,
                timeout=1800  # 30 minute timeout
            )

        # Check result
        if result.returncode == 0:
            print(f"[SUCCESS] {model_key} training completed successfully")
            return True
        else:
            print(f"[ERROR] {model_key} training failed with return code {result.returncode}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n[ERROR] Training failed with return code {result.returncode}\n")
            return False

    except subprocess.TimeoutExpired:
        print(f"[ERROR] {model_key} training timed out after 30 minutes")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n\n[ERROR] Training timed out after 30 minutes\n")
        return False
    except Exception as e:
        print(f"[ERROR] {model_key} training failed with exception: {e}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n[ERROR] Training failed with exception: {e}\n")
        return False

def extract_key_metrics(log_file: str, model_key: str = None) -> dict:
    """Extract key metrics from training log or model_results.json."""
    metrics = {}

    try:
        # Try different encodings (Windows compatibility)
        content = None
        for encoding in ["utf-8", "cp1252", "latin-1"]:
            try:
                with open(log_file, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise UnicodeDecodeError("Could not decode file with any encoding")

        # Extract Test AUC from log
        import re
        auc_match = re.search(r"Classifier Test AUC:\s*([\d.]+)", content)
        if auc_match:
            metrics["test_auc"] = float(auc_match.group(1))

        # Extract Test MAE from log
        mae_match = re.search(r"Regressor Test MAE:\s*([\d.]+)", content)
        if mae_match:
            metrics["test_mae"] = float(mae_match.group(1))

        # For baseline, try to get metrics from model_results.json
        if model_key == "baseline" and not (auc_match and mae_match):
            try:
                import json
                with open("reports/metrics/model_results.json", "r", encoding="utf-8") as f:
                    model_results = json.load(f)

                if "baseline_gbm" in model_results:
                    test_metrics = model_results["baseline_gbm"].get("test", {})
                    clf_metrics = test_metrics.get("classification", {})
                    reg_metrics = test_metrics.get("regression", {})

                    if "roc_auc" in clf_metrics:
                        metrics["test_auc"] = clf_metrics["roc_auc"]
                    if "mae" in reg_metrics:
                        metrics["test_mae"] = reg_metrics["mae"]
            except Exception as e:
                print(f"[WARN] Could not load baseline metrics from JSON: {e}")

        # Extract training completion status
        if ("DONE:" in content or
            "XGBoost Training Complete" in content or
            "XGBoost Optimization Complete" in content or
            "Training Complete" in content):
            metrics["completed"] = True
        else:
            metrics["completed"] = False

    except Exception as e:
        print(f"[WARN] Could not extract metrics from {log_file}: {e}")
        metrics["error"] = str(e)

    return metrics

def create_training_summary():
    """Create a summary of all training results."""
    summary = {
        "training_session": datetime.now().isoformat(),
        "models_trained": [],
        "leaderboard": []
    }

    for model_key, model_info in ESSENTIAL_MODELS.items():
        log_file = model_info["log_file"]
        if os.path.exists(log_file):
            metrics = extract_key_metrics(log_file, model_key)

            model_result = {
                "model": model_key,
                "description": model_info["description"],
                "log_file": log_file,
                "file_size_kb": round(os.path.getsize(log_file) / 1024, 1),
                "metrics": metrics
            }

            summary["models_trained"].append(model_result)

            # Add to leaderboard if completed successfully
            if metrics.get("completed", False):
                auc = metrics.get("test_auc", 0)
                if auc > 0:
                    summary["leaderboard"].append({
                        "model": model_key,
                        "test_auc": auc,
                        "test_mae": metrics.get("test_mae", 0)
                    })

    # Sort leaderboard by AUC
    summary["leaderboard"].sort(key=lambda x: x["test_auc"], reverse=True)

    # Save summary
    summary_file = "reports/training_summary/training_results.json"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Training summary saved to: {summary_file}")

    # Print leaderboard
    print(f"\n{'='*60}")
    print("TRAINING RESULTS LEADERBOARD")
    print(f"{'='*60}")

    if summary["leaderboard"]:
        print("<10")
        print("-" * 50)
        for i, entry in enumerate(summary["leaderboard"], 1):
            auc = entry["test_auc"]
            mae = entry["test_mae"]
            print("10")
    else:
        print("No completed trainings found.")

    return summary

def main():
    """Main function to run all essential model trainings."""
    print("TRAINING LOGS SAVER - Essential Models Only")
    print("=" * 60)
    print(f"Will train {len(ESSENTIAL_MODELS)} essential models:")
    for key, info in ESSENTIAL_MODELS.items():
        print(f"  - {key}: {info['description']}")
    print("=" * 60)

    ensure_dirs()

    results = {}
    for model_key in ESSENTIAL_MODELS.keys():
        success = run_training(model_key)
        results[model_key] = success

    # Create summary
    summary = create_training_summary()

    # Print final status
    print(f"\n{'='*60}")
    print("TRAINING COMPLETION STATUS")
    print(f"{'='*60}")

    all_success = True
    for model_key, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print("15")

        if not success:
            all_success = False

    if all_success:
        print(f"\n🎉 All essential models trained successfully!")
        print("📁 Training logs saved to: training_logs/")
        print("📊 Training summary saved to: reports/training_summary/training_results.json")
    else:
        print(f"\n⚠️  Some models failed to train. Check training_logs/ for details.")

    return all_success

if __name__ == "__main__":
    main()

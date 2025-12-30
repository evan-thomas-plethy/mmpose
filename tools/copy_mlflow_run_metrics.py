import mlflow
import os
import tempfile
import shutil

# Connect to MLflow
mlflow.set_tracking_uri("http://35.165.139.156:5000")

# Source run info
source_experiment = "rtmpose-s_heel_slides_lr_mult"
source_run_name = "backbone_lr_mult_0.1_BN_frozen_True"  # or use run_id directly

# Target experiment
target_experiment = "rtmpose-s_heel_slides_rotation"
target_run_name = "rotate_20"

# Get source run
source_exp = mlflow.get_experiment_by_name(source_experiment)
runs = mlflow.search_runs(
    experiment_ids=[source_exp.experiment_id],
    filter_string=f"run_name = '{source_run_name}'"
)
source_run_id = runs.iloc[0].run_id

# Get source metrics
client = mlflow.tracking.MlflowClient()
source_run = client.get_run(source_run_id)

# Create new run in target experiment
mlflow.set_experiment(target_experiment)
with mlflow.start_run(run_name=target_run_name):
    # Copy parameters
    print("Copying parameters...")
    for key, value in source_run.data.params.items():
        mlflow.log_param(key, value)
    
    # Copy metrics (with their step values)
    print("Copying metrics...")
    for key in source_run.data.metrics.keys():
        metric_history = client.get_metric_history(source_run_id, key)
        for m in metric_history:
            mlflow.log_metric(key, m.value, step=m.step)
    
    # Copy artifacts
    print("Copying artifacts...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download all artifacts from source run
        local_path = client.download_artifacts(source_run_id, "", tmp_dir)
        
        # Log each artifact/directory to the new run
        for item in os.listdir(local_path):
            item_path = os.path.join(local_path, item)
            if os.path.isdir(item_path):
                mlflow.log_artifacts(item_path, item)
            else:
                mlflow.log_artifact(item_path)
        
        print(f"  Downloaded and uploaded artifacts from {local_path}")
    
    print(f"Copied run to {target_experiment}/{target_run_name}")
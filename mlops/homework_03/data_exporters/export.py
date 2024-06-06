import mlflow
import tempfile
import pickle
from pathlib import Path

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    mlflow.set_tracking_uri("http://mlflow:5000")
    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
    mlflow.set_experiment("mlops-zoomcamp-2024-homework-03")
    model, dv = data

    # Why is this needed? Otherwise start_run complains about a run already running
    mlflow.end_run()

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "models")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmpfile = Path(tmp_dir, "dict_vectorizer.dat")
            with tmpfile.open('wb') as f:
                pickle.dump(dv, f)
            mlflow.log_artifact(str(tmpfile))

    return model, dv
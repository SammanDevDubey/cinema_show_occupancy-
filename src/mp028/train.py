
"""MP028 model training, evaluation and v1 registration."""

import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from mp028.config import load_params, REPO_ROOT
from mp028.data_clean import clean_train
from mp028.modeling import build_pipeline


# Types required when logging our own sklearn pipelines.
TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.compose._column_transformer._RemainderColsList",
    "sklearn.tree._tree.Tree",
]


def configure_mlflow(params):
    """Configure MLflow tracking and experiment."""
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"])


def main():
    """Train and compare all model configurations."""

    # 1. Load configuration
    params = load_params()

    features = params["data"]["features"]
    target = params["data"]["target"]

    random_state = params["model"]["random_state"]
    test_size = params["model"]["test_size"]

    # 2. Load and clean training data
    train_path = REPO_ROOT / params["paths"]["train_raw"]

    raw_df = pd.read_csv(train_path)
    df = clean_train(raw_df)

    X = df[features]
    y = df[target]

    print("Dataset loaded successfully.")
    print("Features:", X.shape)
    print("Target:", y.shape)

    # 3. Train-validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    print("\nTraining rows:", len(X_train))
    print("Validation rows:", len(X_valid))

    # 4. Configure MLflow
    configure_mlflow(params)

    results = []
    configurations = params["model"]["grid"]

    # 5. Train all configurations
    for index, config in enumerate(configurations, start=1):

        model_name = config["name"]
        hyperparams = config["hyperparams"]

        run_name = f"{model_name}_{index}"

        print("\n" + "=" * 55)
        print(f"Configuration {index}/{len(configurations)}")
        print("Model:", model_name)
        print("Hyperparameters:", hyperparams)

        pipeline = build_pipeline(
            model_name=model_name,
            hyperparams=hyperparams,
            random_state=random_state,
        )

        with mlflow.start_run(run_name=run_name):

            # Fit model
            pipeline.fit(X_train, y_train)

            # Validation predictions
            predictions = pipeline.predict(X_valid)

            # Calculate metrics
            metrics = {
                "r2": float(r2_score(y_valid, predictions)),
                "mae": float(mean_absolute_error(y_valid, predictions)),
                "rmse": float(
                    mean_squared_error(y_valid, predictions) ** 0.5
                ),
            }

            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(hyperparams)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("test_size", test_size)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                skops_trusted_types=TRUSTED_TYPES,
            )

            results.append({
                "configuration": index,
                "model": model_name,
                "hyperparams": str(hyperparams),
                **metrics,
            })

            print(f"R2 Score : {metrics['r2']:.4f}")
            print(f"MAE      : {metrics['mae']:.4f}")
            print(f"RMSE     : {metrics['rmse']:.4f}")

    # 6. Compare results
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="r2",
        ascending=False,
    )

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)

    print(results_df.to_string(index=False))

    # 7. Display best configuration
    best = results_df.iloc[0]

    print("\nBEST CONFIGURATION")
    print("Model:", best["model"])
    print("Hyperparameters:", best["hyperparams"])
    print("R2:", round(best["r2"], 4))
    print("MAE:", round(best["mae"], 4))
    print("RMSE:", round(best["rmse"], 4))

    print("\nAll configurations trained and logged successfully.")


def register_v1():
    """Train the selected model on all cleaned data and register v1."""

    params = load_params()

    # 1. Read best configuration
    config_path = REPO_ROOT / "reports" / "best_config.json"

    with open(config_path, "r", encoding="utf-8") as file:
        best_config = json.load(file)

    model_name = best_config["model"]
    hyperparams = best_config["hyperparams"]
    random_state = best_config["random_state"]

    # 2. Load complete cleaned training data
    train_path = REPO_ROOT / params["paths"]["train_raw"]

    raw_df = pd.read_csv(train_path)
    df = clean_train(raw_df)

    X = df[params["data"]["features"]]
    y = df[params["data"]["target"]]

    print("\nStarting v1 model training...")
    print("Model:", model_name)
    print("Hyperparameters:", hyperparams)
    print("Training rows:", len(df))

    # 3. Build and train winning pipeline on all 1000 rows
    pipeline = build_pipeline(
        model_name=model_name,
        hyperparams=hyperparams,
        random_state=random_state,
    )

    pipeline.fit(X, y)

    # 4. Save local Joblib model
    model_path = REPO_ROOT / params["paths"]["v1_model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)

    print("\nLocal v1 model saved:", model_path)

    # 5. Configure MLflow
    configure_mlflow(params)

    registered_name = params["mlflow"]["registered_model"]

    # Check whether v1 has already been registered.
    try:
        existing = mlflow.MlflowClient().get_registered_model(
            registered_name
        )
        if existing.latest_versions:
            raise RuntimeError(
                f"{registered_name} already has registered versions. "
                "Check MLflow before registering again."
            )
    except mlflow.exceptions.MlflowException:
        # Model does not exist yet; it can be created below.
        pass

    # 6. Log full-data model
    with mlflow.start_run(run_name="v1_ridge_full_train"):

        mlflow.log_param("model_name", model_name)
        mlflow.log_params(hyperparams)
        mlflow.log_param("training_rows", len(df))
        mlflow.log_param("model_version", "v1")

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            skops_trusted_types=TRUSTED_TYPES,
        )

        model_uri = model_info.model_uri

    # 7. Register model
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=registered_name,
    )

    print("\n" + "=" * 55)
    print("V1 MODEL REGISTERED SUCCESSFULLY")
    print("=" * 55)
    print("Registered name:", registered_name)
    print("Version:", registered_model.version)
    print("Local file:", model_path)
    print("MLflow URI:", model_uri)


if __name__ == "__main__":
    main()


"""Train and compare MP028 cinema occupancy models."""

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


def main():
    # 1. Load configuration
    params = load_params()

    features = params["data"]["features"]
    target = params["data"]["target"]
    random_state = params["model"]["random_state"]
    test_size = params["model"]["test_size"]

    # 2. Load and clean dataset
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
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"])

    results = []

    # Types required by the locally trained sklearn pipelines.
    trusted_types = [
        "numpy.dtype",
        "sklearn.compose._column_transformer._RemainderColsList",
        "sklearn.tree._tree.Tree",
    ]

    # 5. Train all configurations
    for index, config in enumerate(
        params["model"]["grid"], start=1
    ):
        model_name = config["name"]
        hyperparams = config["hyperparams"]
        run_name = f"{model_name}_{index}"

        print("\n" + "=" * 55)
        print(f"Configuration {index}/{len(params['model']['grid'])}")
        print("Model:", model_name)
        print("Hyperparameters:", hyperparams)

        pipeline = build_pipeline(
            model_name=model_name,
            hyperparams=hyperparams,
            random_state=random_state,
        )

        with mlflow.start_run(run_name=run_name):

            # Train
            pipeline.fit(X_train, y_train)

            # Predict
            predictions = pipeline.predict(X_valid)

            # Evaluate
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
                skops_trusted_types=trusted_types,
            )

            # Save comparison results
            results.append({
                "configuration": index,
                "model": model_name,
                "hyperparams": str(hyperparams),
                **metrics,
            })

            print(f"R2 Score : {metrics['r2']:.4f}")
            print(f"MAE      : {metrics['mae']:.4f}")
            print(f"RMSE     : {metrics['rmse']:.4f}")

    # 6. Compare configurations
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


if __name__ == "__main__":
    main()

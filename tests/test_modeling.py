
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from mp028.config import load_params
from mp028.modeling import (
    build_preprocessor,
    build_model,
    build_pipeline,
    clone_pipeline,
)


def sample_data():
    params = load_params()
    features = params["data"]["features"]

    return pd.DataFrame({
        "ticket_price_rs": [150, 200, 180, 220],
        "days_since_release": [10, 20, 15, 25],
        "star_rating": [4.1, 3.8, 4.5, 4.0],
        "show_hour": [10, 14, 18, 21],
        "screen_seats": [100, 120, 90, 150],
        "distance_to_metro_km": [1.2, 2.0, 0.8, 3.0],
        "language": ["Hindi", "English", "Hindi", "English"],
        "location": ["Mall", "City Centre", "Mall", "Highway"],
    })[features]


def test_build_preprocessor():
    preprocessor = build_preprocessor()
    assert preprocessor is not None


@pytest.mark.parametrize(
    "name, expected_class",
    [
        ("ridge", Ridge),
        ("random_forest", RandomForestRegressor),
        ("gradient_boosting", GradientBoostingRegressor),
    ],
)
def test_build_model(name, expected_class):
    model = build_model(name)
    assert isinstance(model, expected_class)


def test_unknown_model_raises_error():
    with pytest.raises(ValueError):
        build_model("unknown_model")


def test_random_forest_random_state():
    model = build_model("random_forest", random_state=123)
    assert model.random_state == 123


def test_build_pipeline():
    pipeline = build_pipeline("ridge", {"alpha": 1.0})

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_pipeline_can_fit_and_predict():
    X = sample_data()
    y = np.array([45, 60, 55, 70])

    pipeline = build_pipeline("ridge")
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)
    assert np.isfinite(predictions).all()


def test_clone_pipeline_is_unfitted():
    X = sample_data()
    y = np.array([45, 60, 55, 70])

    pipeline = build_pipeline("ridge")
    pipeline.fit(X, y)

    cloned = clone_pipeline(pipeline)

    assert isinstance(cloned, Pipeline)
    assert cloned is not pipeline
    assert not hasattr(cloned, "predict") or not hasattr(
        cloned.named_steps["model"], "coef_"
    )

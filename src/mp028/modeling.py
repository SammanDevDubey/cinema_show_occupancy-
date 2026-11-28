
"""Model preprocessing and pipeline construction for MP028."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

from mp028.config import load_params


def build_preprocessor():
    """Build preprocessing for numerical and categorical features."""

    params = load_params()
    features = params["data"]["features"]

    numeric_features = [
        "ticket_price_rs",
        "days_since_release",
        "star_rating",
        "show_hour",
        "screen_seats",
        "distance_to_metro_km",
    ]

    categorical_features = ["language", "location"]

    # Ensure feature configuration matches the expected columns.
    if set(features) != set(numeric_features + categorical_features):
        raise ValueError("Feature configuration does not match expected features.")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return preprocessor


def build_model(model_name, hyperparams=None, random_state=None):
    """Create a regression model using the requested configuration."""

    hyperparams = hyperparams or {}

    params = load_params()
    if random_state is None:
        random_state = params["model"]["random_state"]

    if model_name == "ridge":
        return Ridge(**hyperparams)

    if model_name == "random_forest":
        return RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
            **hyperparams,
        )

    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(
            random_state=random_state,
            **hyperparams,
        )

    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(model_name, hyperparams=None, random_state=None):
    """Combine preprocessing and model into one sklearn pipeline."""

    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", build_model(model_name, hyperparams, random_state)),
    ])


def clone_pipeline(pipeline):
    """Return an unfitted copy of a pipeline."""

    return clone(pipeline)

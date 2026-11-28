
from mp028 import train


def test_trusted_types_are_defined():
    assert "numpy.dtype" in train.TRUSTED_TYPES
    assert "sklearn.tree._tree.Tree" in train.TRUSTED_TYPES


def test_configure_mlflow(monkeypatch):
    calls = {}

    def fake_set_tracking_uri(uri):
        calls["uri"] = uri

    def fake_set_experiment(name):
        calls["experiment"] = name

    monkeypatch.setattr(
        train.mlflow, "set_tracking_uri", fake_set_tracking_uri
    )
    monkeypatch.setattr(
        train.mlflow, "set_experiment", fake_set_experiment
    )

    params = {
        "mlflow": {
            "tracking_uri": "sqlite:///test.db",
            "experiment": "test-experiment",
        }
    }

    train.configure_mlflow(params)

    assert calls["uri"] == "sqlite:///test.db"
    assert calls["experiment"] == "test-experiment"


def test_train_module_has_required_functions():
    assert callable(train.main)
    assert callable(train.register_v1)
    assert callable(train.configure_mlflow)

"""
XGBoost hyperparameter optimization with neuropt.

    neuropt run examples/train_xgboost.py --backend claude

Just define the model and a training function — neuropt figures out
what to search over (asks the LLM for ranges, or uses sensible defaults).
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(verbosity=0)


def train_fn(config):
    m = config["model"]
    m.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    results = m.evals_result()
    return {
        "score": results["validation_1"]["logloss"][-1],
        "train_losses": results["validation_0"]["logloss"],
        "val_losses": results["validation_1"]["logloss"],
    }

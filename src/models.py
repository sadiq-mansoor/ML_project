"""
Logistic Regression model factory.

sklearn 1.8 API: penalty= is deprecated. Use l1_ratio + C directly:
  l1_ratio=0.0  → pure L2 (Ridge)
  l1_ratio=1.0  → pure L1 (Lasso)
  0 < l1_ratio < 1 → ElasticNet
  C=1e9         → effectively no regularization (penalty=None equivalent)

solver must match:
  lbfgs  → L2 only (l1_ratio=0)
  saga   → L1, ElasticNet (l1_ratio > 0)
"""
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
C_VALUES = [0.01, 0.1, 1, 10]
L1_RATIOS = [0.3, 0.5, 0.7]


def make_baseline():
    """No regularization baseline — C=1e9 ≈ penalty=None."""
    return LogisticRegression(
        l1_ratio=0.0, C=1e9, solver='lbfgs',
        max_iter=5000, random_state=RANDOM_STATE
    )


def make_underfit():
    """Strong L2 regularization — high bias, low variance."""
    return LogisticRegression(
        l1_ratio=0.0, C=0.0001, solver='lbfgs',
        max_iter=5000, random_state=RANDOM_STATE
    )


def make_regularized(reg_type: str, C: float, l1_ratio: float = 0.5):
    """
    reg_type: 'L1' | 'L2' | 'ElasticNet' | 'None'
    Uses sklearn 1.8 l1_ratio API — no penalty= kwarg to avoid deprecation warnings.
    """
    if reg_type == 'L1':
        # l1_ratio=1.0 → pure L1; saga required
        return LogisticRegression(
            l1_ratio=1.0, C=C, solver='saga',
            max_iter=10000, random_state=RANDOM_STATE
        )
    elif reg_type == 'L2':
        # l1_ratio=0.0 → pure L2; lbfgs is fastest for L2
        return LogisticRegression(
            l1_ratio=0.0, C=C, solver='lbfgs',
            max_iter=5000, random_state=RANDOM_STATE
        )
    elif reg_type == 'ElasticNet':
        # 0 < l1_ratio < 1 → ElasticNet; saga required
        return LogisticRegression(
            l1_ratio=l1_ratio, C=C, solver='saga',
            max_iter=10000, random_state=RANDOM_STATE
        )
    elif reg_type == 'None':
        return LogisticRegression(
            l1_ratio=0.0, C=1e9, solver='lbfgs',
            max_iter=5000, random_state=RANDOM_STATE
        )
    else:
        raise ValueError(f"Unknown reg_type: {reg_type}")

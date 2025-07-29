import math
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from scipy.stats import randint, uniform, loguniform

def unwrap_estimator(estimator):
    """Unwrap multi-output or search wrappers to get the base estimator."""
    while hasattr(estimator, 'estimator'):
        estimator = estimator.estimator
    return estimator

def estimate_search_space_size(params):
    """Estimate the size of the parameter search space for auto-tuning learners."""
    total_combinations = 1
    for param, values in params.items():
        if isinstance(values, (list, tuple)):
            total_combinations *= len(values)
        elif isinstance(values, (randint, uniform, loguniform)):
            # Approximate continuous distributions with a reasonable number of discrete points
            total_combinations *= 10  # Assume ~10 effective values for continuous ranges
        else:
            total_combinations *= 1  # Single value, no contribution
    return total_combinations

def get_complexity_score(estimator: BaseEstimator) -> float:
    """
    Computes a generic complexity score for sklearn estimators, including auto-tuning learners.
    Higher score indicates higher model complexity, correlating with higher overfitting risk.
    
    - OLS (LinearRegression) is baseline 1.0
    - Regularized linear models (e.g., Ridge) have scores <= 1.0 based on regularization strength
    - Tree-based models have scores > 1.0 based on depth, number of trees, etc.
    - Auto-tuning learners (GridSearchCV, etc.) multiply base estimator score by search space factor
    - Normalized roughly: default RandomForest ~5-10, default XGBoost ~3-5, GridSearchCV ~10x base
    
    Usage: Call before or after fitting to gauge overfitting risk in the Capstone Simulator.
    Can adjust performance metrics (e.g., complexity-adjusted return = return / score).
    """
    # Handle auto-tuning learners
    if isinstance(estimator, (GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV)):
        base_estimator = unwrap_estimator(estimator)
        base_score = get_complexity_score(base_estimator)  # Recursively get base score
        
        # Estimate search space complexity
        param_grid = estimator.param_grid if hasattr(estimator, 'param_grid') else estimator.param_distributions
        search_space_size = estimate_search_space_size(param_grid)
        
        # For RandomizedSearchCV and HalvingRandomSearchCV, cap the effective size
        if isinstance(estimator, (RandomizedSearchCV, HalvingRandomSearchCV)):
            n_iter = estimator.n_iter if hasattr(estimator, 'n_iter') else 10
            search_space_size = min(search_space_size, n_iter)
        
        # Halving searches reduce effective trials, so adjust
        if isinstance(estimator, (HalvingGridSearchCV, HalvingRandomSearchCV)):
            factor = estimator.factor if hasattr(estimator, 'factor') else 3
            search_space_size = max(1, search_space_size // factor)
        
        # Multiply base score by log of search space size to reflect overfitting risk
        return base_score * (1.0 + math.log1p(search_space_size) / math.log(10))  # Log10 for scaling

    # Handle base estimators
    estimator = unwrap_estimator(estimator)
    params = estimator.get_params()
    cls_name = type(estimator).__name__
    
    if 'LinearRegression' in cls_name:
        return 1.0
    
    elif 'Ridge' in cls_name:
        alpha = params.get('alpha', 1.0)
        return 1.0 / (1.0 + alpha)  # Decreases with stronger regularization, e.g., alpha=1 -> 0.5
    
    elif 'Lasso' in cls_name:
        alpha = params.get('alpha', 1.0)
        return 1.0 / (1.0 + alpha * 10)  # Stronger penalty for Lasso sparsity
    
    elif 'RandomForest' in cls_name:
        max_depth = params.get('max_depth')
        if max_depth is None:
            max_depth = 20  # Assumption for unbounded trees
        n_estimators = params.get('n_estimators', 100)
        max_features = params.get('max_features', 1.0)
        if isinstance(max_features, str):
            if max_features in ['auto', 'sqrt', 'log2']:
                max_features = 0.33  # Approximate fraction
            else:
                max_features = 1.0
        elif max_features is None:
            max_features = 1.0
        effective_complexity = n_estimators * max_depth * max_features
        return 1.0 + effective_complexity / 200.0  # Default ~1 + (100*20*1)/200 = 11.0

    elif 'XGB' in cls_name:  # For XGBoost
        max_depth = params.get('max_depth', 6)
        n_estimators = params.get('n_estimators', 100)
        effective_complexity = n_estimators * (2 ** max_depth)
        return 1.0 + math.log(1 + effective_complexity) / math.log(2) / 10.0  # Default ~2.26

    elif 'DecisionTree' in cls_name:
        max_depth = params.get('max_depth')
        if max_depth is None:
            max_depth = 20
        return 1.0 + max_depth / 5.0

    elif 'SVC' in cls_name or 'SVR' in cls_name:
        kernel = params.get('kernel', 'rbf')
        if kernel == 'linear':
            return 1.2
        else:
            C = params.get('C', 1.0)
            gamma = params.get('gamma', 'scale')
            if gamma == 'scale':
                gamma = 1.0
            elif gamma == 'auto':
                gamma = 0.1
            return 2.0 + math.log(1 + C * gamma)

    elif 'KNeighbors' in cls_name:
        n_neighbors = params.get('n_neighbors', 5)
        return 1.0 + 10.0 / n_neighbors

    elif 'MLP' in cls_name:
        hidden = params.get('hidden_layer_sizes', (100,))
        if isinstance(hidden, int):
            hidden = (hidden,)
        total_neurons = sum(hidden)
        n_layers = len(hidden)
        return 1.0 + (n_layers * total_neurons) / 50.0  # Default ~3.0

    else:
        return 2.0  # Default for unknown estimators
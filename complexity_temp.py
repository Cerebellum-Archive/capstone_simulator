import math
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from scipy.stats import randint, uniform, loguniform
from numpy.linalg import svd, pinv # Added for SVD in Ridge EDF calculation

def unwrap_estimator(estimator):
    """
    Unwrap multi-output, search, or ensemble wrappers to get the base estimator.
    Handles nested wrappers and fitted search models.
    """
    # Unwrap search CV models to their best_estimator_ if fitted
    if hasattr(estimator, 'best_estimator_') and estimator.best_estimator_ is not None:
        estimator = estimator.best_estimator_
    
    # Unwrap multi-output or other generic wrappers
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

def get_complexity_score(estimator: BaseEstimator, X=None, y=None) -> float:
    """
    Computes a generic complexity score for sklearn estimators, including auto-tuning learners.
    Higher score indicates higher model complexity, correlating with higher overfitting risk,
    especially for time-series generalization.
    
    - OLS (LinearRegression) is baseline 1.0.
    - Regularized linear models (e.g., Ridge) have scores <= 1.0 based on regularization strength,
      as they tend to generalize better on time-series data (e.g., reduced variance in forecasts).
    - Tree-based models have scores > 1.0 based on depth, number of trees, etc., reflecting
      higher risk of overfitting to noise in sequential data.
    - Auto-tuning learners (GridSearchCV, etc.) multiply base estimator score by a search space factor
      to account for increased risk from multiple fits.
    - Normalized roughly: default RandomForest ~5-10, default XGBoost ~3-5, GridSearchCV ~10x base.
    - When X and y are provided (assuming fitted estimator), uses data-dependent effective degrees of
      freedom where possible for more accurate assessment (based on academic methods like trace of
      hat matrix for linear models).
    - Inspired by academic research on effective degrees of freedom (e.g., Ye 1998, Hastie et al. 2009)
      and model complexity measures for generalization (e.g., AIC/BIC for penalizing complexity,
      surveys on deep learning complexity).
    
    Usage: Call before or after fitting to gauge overfitting risk in the Capstone Simulator.
    Can adjust performance metrics (e.g., complexity-adjusted return = return / score).
    """
    # Check if the estimator is fitted. This is a heuristic.
    # A more robust check might involve sklearn.utils.validation.check_is_fitted
    # but that would require fitting the model if not already.
    is_fitted = hasattr(estimator, 'coef_') or \
                hasattr(estimator, 'feature_importances_') or \
                hasattr(estimator, 'n_features_in_') or \
                (hasattr(estimator, 'predict') and hasattr(estimator, '_is_fitted')) # _is_fitted is more internal

    # Handle auto-tuning learners (ee.g., GridSearchCV)
    if isinstance(estimator, (GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV)):
        # Recursively get the complexity score of the best base estimator
        # Pass X and y down to allow data-dependent calculation for the best_estimator_
        base_estimator = unwrap_estimator(estimator) # This will get the best_estimator_ if fitted
        base_score = get_complexity_score(base_estimator, X, y)
        
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
            # Reduce search space size by factor, ensuring it's at least 1
            search_space_size = max(1, search_space_size // factor)
        
        # Multiply base score by log of search space size to reflect overfitting risk
        # Using log1p for robustness with small search_space_size
        return base_score * (1.0 + math.log1p(search_space_size) / math.log(10)) # Log10 for scaling ~10x for large grids

    # Unwrap the estimator again if it was a search CV that wasn't fitted,
    # or if it's an ensemble that wasn't handled by best_estimator_ above.
    estimator = unwrap_estimator(estimator)
    params = estimator.get_params()
    cls_name = type(estimator).__name__
    
    # --- Data-dependent mode if X and y provided and estimator is fitted ---
    if X is not None and y is not None and is_fitted:
        n, p = X.shape # n_samples, n_features
        # Ensure p is at least 1 to avoid division by zero
        if p == 0:
            return 2.0 # Fallback for no features

        # Normalization base for effective degrees of freedom (e.g., p+1 for intercept)
        # This helps normalize scores across models to be comparable to OLS baseline of 1.0
        normalization_base = p + (1 if params.get('fit_intercept', True) else 0)
        if normalization_base == 0: # Avoid division by zero if somehow no features and no intercept
            normalization_base = 1

        try: # Use try-except for data-dependent calculations as they can fail if model state is unexpected
            if 'LinearRegression' in cls_name:
                # For OLS, effective df = p + 1 (assuming fit_intercept=True)
                # Score is baseline 1.0, as per problem statement
                return 1.0
            
            elif 'Ridge' in cls_name:
                alpha = params.get('alpha', 1.0)
                # Compute effective df using SVD (Hastie et al., Elements of Statistical Learning)
                # X_centered is used for effective df calculation in Ridge
                X_centered = X - np.mean(X, axis=0) if params.get('fit_intercept', True) else X
                _, s, _ = svd(X_centered, full_matrices=False)
                # Ensure alpha is not zero to avoid division by zero, and add a small epsilon if needed
                d = s**2 / (s**2 + alpha)
                effective_df = np.sum(d)
                # Normalize to OLS baseline. Score <1 for regularized models.
                return effective_df / normalization_base
            
            elif 'Lasso' in cls_name:
                # Approximate effective df as number of non-zero coefficients (Zou et al., 2007)
                coef = estimator.coef_
                # Handle multi-output Lasso if coef_ is 2D
                if coef.ndim > 1:
                    non_zero = np.sum(np.any(np.abs(coef) > 1e-5, axis=0))
                else:
                    non_zero = np.sum(np.abs(coef) > 1e-5)
                
                if params.get('fit_intercept', True):
                    non_zero += 1
                # Normalize. Score <1 if sparse.
                return non_zero / normalization_base
            
            elif 'DecisionTree' in cls_name:
                # Approximate effective df as number of leaves (loaded from fitted tree)
                n_leaves = estimator.tree_.n_leaves
                # Normalized, higher for deeper trees. Scale by number of samples to avoid huge scores for small trees.
                return n_leaves / (n / 10.0 + normalization_base) # Added n/10.0 to scale with dataset size
            
            elif 'RandomForest' in cls_name:
                # Average number of leaves across trees (Mentch & Zhou, 2020 on RF dof)
                if hasattr(estimator, 'estimators_') and estimator.estimators_:
                    n_leaves_avg = np.mean([tree.tree_.n_leaves for tree in estimator.estimators_])
                else: # Fallback if estimators_ not available or empty (e.g., not fully fitted)
                    n_leaves_avg = params.get('max_leaf_nodes', 100) # Heuristic
                
                # Complexity scales with n_estimators, avg_leaves, and max_features
                n_estimators = params.get('n_estimators', 100)
                max_features = params.get('max_features', 1.0)
                if isinstance(max_features, str):
                    if max_features in ['auto', 'sqrt', 'log2']:
                        max_features = math.sqrt(p) / p if p > 0 else 1.0 # Approximate as sqrt(p)/p
                    else:
                        max_features = 1.0 # Assume all features if 'None' or other string
                elif max_features is None:
                    max_features = 1.0
                
                # Heuristic for effective complexity for tree ensembles
                effective_complexity = n_estimators * n_leaves_avg * max_features
                # Normalize based on dof studies, scaling with input dimensions
                return effective_complexity / (n / 100.0 + normalization_base) # Scale with dataset size
            
            elif 'XGB' in cls_name or 'LGBM' in cls_name or 'CatBoost' in cls_name: # Handle common gradient boosting models
                # Approximate using number of leaves in boosted trees
                # This requires the model to be fully fitted and have access to its internal tree structure
                try:
                    if 'XGB' in cls_name:
                        booster = estimator.get_booster()
                        trees_dump = booster.get_dump()
                        # Sum of leaves across all trees
                        n_leaves_total = sum(tree.count('leaf') for tree in trees_dump)
                    elif 'LGBM' in cls_name:
                        # LightGBM trees are accessible via _tree_to_json
                        n_leaves_total = sum(tree['num_leaves'] for tree in estimator.booster_.dump_as_json())
                    elif 'CatBoost' in cls_name:
                        # CatBoost structure is more complex, can approximate via model_size or depth
                        # Fallback to heuristic for CatBoost if direct leaf count is hard
                        n_leaves_total = params.get('iterations', 100) * (2 ** params.get('depth', 6)) # Heuristic for CatBoost
                except Exception:
                    # Fallback to heuristic if direct leaf count from booster fails
                    n_leaves_total = params.get('n_estimators', 100) * (2 ** params.get('max_depth', 6))

                effective_complexity = n_leaves_total
                # Normalize based on complexity surveys and input dimensions
                return effective_complexity / (n / 100.0 + normalization_base)
            
            elif 'SVC' in cls_name or 'SVR' in cls_name:
                # Effective df approximated by number of support vectors (Dietrich et al., 1999)
                if hasattr(estimator, 'support_vectors_'):
                    n_sv = len(estimator.support_vectors_)
                elif hasattr(estimator, 'n_support_'): # For multi-class SVC
                    n_sv = np.sum(estimator.n_support_)
                else: # Fallback if support vectors not directly accessible
                    n_sv = n / 2 # Heuristic, assume half samples are SVs
                
                # Normalize by features, often higher for non-linear kernels
                return n_sv / (n / 10.0 + normalization_base) # Scale with dataset size
            
            elif 'KNeighbors' in cls_name:
                # Effective df ~ n / k (Hastie et al.)
                k = params.get('n_neighbors', 5)
                # Ensure k is not zero or too small
                if k < 1: k = 1
                return (n / k) / normalization_base # Normalized
            
            elif 'MLP' in cls_name:
                # Effective df approximated by total parameters, reduced for regularization
                hidden = params.get('hidden_layer_sizes', (100,))
                if isinstance(hidden, int):
                    hidden = (hidden,)
                
                # Calculate total parameters (weights + biases)
                total_params = 0
                layers = [p] + list(hidden) + [y.shape[1] if y.ndim > 1 else 1] # Input, hidden, output
                for i in range(len(layers) - 1):
                    total_params += layers[i] * layers[i + 1] + layers[i + 1] # Weights + biases
                
                alpha = params.get('alpha', 0.0001) # L2 penalty
                effective_df = total_params / (1 + alpha) # Reduced by regularization
                return effective_df / (n / 100.0 + normalization_base) # Normalized based on complexity surveys (Hu et al., 2021)
            
        except Exception as e:
            # Fallback to parameter-based heuristics if data-dependent calculation fails
            print(f"Warning: Data-dependent complexity calculation failed for {cls_name}. Error: {e}. Falling back to parameter-based heuristic.")
            pass # Continue to parameter-based fallback

    # --- Fall back to parameter-based heuristics if no data or not applicable ---
    # These heuristics are less precise but provide a score without fitted model or data
    
    # Normalization base for parameter-based heuristics.
    # If X is not provided, we can't use X.shape[1]. Use a default or infer from params if possible.
    # For a general heuristic, a fixed base or a base derived from typical feature counts might be used.
    # Here, we keep it simple, assuming a relative score.
    
    if 'LinearRegression' in cls_name:
        return 1.0 # Baseline
    
    elif 'Ridge' in cls_name:
        alpha = params.get('alpha', 1.0)
        return 1.0 / (1.0 + alpha) # Decreases with stronger regularization
    
    elif 'Lasso' in cls_name:
        alpha = params.get('alpha', 1.0)
        return 1.0 / (1.0 + alpha * 10) # Stronger penalty for sparsity
    
    elif 'RandomForest' in cls_name:
        max_depth = params.get('max_depth')
        if max_depth is None:
            max_depth = 20 # Assumption for unbounded trees
        n_estimators = params.get('n_estimators', 100)
        max_features = params.get('max_features', 1.0)
        if isinstance(max_features, str):
            if max_features in ['auto', 'sqrt', 'log2']:
                max_features = 0.33 # Approximate fraction
            else:
                max_features = 1.0
        elif max_features is None:
            max_features = 1.0
        effective_complexity = n_estimators * max_depth * max_features
        return 1.0 + effective_complexity / 200.0 # Heuristic scaling
        
    elif 'XGB' in cls_name or 'LGBM' in cls_name or 'CatBoost' in cls_name:
        max_depth = params.get('max_depth', 6)
        n_estimators = params.get('n_estimators', 100)
        effective_complexity = n_estimators * (2 ** max_depth) # Exponential growth with depth
        return 1.0 + math.log(1 + effective_complexity) / math.log(2) / 10.0 # Logarithmic scaling

    elif 'DecisionTree' in cls_name:
        max_depth = params.get('max_depth')
        if max_depth is None:
            max_depth = 20
        return 1.0 + max_depth / 5.0 # Linear scaling with depth

    elif 'SVC' in cls_name or 'SVR' in cls_name:
        kernel = params.get('kernel', 'rbf')
        if kernel == 'linear':
            return 1.2 # Simple linear model approximation
        else:
            C = params.get('C', 1.0)
            gamma = params.get('gamma', 'scale')
            if gamma == 'scale':
                gamma = 1.0
            elif gamma == 'auto':
                gamma = 0.1
            return 2.0 + math.log(1 + C * gamma) # Complexity increases with C and gamma

    elif 'KNeighbors' in cls_name:
        n_neighbors = params.get('n_neighbors', 5)
        # Complexity is inversely proportional to n_neighbors (smoother model with more neighbors)
        return 1.0 + 10.0 / n_neighbors

    elif 'MLP' in cls_name:
        hidden = params.get('hidden_layer_sizes', (100,))
        if isinstance(hidden, int):
            hidden = (hidden,)
        total_neurons = sum(hidden)
        n_layers = len(hidden)
        # Complexity scales with number of layers and total neurons
        return 1.0 + (n_layers * total_neurons) / 50.0

    else:
        # Default score for estimators not explicitly handled
        return 2.0
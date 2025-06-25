import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from joblib import Parallel, delayed
from sklearn.datasets import load_iris

def gcorCI(x, y, confidence_level=0.95, alpha=1, parallel=False, n_jobs=-2):
    """
    Compute an approximate confidence interval for the Categorical Gini Correlation based on asymptotic normality. 
    
    Parameters:
    ----------
    x : array-like
        A single numeric feature or a set of numeric features 
        (e.g., a column or matrix of data).

    y : array-like
        A list or array of categorical labels, with one label per sample in `x`.

    alpha : float, optional (default=1)
        A power parameter applied to the pairwise distances.
        For standard CGC, set alpha = 1. 
        In the presence of outliers, choose a smaller alpha value.

    confidence_level : float, optional (default=0.95)
        Confidence level for the interval.

    parallel : bool, optional (default=False)
        Whether to parallelize the jackknife computations.

    n_jobs : int, optional (default=-2)
        Number of jobs for parallel processing.
        -1 means use all cores; -2 means all but one core.

    Returns:
    -------
    ci : list of float
        Confidence interval for the Gini correlation [lower, upper].
    """

    x = np.asarray(x)
    y = np.asarray(y)
    n_samples = x.shape[0]

    def compute_gini_corr(x_inner, y_inner, alpha_inner=alpha):
        dist_matrix = cdist(x_inner, x_inner, 'euclidean') ** alpha_inner
        sorted_idx = np.argsort(y_inner)
        sorted_dist = dist_matrix[sorted_idx][:, sorted_idx]
        _, counts = np.unique(y_inner, return_counts=True)

        total_mean = np.sum(dist_matrix) / (len(x_inner) * (len(x_inner) - 1))
        start_idx = 0
        block_sum = 0.0

        for count in counts:
            if count < 2:
                start_idx += count
                continue
            block = sorted_dist[start_idx:start_idx + count, start_idx:start_idx + count]
            block_sum += np.sum(block) / (count - 1)
            start_idx += count

        between_var = block_sum / len(x_inner)
        gdcov = total_mean - between_var
        return gdcov / total_mean
    
    rhog = compute_gini_corr(x, y)

    def jackknife_leave_one_out(i):
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        return compute_gini_corr(x[mask], y[mask])

    if parallel:
        jackknife_estimates = Parallel(n_jobs=n_jobs)(delayed(jackknife_leave_one_out)(i) for i in range(n_samples))
    else:
        jackknife_estimates = [jackknife_leave_one_out(i) for i in range(n_samples)]

    jackknife_estimates = np.array(jackknife_estimates)

    var_jackknife = np.var(jackknife_estimates, ddof=1)
    se_jackknife = (n_samples - 1) * np.sqrt(var_jackknife / n_samples)

    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    ci_lower = rhog - z_score * se_jackknife
    ci_upper = rhog + z_score * se_jackknife

    print(f"Categorical Gini Correlation: {rhog:.6f}")
    print(f"{100 * confidence_level:.0f}% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]")

    return [ci_lower, ci_upper]


# Example Usage

iris = load_iris()
X = iris.data[:, :2]  # Use first two features
y = iris.target

gcorCI(X, y, confidence_level=0.95, parallel=False)

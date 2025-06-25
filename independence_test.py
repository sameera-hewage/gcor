import numpy as np
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

def gcor(x, y, alpha=1):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = len(x)
    xm = cdist(x, x, metric='euclidean') ** alpha
    total = np.sum(xm) / (n * (n - 1))

    sorted_indices = np.argsort(y)
    xs = xm[sorted_indices][:, sorted_indices]
    _, class_counts = np.unique(y, return_counts=True)

    idx = 0
    bs = 0.0
    for count in class_counts:
        if count < 2:
            idx += count
            continue
        block = xs[idx:idx + count, idx:idx + count]
        block_sum = np.sum(block) - np.trace(block)
        bs += block_sum / (count - 1)
        idx += count

    bv = bs / n
    cgc = (total - bv) / total
    return cgc

def independence_test(x, y, B=1000, slevel=0.05, alpha=1, parallel=True, n_jobs=-2):
    """
    Perform a permutation test for independence using CGC.

    Parameters:
    x : array-like
        A single numeric feature or a set of numeric features 
        (e.g., a column or matrix of data).
    y : array-like
        A list or array of categorical labels, with one label per sample in `x`.
    - B: int, number of permutations
    - slevel: float, significance level (default 0.05)
    - alpha: float, distance exponent in CGC
    - parallel: bool, whether to use parallel computation
    - n_jobs: int, number of jobs for parallelization (-1 = all cores)

    Returns:
    - critical_value: empirical (1 - significance_level)-quantile from permutation distribution
    - p_value: permutation p-value
    - reject_null: True if null hypothesis is rejected
    """
    gc_obs = gcor(x, y, alpha) #observed statistic

    if parallel:
        gc_perm = Parallel(n_jobs=n_jobs)(
            delayed(gcor)(x, np.random.permutation(y)) for _ in range(B)
        )
    else:
        gc_perm = [gcor(x, np.random.permutation(y)) for _ in range(B)]

    gc_perm = np.array(gc_perm)
    p_value = np.mean(gc_perm >= gc_obs)
    critical_value = np.quantile(gc_perm, 1 - slevel)
    reject_null = gc_obs > critical_value

    print(f"Critical value at alpha = {slevel}: {critical_value:.4f}")
    print(f"P-value from permutation test: {p_value:.4f}")
    print(f"Reject null hypothesis of independence: {reject_null}")
    return critical_value, p_value, reject_null


#Example Usage

np.random.seed(123)

# Generate 3-class data from same distribution
n_per_group = 50
x1 = np.random.normal(loc=0, scale=1, size=(n_per_group, 2))
x2 = np.random.normal(loc=0, scale=1, size=(n_per_group, 2))
x3 = np.random.normal(loc=0, scale=1, size=(n_per_group, 2))

x = np.vstack([x1, x2, x3])
y = np.array([0]*n_per_group + [1]*n_per_group + [2]*n_per_group)

crit, p_value, reject_null = independence_test(x, y, B=1000, slevel=0.05, alpha=1, parallel=False)



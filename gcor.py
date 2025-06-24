import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris

def gcor(x, y, alpha=1):
    """
    Computes the Categorical Gini Correlation (CGC) between numerical data `x`
    and a set of categorical labels `y`.

    Parameters
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

    Returns
    -------
    gdc : float
        The estimated Categorical Gini Correlation value.
    """
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

    print(f"Categorical Gini Correlation: {cgc:.6f}")
    return cgc

# Example usage with the Iris dataset

iris = load_iris()
x = iris.data[:, :4]  # Use all four features
y = iris.target       # Species as class labels

gcor(x, y, alpha=1)

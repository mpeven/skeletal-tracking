import numpy as np

def k_means(vectors, k, max_iterations=300, visualize=False):
    np.random.seed(7) # TESTING ONLY

    """ K-means helper functions """
    def initialize(vectors, k):
        v_copy = vectors.copy()
        np.random.shuffle(v_copy)
        return v_copy[:k]

    def assign(vectors, means):
        distances = np.sqrt(((vectors - means[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def update(vectors, assigned_mean, means):
        return np.array([vectors[assigned_mean==k].mean(axis=0) for k in range(means.shape[0])])

    """ K-means pipeline """
    means = initialize(vectors, k)
    for x in range(max_iterations):
        old_means = means
        assignments = assign(vectors, means)
        means = update(vectors, assignments, means)
        if np.array_equal(old_means, means):
            print("Done.  Converged on iteration {}".format(x))
            break

    return means

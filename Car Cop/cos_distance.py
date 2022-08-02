def cos_distance (d1, d2):
    """
    Calculates the cosine distance between two descriptor vectors.

    Parameters
    ----------
    d1 : numpy.ndarray, shape-(M, D)
        The first descriptor vector, an array of M descriptor vectors.

    d2 : numpy.ndarray, shape-(N, D)
        The second descriptor vector, an array of N descriptor vectors.

    Returns
    -------
    numpy.ndarray, shape-(M, N)
        An array of cosine distances which holds all MxN combination so of pairwise cosine distances.
    """
    import numpy as np

    d1_mag = np.linalg.norm(d1, axis=1)
    d2_mag = np.linalg.norm(d2, axis=1)
    return 1 - ( np.matmul(d1, d2.T) / (d1_mag * d2_mag) )

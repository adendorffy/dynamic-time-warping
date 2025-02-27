import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path


def dtw_sweep_min(query_seq, search_seq, n_step=3):
    """
    Return the minimum DTW cost as `query_seq` is swept across `search_seq`.

    Step size can be specified with `n_step`.
    """

    from cython_dtw import _dtw

    dtw_cost_func = _dtw.multivariate_dtw_cost_cosine

    i_start = 0
    n_query = query_seq.shape[0]
    n_search = search_seq.shape[0]
    min_cost = np.inf

    while i_start <= n_search - n_query or i_start == 0:
        cost = dtw_cost_func(query_seq, search_seq[i_start : i_start + n_query], True)
        i_start += n_step
        if cost < min_cost:
            min_cost = cost

    return min_cost


def encodings_from_words(words):
    encodings = []
    for word in words:
        encodings.append(word.clean_encoding)
    return encodings


def dtw(encodings, save=False):
    num_features = len(encodings)
    norm_distance_mat = np.zeros((num_features, num_features))
    encodings = [f.cpu().numpy().astype(np.float64) for f in encodings]

    for i in tqdm(range(num_features), desc="Calculating Distances"):
        dists_i = Parallel(n_jobs=8)(
            delayed(dtw_sweep_min)(encodings[i], encodings[j])
            for j in range(i + 1, num_features)
        )

        for j, dist in zip(range(i + 1, num_features), dists_i):
            norm_distance_mat[i, j] = dist
            norm_distance_mat[j, i] = dist

    if save:
        out_dir = Path(save)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "dist_mat.npy", norm_distance_mat)

    return norm_distance_mat

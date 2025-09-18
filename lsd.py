import librosa
import numpy as np
import scipy
from sklearn.cluster import KMeans

def construct_graph(rep_feat, loc_feat, rep_width, rec_smooth, rep_metric="cosine"):
    R = librosa.segment.recurrence_matrix(
        rep_feat, mode="affinity", metric=rep_metric, width=rep_width, sym=True
    )

    # Enhance diagonals with a median filter (Equation 2 from LSD paper)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, rec_smooth))

    path_distance = np.sum(np.diff(loc_feat, axis=1) ** 2, axis=0)
    path_bw = np.median(path_distance)
    path_sim = np.exp(-path_distance / path_bw)

    # And compute the balanced combination (Equations 6, 7, 9)
    # Combining R and path_sim
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)

    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec) ** 2)

    A = mu * Rf + (1 - mu) * R_path

    return A


# TODO: I could save the beat embedding for each track feature combo?
def beat_embedding(A, smooth=9):
    # build graph laplacian
    L = scipy.sparse.csgraph.laplacian(A, normed=True)

    # and its spectral decomposition
    evals, evecs = scipy.linalg.eigh(L)

    # We can clean this up further with a median filter.
    # This can help smooth over small discontinuities
    if smooth is not None:
        evecs = scipy.ndimage.median_filter(evecs, size=(smooth, 1))

    return evals, evecs


def segment_embedding(evecs, Cnorm, k):
    # Normalize the eigenvectors by the cumulative norm
    # This is needed for symmetric normalized laplacian eigenvectors
    # See Tutorial on Spectral Clustering paper
    X = evecs[:, :k] / (Cnorm[:, k - 1 : k] + 1e-8)
    KM = KMeans(n_clusters=k, init="k-means++", n_init=10)
    seg_ids = KM.fit_predict(X)

    # Locate segment boundaries from the label sequence
    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

    # Count beat 0 as a boundary
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

    # Compute the segment label for each boundary
    seg_labels = list(seg_ids[bound_beats])

    return bound_beats, seg_labels


def run(
    feats, rep_feat='openl3', loc_feat='mfcc', depth=16, 
    rep_width=5, rec_smooth=9, evec_smooth=11, delay_steps=4,
):
    from mir_eval.util import boundaries_to_intervals
    # time delay embeddings
    rep_feat_mat = librosa.feature.stack_memory(feats[rep_feat], n_steps=delay_steps, mode='edge')
    loc_feat_mat = librosa.feature.stack_memory(feats[loc_feat], n_steps=delay_steps, mode='edge')
    
    # construct recurrence matrix
    A = construct_graph(
        rep_feat_mat, loc_feat_mat, rep_width=rep_width, rec_smooth=rec_smooth
    )
    evals, evecs = beat_embedding(A, smooth=evec_smooth)

    # extract top k eigenvectors
    first_evecs = evecs[:, :depth]
    Cnorm = np.cumsum(first_evecs**2, axis=1) ** 0.5

    # iterate through all k levels and build segmentation 1 layer at a time
    itvls = []
    labels = []
    for k in range(1, depth + 1):
        bound_beats, seg_labels = segment_embedding(first_evecs, Cnorm, k)
        # add-in last beat, change to intervals and make string labels
        bound_time = np.append(feats["bs"][bound_beats], feats["bs"][-1])
        lvl_bdry, lvl_label = remove_empty_segments(bound_time, seg_labels)

        lvl_itvls = boundaries_to_intervals(lvl_bdry)
        itvls.append(lvl_itvls)
        labels.append(lvl_label)

    return itvls, labels


# region: Post-processing Clean ups


def remove_empty_segments(boundaries, labels):
    assert len(boundaries) - 1 == len(labels)
    bad_idx = np.where(np.diff(boundaries) == 0)
    return np.delete(boundaries, bad_idx), np.delete(labels, bad_idx).astype(str)

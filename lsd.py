import librosa
import numpy as np
from scipy import sparse
import scipy
from sklearn.cluster import KMeans
from collections import defaultdict

def mask_diag(sq_mat, width=2):
    # carve out width from the full ssm
    sq_mat_lil = sparse.lil_matrix(sq_mat)
    for diag in range(-width + 1, width):
        sq_mat_lil.setdiag(0, diag)
    return sq_mat_lil.toarray()


def construct_graph(rep_feat, loc_feat,
                    rep_width=3, rec_smooth=7,
                    rep_metric='cosine'):
    
    R = librosa.segment.recurrence_matrix(
        rep_feat, mode='affinity', metric=rep_metric,
        width=rep_width, sym=True
    )

    # Enhance diagonals with a median filter (Equation 2 from LSD paper)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, rec_smooth))

    path_distance = np.sum(np.diff(loc_feat, axis=1)**2, axis=0)
    path_bw = np.median(path_distance)
    path_sim = np.exp(-path_distance / path_bw)

    # And compute the balanced combination (Equations 6, 7, 9)
    # Combining R and path_sim
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
    
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)

    
    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

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
    evecs = scipy.ndimage.median_filter(evecs, size=(smooth, 1))

    return evals, evecs


def segment_embedding(evecs, Cnorm, k):
    # Normalize the eigenvectors by the cumulative norm
    # This is needed for symmetric normalized laplacian eigenvectors
    # See Tutorial on Spectral Clustering paper
    X = evecs[:, :k] / (Cnorm[:, k-1:k] + 1e-8)
    KM = KMeans(n_clusters=k, init="k-means++")
    seg_ids = KM.fit_predict(X)

    # Locate segment boundaries from the label sequence
    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

    # Count beat 0 as a boundary
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

    # Compute the segment label for each boundary
    seg_labels = list(seg_ids[bound_beats])

    return bound_beats, seg_labels


def do_hierarchy(feats, rep_ftype, loc_ftype, depth=16, verbose=True):
    from mir_eval.util import boundaries_to_intervals
    A = construct_graph(feats[rep_ftype], feats[loc_ftype])
    evals, evecs = beat_embedding(A)

    first_evecs = evecs[:, :depth]
    Cnorm = np.cumsum(first_evecs**2, axis=1) ** 0.5
    
    # iterate through all k levels and build segmentation 1 layer at a time
    itvls = []
    labels = []
    for k in range(1, depth + 1):
        bound_beats, seg_labels = segment_embedding(first_evecs, Cnorm, k)
        # add-in last beat, change to intervals and make string labels
        bound_time = np.append(feats['bs'][bound_beats], feats['bs'][-1])
        lvl_bdry, lvl_label = remove_empty_segments(bound_time, seg_labels)
        
        lvl_itvls = boundaries_to_intervals(lvl_bdry)
        itvls.append(lvl_itvls)
        labels.append(lvl_label)

    return reindex(itvls, labels)

# region: Post-processing Clean ups

def remove_empty_segments(boundaries, labels):
    assert len(boundaries) - 1 == len(labels)
    bad_idx = np.where(np.diff(boundaries) == 0)
    return np.delete(boundaries, bad_idx), np.delete(labels, bad_idx).astype(str)

    
def _reindex_labels(ref_int, ref_lab, est_int, est_lab):
    # for each estimated label
    #    find the reference label that is maximally overlaps with
    score_map = defaultdict(lambda: 0)

    for r_int, r_lab in zip(ref_int, ref_lab):
        for e_int, e_lab in zip(est_int, est_lab):
            score_map[(e_lab, r_lab)] += max(0, min(e_int[1], r_int[1]) -
                                             max(e_int[0], r_int[0]))

    r_taken = set()
    e_map = dict()

    hits = [(score_map[k], k) for k in score_map]
    hits = sorted(hits, reverse=True)

    while hits:
        cand_v, (e_lab, r_lab) = hits.pop(0)
        if r_lab in r_taken or e_lab in e_map:
            continue
        e_map[e_lab] = r_lab
        r_taken.add(r_lab)

    # Anything left over is unused
    unused = set(est_lab) - set(ref_lab)

    for e, u in zip(set(est_lab) - set(e_map.keys()), unused):
        e_map[e] = u

    return [e_map[e] for e in est_lab]


def reindex(itvls, labels):
    new_labels = [labels[0]]
    for i in range(1, len(labels)):
        labs = _reindex_labels(
            itvls[i-1], new_labels[i - 1], itvls[i], labels[i]
        )
        new_labels.append(labs)

    return itvls, new_labels

# endregion
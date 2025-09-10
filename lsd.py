import librosa
import numpy as np
from scipy import sparse
import scipy
from .features import load_synced_feats

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

    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)

    # And compute the balanced combination (Equations 6, 7, 9)
    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

    A = mu * Rf + (1 - mu) * R_path
    
    return A


# I need to save the beat embedding for each track feature combo
def beat_embedding(A, smooth=9, n_evecs=16):
    # build graph laplacian
    L = scipy.sparse.csgraph.laplacian(A, normed=True)

    # and its spectral decomposition
    evals, evecs = scipy.linalg.eigh(L)

    # We can clean this up further with a median filter.
    # This can help smooth over small discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(smooth, 1))

    return evecs


def segment_embedding(evecs, Cnorm, k):
    from sklearn.cluster import KMeans
    
    # Normalize the eigenvectors by the cumulative norm
    # This is needed for symmetric normalized laplacian eigenvectors
    # See Tutorial on Spectral Clustering paper
    X = evecs[:, :k] / (Cnorm[:, k-1:k] + 1e-8)
    KM = KMeans(n_clusters=k, init="auto")
    seg_ids = KM.fit_predict(X)

    # Locate segment boundaries from the label sequence
    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

    # Count beat 0 as a boundary and also the end of the track
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0, x_max=X.shape[0])

    # Compute the segment label for each boundary
    seg_labels = list(seg_ids[bound_beats])

    return bound_beats, seg_labels


def do_hierarchy(audio_path, rep_ftype, loc_ftype):
    from mir_eval.util import boundaries_to_intervals
    feats = load_synced_feats(audio_path)
    A = construct_graph(feats[rep_ftype], feats[loc_ftype])
    evecs = beat_embedding(A)
    Cnorm = np.cumsum(evecs**2, axis=0) ** 0.5

    # iterate through all k levels and build segmentation 1 layer at a time
    itvls = []
    labels = []
    for k in range(1, evecs.shape[1] + 1):
        bound_beats, seg_labels = segment_embedding(evecs, Cnorm, k)
        boundaries = feats['bs'][bound_beats]
        itvls.append(boundaries_to_intervals(boundaries))
        labels.append(seg_labels)
    return itvls, labels

    
    
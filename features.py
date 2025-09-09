import os
import numpy as np
import torch
import tensorflow as tf
import librosa
import gc
import librosa.display
import openl3
import openl3.models
import crema
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from functools import lru_cache

# region: Model and Audio Loading

def torch_device_default():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def clear_gpu_memory():
    """Clear GPU memory for both PyTorch and TensorFlow"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Clear TensorFlow memory
    tf.keras.backend.clear_session()
    gc.collect()


@lru_cache(maxsize=1)
def load_yamnet_model(model_dir: str = '/Users/tomxi/code/yamnet_model'):
    return tf.saved_model.load(model_dir)


@lru_cache(maxsize=1)
def load_openl3_model():
    return openl3.models.load_audio_embedding_model(
        'mel256', 'music', 512, frontend='kapre'
    )


@lru_cache(maxsize=1)
def load_crema_model():
    return crema.models.chord.ChordModel()



@lru_cache(maxsize=32)
def cached_load_audio(audio_path, sr):
    return librosa.load(str(audio_path), sr=sr)

def get_track_basename(audio_path):
    track_bn = os.path.basename(audio_path).split('.')[0]
    # Salami tracks are named 'audio.mp3' so we need to get the directory name
    if track_bn == 'audio':
        track_bn = os.path.dirname(audio_path).split('/')[-1]
    return track_bn

# endregion

# region: Beat Processing
def get_beats(audio_path):
    y, sr = cached_load_audio(audio_path, 22050)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False, start_bpm=80, units='time')
    return fix_beats(beats, y.shape[0] / sr)


def fix_beats(beats, track_dur, min_beat_dist=0.2):
    """
    Makes sure that 0 and track_dur are always included as first and last beat.
    Also removes any beats falling outside the track duration.
    Also removes any beats that are too close to each other, including 0 and track_dur.
    """
    # make sure all beats are within the track duration
    candidates = np.unique(np.clip(beats, 0, track_dur))
    
    # Iteratively get rid of any beats that's too close
    good_beats = [0] # always have 0 as first beat
    for beat in candidates:
        if beat - good_beats[-1] >= min_beat_dist:
            good_beats.append(beat)

    # If the last beat is too close to the end, replace it by track_dur
    # otherwise append track_dur.
    if good_beats[-1] > (track_dur - min_beat_dist):
        good_beats[-1] = track_dur
    else:
        good_beats.append(track_dur)

    return np.array(good_beats)


def beat_sync_features(features, ts, beats, upsample_sr=20):
    """
    Synchronizes a feature matrix to a set of beat times using high-resolution
    interpolation.
    Assumes features is a 2D array with time as the last axis.
    """
    duration = beats[-1]
    target_ts = np.linspace(0, duration, num=int(duration * upsample_sr))

    interpolator = interp1d(
        ts,
        features,
        axis=-1,
        bounds_error=False,
        fill_value=(features[:, 0], features[:, -1])
    )
    upsampled_feats = interpolator(target_ts)

    beat_frames = np.searchsorted(target_ts, beats, side='left')
    synced_feats = librosa.util.sync(
        upsampled_feats,
        beat_frames,
        aggregate=np.median, 
        pad=False,
        axis=-1
    )

    if synced_feats.shape[-1] != (len(beats) - 1):
        raise ValueError(
            "Beat synchronization shape mismatch! " + \
            f"{synced_feats.shape[-1]} != {len(beats)} - 1"
        )
    return synced_feats


# endregion

# region: Feature Extraction

# def compute_and_sync_feature(audio_path, output_dir, feature_name, beats_dir, recompute, compute_fn):
#     os.makedirs(output_dir, exist_ok=True)
#     track_bn = get_track_basename(audio_path)
#     feat_path = os.path.join(output_dir, f'{track_bn}_{feature_name}.npz')
    
#     if recompute or not os.path.exists(feat_path):
#         feat, emb_sr = compute_fn(audio_path)
#         track_beats = adjusted_beats(audio_path, beats_dir, recompute)['adjusted_beats']
#         feat_sync, feat_boundaries = beat_sync(feat, track_beats, emb_sr)
#         np.savez(feat_path, feature=feat_sync, ts=feat_boundaries)
#     return np.load(feat_path)


def yamnet_emb(audio_path):
    yamnet_model = load_yamnet_model()
    audio, sr = cached_load_audio(audio_path, 16000)
    _, emb, _ = yamnet_model(audio)
    emb_arr = emb.numpy().T
    # YamNet frames are left aligned and uses a hop length of 0.48s and a window size of 0.96s
    ts = librosa.times_like(emb_arr, sr=sr, hop_length=7680, n_fft=15360)
    return emb_arr, ts


def openl3_emb(audio_path):
    openl3_model = load_openl3_model()
    y, sr = cached_load_audio(audio_path, 22050)
    emb, ts = openl3.get_audio_embedding(
        y, sr, model=openl3_model, center=True,
        input_repr='mel256', content_type='music', embedding_size=512
    )
    return emb.T, ts


def crema_emb(audio_path, device='cpu'):
    with tf.device(device):
        chord_model = load_crema_model()
        crema_out = chord_model.outputs(filename=str(audio_path))

    crema_op = chord_model.pump.ops[2]
    crema_emb = np.concatenate([crema_out['chord_bass'], crema_out['chord_pitch']], axis=1).T
    ts = librosa.times_like(crema_emb, sr=crema_op.sr, hop_length=crema_op.hop_length)
    return crema_emb, ts


def mfcc(audio_path):
    y, sr = cached_load_audio(audio_path, 22050)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=40, center=True,
        hop_length=4096, n_fft=8192, lifter=0.6)
    normalized_mfcc = (mfcc - np.mean(mfcc, axis=1)[:, None]) / np.std(mfcc, axis=1, ddof=1)[:,None]
    ts = librosa.times_like(normalized_mfcc, sr=sr, hop_length=4096)
    return normalized_mfcc, ts


def tempogram(audio_path):
    y, sr = cached_load_audio(audio_path, 22050)
    tempogram = librosa.feature.tempogram(
        y=y, sr=sr, 
        hop_length=512, win_length=384, center=True
    )
    ts = librosa.times_like(tempogram, sr=sr, hop_length=512)
    return tempogram, ts

# endregion

def plot(features, ts, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    feat_dim = features.shape[0]
    if len(ts) == features.shape[1]+1:
        y_coords = np.arange(feat_dim + 1)
    else:
        y_coords = np.arange(feat_dim)
    # Use specshow to plot the features. 
    mesh = librosa.display.specshow(
        features, 
        x_axis='time', x_coords=ts, 
        y_axis='none', y_coords=y_coords,
        ax=ax
    )    
    fig.colorbar(mesh, ax=ax)
    return ax


## Deprecated ##
def consolidate_features(audio_path, output_dir='feats', reconsolidate=False):
    track_bn = get_track_basename(audio_path)
    outpath = os.path.join(output_dir, f'{track_bn}_feats.npz')
    
    if reconsolidate or not os.path.exists(outpath):
        raw_feats = dict(
            yamnet = yamnet_emb(audio_path),
            openl3 = openl3_emb(audio_path),
            crema = crema_emb(audio_path),
            mfcc = mfcc(audio_path),
            tempogram = tempogram(audio_path)
        )

        # get the union of all the sets in feat_ts, since the features are on different time_grids
        feat_ts_sets = [set(raw_feats[f]['ts']) for f in raw_feats]
        common_ts = set.intersection(*feat_ts_sets)
        common_ts = np.array(sorted(common_ts))
        synced_feats = dict(ts=common_ts)
        for feat in raw_feats:
            frame_indices = np.searchsorted(raw_feats[feat]['ts'], common_ts)
            synced_feats[feat] = librosa.util.sync(raw_feats[feat]['feature'], frame_indices, aggregate=np.median, pad=False)
            if synced_feats[feat].shape[1] != (len(common_ts) - 1):
                raise ValueError(
                    "Feature synchronization shape mismatch!" + \
                    f"{synced_feats[feat].shape[1]} != {len(common_ts)} - 1"
                )
        os.makedirs(output_dir, exist_ok=True)
        np.savez(outpath, **synced_feats)
    return np.load(outpath)
    
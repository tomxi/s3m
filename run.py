import bnl
import features as ft
import random

if __name__ == '__main__':
    slm_ds = bnl.Dataset("/scratch/qx244/data/salami/metadata.csv")
    tids = slm_ds.track_ids
    random.shuffle(tids)

    for tid in tids:
        audio_path = slm_ds[tid].info['audio_mp3_path']
        try:
            ft.load_synced_feats(audio_path)
        except Exception as e:
            print(f"Failed to load features for {tid}: {e}")

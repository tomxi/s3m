import bnl
import features as ft
import random
import argparse

if __name__ == '__main__':
    slm_ds = bnl.Dataset("/scratch/qx244/data/salami/metadata.csv")
    tids = slm_ds.track_ids
    random.shuffle(tids)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ftype", type=str, default="yamnet")
    args = parser.parse_args()
    
    for tid in tids:
        audio_path = slm_ds[tid].info['audio_mp3_path']
        try:
            ft.load_feats(audio_path, feat_type=args.ftype)
        except Exception as e:
            print(f"Failed to load features for {tid}: {e}")

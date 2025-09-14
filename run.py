import bnl
import exp
import random
from tqdm import tqdm
import os

if __name__ == '__main__':
    slm_ds = bnl.Dataset("/scratch/qx244/data/salami/metadata.csv")
    tids = slm_ds.track_ids
    random.shuffle(tids)

    lsd_configs = {
        'rep_width': 5,
        'rec_smooth': 7,
        'evec_smooth': 9,
        'delay_steps': 4,
    }

    os.makedirs("/scratch/qx244/data/salami/scores", exist_ok=True)

    for tid in tqdm(tids):
        audio_path = slm_ds[tid].info['audio_mp3_path']
        score_path = f"/scratch/qx244/data/salami/scores/{tid}.feather"
        if os.path.exists(score_path):
            continue
        try:
            scores = exp.all_feat_combo_scores(slm_ds[tid], lsd_configs)
            scores.to_feather(score_path)
        except Exception as e:
            print(f"Failed to compute scores for {tid}: {e}")

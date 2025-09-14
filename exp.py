import bnl
import lsd
import os
import random
import json
import frameless_eval as fle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from pqdm.processes import pqdm
from mpl_toolkits.axes_grid1 import ImageGrid

SLM_DIR = "/scratch/qx244/data/salami"

def get_small_set(size=10, seed=42):
    slm_ds = bnl.Dataset(os.path.join(SLM_DIR, "metadata.csv"))
    random.seed(seed)
    random_ids = random.sample(slm_ds.track_ids, size)
    return [slm_ds[tid] for tid in random_ids]

def lsd_option_to_str(lsd_configs):
    short_keys = {
        'rep_feat': 'rf', 
        'loc_feat': 'lf', 
        'rep_width': 'rw', 
        'rec_smooth': 'rs', 
        'evec_smooth': 'es',
        'delay_steps': 'ds'
    }
    return '_'.join([f'{short_keys[k]}={v}' for k, v in lsd_configs.items()])

def serialize_ms(ms):
    itvls_json = [itvl.round(3).tolist() for itvl in ms.itvls]
    return {'itvls': itvls_json, 'labels': ms.labels}


def get_est_raw(track, lsd_configs, recompute=False):
    cache_dir = os.path.join(SLM_DIR, 'lsd')
    os.makedirs(cache_dir, exist_ok=True)
    est_name = f'{track.track_id}_{lsd_option_to_str(lsd_configs)}'
    cache_path = os.path.join(
        cache_dir, 
        f'{est_name}.json'
    )

    if recompute or not os.path.exists(cache_path):
        est_raw = (
            bnl.MS.from_itvls(*lsd.run(track.feats, **lsd_configs))
            .prune_layers()
            .relabel()
        )
        result = {'track_id': track.track_id}
        result.update(serialize_ms(est_raw))
        result.update(lsd_configs)
        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSuccessfully saved data to '{cache_path}'")
    else:
        with open(cache_path, 'r') as f:
            result = json.load(f)
    return bnl.MS.from_itvls(result['itvls'], result['labels'], name=est_name)

def get_scores(track, lsd_configs):
    ref = track.ref.expand_labels()
    est_raw = get_est_raw(track, lsd_configs).align(ref)
    ref_bh = ref.contour('count').level()
    est_bh_raw = est_raw.contour('depth')
    est_bh_cleaned = est_raw.contour('prob').clean('kde', bw=0.8).level('mean_shift', bw=0.12)
    est_cleaned = est_raw.lam('prob').decode(est_bh_cleaned, aff_mode='area', starting_k=2, min_k_inc=1)
    
    l_score_raw = fle.lmeasure(ref.itvls, ref.labels, est_raw.itvls, est_raw.labels)
    b_score_raw = bnl.metrics.bmeasure(ref_bh, est_bh_raw, window=1.5)
    l_score_cleaned = fle.lmeasure(ref.itvls, ref.labels, est_cleaned.itvls, est_cleaned.labels)
    b_score_cleaned = bnl.metrics.bmeasure(ref_bh, est_bh_cleaned, window=1.5)

    # collect prf
    scores = {
        'raw_b': (b_score_raw['b_p'], b_score_raw['b_r'], b_score_raw['b_f']),
        'raw_l': l_score_raw,
        'cleaned_b': (b_score_cleaned['b_p'], b_score_cleaned['b_r'], b_score_cleaned['b_f']),
        'cleaned_l': l_score_cleaned
    }

    # Transform to long format
    data_for_df = []
    for score_name, (p, r, f) in scores.items():
        data_for_df.append({'score_type': score_name, 'prf': 'p', 'score': p})
        data_for_df.append({'score_type': score_name, 'prf': 'r', 'score': r})
        data_for_df.append({'score_type': score_name, 'prf': 'f', 'score': f})
    
    df = pd.DataFrame(data_for_df)
    df['track_id'] = track.track_id
    df['rep_feat'] = lsd_configs['rep_feat']
    df['loc_feat'] = lsd_configs['loc_feat']
    df['rep_width'] = lsd_configs['rep_width']
    df['rec_smooth'] = lsd_configs['rec_smooth']
    df['evec_smooth'] = lsd_configs['evec_smooth']
    df['delay_steps'] = lsd_configs['delay_steps']
    
    # long_df = df[['track_id', 'rep_feat', 'loc_feat', 'rep_width', 'rec_smooth', 'evec_smooth', 'delay_steps', 'score_type', 'prf', 'score']]
    return df.pivot_table(
        index=['track_id', 'rep_width', 'rec_smooth', 'evec_smooth', 'delay_steps', 'rep_feat', 'loc_feat', 'prf'], 
        columns='score_type', 
        values='score'
    )

def all_feat_combo_scores(track, lsd_configs):
    available_feats = ['openl3', 'crema', 'mfcc', 'tempogram', 'yamnet']
    
    all_scores = []
    for rep_feat in available_feats:
        lsd_configs['rep_feat'] = rep_feat
        for loc_feat in available_feats:
            lsd_configs['loc_feat'] = loc_feat
            all_scores.append(get_scores(track, lsd_configs))
    return pd.concat(all_scores).reset_index()

def plot_scores(score_df, prf='f', score_type='cleaned_b', ax=None, vmin=None, vmax=None, cbar=True, show_ylabel=True):
    filtered_df = score_df[score_df.prf == prf]
    # Aggregate scores across track_id using mean
    aggregated_df = filtered_df.groupby(
        ['rep_feat', 'loc_feat', 'rep_width', 'rec_smooth', 'evec_smooth', 'delay_steps']
    ).mean(numeric_only=True).reset_index()
    
    # filtered_df is 25 rows, with rep_feat and loc_feat taking on each of the 5 available features as values
    # I want to plot a heatmap that's 5-by-5 to visualize the scores.
    heatmap_data = aggregated_df.pivot_table(index='rep_feat', columns='loc_feat', values=score_type)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    mappable = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='viridis', ax=ax, vmin=vmin, vmax=vmax, cbar=cbar)
    ax.set_title(f'{score_type} {prf}-score')
    ax.set_xlabel('Local Feature')
    if show_ylabel:
        ax.set_ylabel('Repetition Feature')
    else:
        ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=45)
    ax.set_aspect('equal')
    return ax, mappable

def plot_all_scores(score_df, prf='f', figsize=(11,5)):
    score_types = ['cleaned_b', 'cleaned_l', 'raw_l']
    
    # Calculate global vmin and vmax for a unified color scale
    filtered_df = score_df[score_df.prf == prf]
    vmin = filtered_df[score_types].min().min()
    vmax = filtered_df[score_types].max().max()

    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,  # as in subplot(111)
                   nrows_ncols=(1, 3),
                   axes_pad=0.15,
                   share_all=True,
                   cbar_location="right",
                   cbar_mode="single",
                   cbar_size="5%",
                   cbar_pad=0.15,
                   )

    # Plot the data on each of the axes in the grid
    for i, (ax, score_type) in enumerate(zip(grid, score_types)):
        _, mappable = plot_scores(
            score_df,
            prf=prf,
            score_type=score_type,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            show_ylabel=(i == 0)
        )

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # Set a dummy array for the mappable
    sm.set_array([])
    # Draw the colorbar in the grid's colorbar axis
    grid.cbar_axes[0].colorbar(sm)
    # grid.cbar_axes[0].set_label('Score')

    fig.suptitle('Comparison of Score Types (F-score)', fontsize=16, y=0.88)
    return fig


def lsd_config_selection_experiment():
    """ Search over a small grid of lsd_configs and save the results for analysis. """
    small_set = get_small_set(size=10, seed=42)
    results = []
    configs = []
    # create a grid of configs
    for rep_width in [3, 6, 9]:
        for rec_smooth in [5, 7, 9]:
            for evec_smooth in [7, 9, 11]:
                configs.append({
                    'rep_width': rep_width,
                    'rec_smooth': rec_smooth,
                    'evec_smooth': evec_smooth,
                    'delay_steps': 4,
                })
    
    # Create a list of all (track, config) argument pairs
    tasks = list(itertools.product(small_set, configs))

    # use pqdm for parallelization
    # The function `all_feat_combo_scores` takes (track, config) as arguments.
    # We use argument_type='args' to unpack each tuple from `tasks`.
    results = pqdm(tasks, all_feat_combo_scores, n_jobs=8, argument_type='args', desc='Processing configs')
    
    # save results
    pd.concat(results).to_feather('lsd_config_selection_results.feather')
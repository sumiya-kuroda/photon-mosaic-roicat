from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt

from roicat import pipelines, util, helpers, data_importing, visualization
from photon_mosaic_roicat.notification import slack_bot
from photon_mosaic_roicat.roicat_helpers import io, generate_report

PIPELINES = {
    'tracking': pipelines.pipeline_tracking,
}

def constuct_roicat_params(config=None, subject:str = ''):
    if config is not None:
        processed_data_base = Path(config["processed_data_base"]).resolve()

    dir_data = processed_data_base / subject
    dir_save = dir_data / 'funcimg_tracked'
    dir_save.mkdir(parents=True, exist_ok=True)

    return dir_data, dir_save

def run_roicat_with_monitoring(
    pipeline_name: str = 'tracking',
    path_params: str = '',
    dir_data: str = '',
    dir_save: str = '',
    subject:str = ''
):
    """
    Call a pipeline with the specified parameters.
    """

    # Load in parameters to use
    if path_params is not None:
        params = helpers.yaml_load(path_params)
        params.pop("processed_data_base", None) # Because this is specific to this pipeline
    else:
        print(f"WARNING: No parameters file specified. Using default parameters for pipeline '{pipeline_name}'")
        params = {}

    # These lines are for safety, to make sure that all params are present and valid
    params_defaults = util.get_default_parameters(pipeline=pipeline_name)
    params = helpers.prepare_params(params=params, defaults=params_defaults)

    # User specified directories
    def inplace_update_if_not_none(d, key, value):
        if value is not None:
            d[key] = value
    inplace_update_if_not_none(params['data_loading'], 'dir_outer', dir_data)
    inplace_update_if_not_none(params['results_saving'], 'dir_save', dir_save)

    try:
        if params['data_loading']['data_kind'] == 'data_VRABCD':
            custom_data = load_VRABCD(params)
            # inplace_update_if_not_none(params['results_saving'], 'dir_save', 'data_suite2p') # because we use suite2p
            results, run_data, params = PIPELINES[pipeline_name](params=params, custom_data=custom_data) # Run pipeline
        else:
            results, run_data, params = PIPELINES[pipeline_name](params=params, custom_data=None) # Run pipeline

        _, idx_original_aligned = align_rois(str(dir_save), 'roicat_aligned_ROIs.npy')
        plot_cell_tracking(idx_original_aligned, Path(dir_save) / 'visualization' / 'tracked_cells.png')
        msg = f"✅ ROICaT job for 🐭 subject {subject} completed successfully! See attached reports."

        # Generate PDF report
        generate_report.generate_roicat_report(dir_save)
        is_pdfmade = True
    except Exception as e:
        msg = f"❌ ROICaT job for subject {subject} failed. Error message: {e}"
        is_pdfmade = False

    print(msg)
    if is_pdfmade:
        slack_bot.notify_slack_with_file_many(msg, [Path(dir_save) / 'roicat_report.pdf', Path(dir_save) / 'visualization' / 'FOV_clusters.gif'])
        # slack_bot.notify_slack_with_file(msg, Path(dir_save) / 'roicat_report.pdf')

    else:
        slack_bot.notify_slack(msg)

def load_VRABCD(params: dict):
    # this function load data from VR ABCD project
    
    paths_allStat = helpers.find_paths(
        dir_outer=params['data_loading']['dir_outer'],
        reMatch='stat.npy',
        reMatch_in_path=params['data_loading']['reMatch_in_path'],
        depth=6,
        find_files=True,
        find_folders=False,
        natsorted=True,
    )[:]
    paths_allOps = [str(Path(path).resolve().parent / 'ops.npy') for path in paths_allStat][:]

    if len(paths_allStat) == 0:
        raise FileNotFoundError(f"No stat.npy files found in '{params['data_loading']['dir_outer']}'")
    
    # Sort suite2p data based on session ids
    paths_allStat = io.natsort_by_sesids(paths_allStat)
    paths_allOps = io.natsort_by_sesids(paths_allOps)

    print(f"Found the following stat.npy files:")
    [print(f"    {path}") for path in paths_allStat]
    print(f"Found the following corresponding ops.npy files:")
    [print(f"    {path}") for path in paths_allOps]

    ## Import data
    data = data_importing.Data_suite2p(
        paths_statFiles=paths_allStat[:],
        paths_opsFiles=paths_allOps[:],
        verbose=params['general']['verbose'],
        **{**params['data_loading']['common'], **params['data_loading']['data_suite2p']},
    )
    assert data.check_completeness(verbose=False)['tracking'], f"Data object is missing attributes necessary for tracking."

    return data

def generate_roicat_FOVs(results_all: dict):
    FOV_clusters = visualization.compute_colored_FOV(
        spatialFootprints=[r.power(1.0) for r in results_all['ROIs']['ROIs_aligned']],  ## Spatial footprint sparse arrays
        FOV_height=results_all['ROIs']['frame_height'],
        FOV_width=results_all['ROIs']['frame_width'],
        labels=results_all["clusters"]["labels_bySession"],
    )

    return [(f * 255).astype(np.uint8) for f in FOV_clusters]
    # helpers.save_gif(
    #     array=helpers.add_text_to_images(
    #         images=[(f * 255).astype(np.uint8) for f in FOV_clusters], 
    #         text=[[f"{ii}",] for ii in range(len(FOV_clusters))], 
    #         font_size=3,
    #         line_width=10,
    #         position=(30, 90),
    #     ), 

def align_rois(dir_save: str,
               filename=None,
               sessions_to_align=None, alignment_method='F', 
               use_iscell=False,
               force_reload=False):
    '''Align the neural data according to the ROI they belong to, 
    For more details look at the ROICaT documentation https://roicat.readthedocs.io/en/latest/index.html.'''
    
    # Look for already aligned labels
    if filename is not None and os.path.exists(os.path.join(dir_save, filename)) and not force_reload:
        print('Aligned clusters already found. Loading...')
        idx_original_aligned = np.load(os.path.join(dir_save, filename))
        return None, idx_original_aligned

    else:
        roicat_results = util.RichFile_ROICaT(path=dir_save + '/roicat.tracking.results_all.richfile').load()

        # Load neural data
        subject, paths_s2p = io.extract_neuroblueprint_from_roicat(roicat_results)
        print(f'Aligning neural data for {subject} using ROICaT data')
        neural_data, arecells = load_neural_data(paths_s2p, data_type=alignment_method)
    
        # Define UCIDs 
        labels_bySession = roicat_results['clusters']['labels_bySession']
        if sessions_to_align is None:
            roi_labels = [rois for rois in labels_bySession]  
        else:
            raise NotImplementedError('Session selection not implemented yet')

        # Update UCIDs with valid cells
        if use_iscell:
            raise NotImplementedError('iscell selection not implemented yet')
            # labels_iscell = roicat.util.mask_UCIDs_with_iscell(ucids=roi_labels, iscell=iscell)
        else:
            labels_iscell = roi_labels
        
        # Squeeze the labels to remove the unassigned labels (not necessary, but reduces the number of unique labels)
        labels_iscell = util.squeeze_UCID_labels(ucids=labels_iscell, return_array=True)  ## [(n_rois,)] * n_sessions

        # Align the data with the masked labels
        data_aligned_masked, idx_original_aligned = util.match_arrays_with_ucids(
            arrays=neural_data,  ## expects list (length n_sessions) of numpy arrays (shape (n_rois, n_timepoints))
            ucids=labels_iscell,  ## expects list (length n_sessions) of numpy arrays (shape (n_rois,))  OR   concatenated numpy array (shape (n_rois_total,))
            squeeze=True, return_indices=True
        )

        # make list to array
        idx_original_aligned = np.vstack(idx_original_aligned)

        # Save indices of aligned data 
        if filename is not None:
            np.save(os.path.join(dir_save, filename), idx_original_aligned)

    return data_aligned_masked, idx_original_aligned

def load_neural_data(ses_dirs: list, data_type='F'):
    # Load neural data 
    data = [[] for s in range(len(ses_dirs))]
    iscell = [[] for s in range(len(ses_dirs))]

    for s, sess in enumerate(ses_dirs):
        print(f'Loading neural data for session {str(sess)}')
        if data_type == 'F':
            datapath = sess / 'funcimg' / 'suite2p' / 'plane0' / 'F.npy'
        elif data_type == 'DF_F0':
            datapath = sess / 'funcimg' / 'suite2p' / 'plane0' / 'DF_F0.npy'
            if not os.path.exists(datapath):
                raise FileNotFoundError('The DF_F0.npy file does not exist in this directory.')
        else:
            raise KeyError('This is not a valid data format for ROI alignment.')
        data[s] = np.load(datapath)

        iscellpath = sess / 'funcimg' / 'suite2p' / 'plane0' / 'iscell.npy'
        iscell[s] = np.load(iscellpath)[:,0]

    return data, iscell

def calc_tracked_proportions(idx_original_aligned: np.array):
    # Compute: which cells were valid (non-NaN) at each timepoint
    valid = ~np.isnan(idx_original_aligned)

    # Only consider cells that were valid on the **first day**
    initial_valid = valid[0, :]  # shape: (812,)

    # Option 1
    # # Now, for each timepoint, count how many of those are still being tracked
    # tracked_counts = np.sum(valid[:, initial_valid], axis=1)

    # Option 2
    # once a cell becomes invalid, it stays invalid
    cumulative_valid = np.cumprod(valid[:, initial_valid], axis=0).astype(bool)
    tracked_counts = np.sum(cumulative_valid, axis=1)

    tracked_proportions = tracked_counts / np.sum(initial_valid)

    return tracked_proportions, tracked_counts

def plot_cell_tracking(idx_original_aligned: np.array,
                       saving_file_location=None):
    tracked_proportions, tracked_counts = calc_tracked_proportions(idx_original_aligned)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.plot(range(0, idx_original_aligned.shape[0]), tracked_proportions, marker='o', label='Proportion tracked', linewidth=2.5)
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 0.5, 1])  # Set y-axis ticks
    plt.xlabel("Session")
    plt.ylabel("Proportion of cells tracked")
    # xticks = list(range(0, 18, 5))
    # plt.xticks(xticks)

    # Annotate with absolute counts
    for i, (x, y) in enumerate(zip(range(1, 19), tracked_proportions)):
        count = tracked_counts[i]
        offset = -0.13 if i in [4, 6, 8] else 0.05  # position below for sessions 3, 4, 5
        plt.text(x-1, y + offset, str(count), ha='center', color='black')

    plt.tight_layout()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    if saving_file_location is not None:
        fig.savefig(saving_file_location)
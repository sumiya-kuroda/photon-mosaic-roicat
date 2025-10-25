import submitit
from submitit.helpers import CommandFunction
from roicat import pipelines, util, helpers, run_pipeline
from photon_mosaic_roicat.roicat_helpers import io, tracking_helpers
from .run_pm import run_pm

def run_roicat(pipeline='tracking'):
    # Run ROICaT
    ## Get subject ids
    config, config_path = io.load_and_process_config(reset_config=False)
    subject_paths = io.get_subject_path(config)

    for subject in subject_paths:
        executor_roicat = submitit.AutoExecutor(folder="../.submitit")
        executor_roicat.update_parameters(
            slurm_partition="gpu",
            slurm_job_name="roicat",
            slurm_time="4:00:00",
            mem_gb=128,
            slurm_gres="gpu:1",
        )
        dir_data, dir_save = tracking_helpers.constuct_roicat_params(config, subject)
        job_roicat = executor_roicat.submit(run_pipeline, pipeline, str(config_path), str(dir_data), str(dir_save))
        print("Submitted job_id:", job_roicat.job_id)

        # monitor_roicat_and_notify(job_roicat)

if __name__ == '__main__':
    run_roicat()
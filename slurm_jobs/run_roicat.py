import submitit
from photon_mosaic_roicat.roicat_helpers import io, tracking_helpers

def run_roicat(pipeline='tracking', after_this_job:submitit.Job =None):
    # Run ROICaT
    ## Get subject ids
    config, config_path = io.load_and_process_config(reset_config=False)
    subject_paths = io.get_subject_path(config)

    jobs_roicat = []
    for subject in subject_paths:
        executor_roicat = submitit.AutoExecutor(folder="../.submitit")
        if after_this_job is None:
            executor_roicat.update_parameters(
                slurm_partition="gpu",
                slurm_job_name="roicat",
                slurm_time="4:00:00",
                mem_gb=128,
                slurm_gres="gpu:1",
            )
        else:
            executor_roicat.update_parameters(
                slurm_partition="gpu",
                slurm_job_name="roicat",
                slurm_time="4:00:00",
                mem_gb=128,
                slurm_gres="gpu:1",
                slurm_additional_parameters={
                    "dependency": f"afterok:{str(after_this_job.job_id)}",
                },
            )
 
        dir_data, dir_save = tracking_helpers.constuct_roicat_params(config=config, subject=subject)
        # tracking_helpers.run_roicat_with_monitoring(pipeline, str(config_path), str(dir_data), str(dir_save), subject)
        job_roicat = executor_roicat.submit(tracking_helpers.run_roicat_with_monitoring, pipeline, str(config_path), str(dir_data), str(dir_save), subject)
        if after_this_job is None:
            print("Submitted job_id:", job_roicat.job_id)
        else:
            print(
                f"Waiting for submission jobs ({after_this_job.job_id}) to complete before running collector job ({job_roicat.job_id})."
            )

        jobs_roicat.append(job_roicat)

    return jobs_roicat

if __name__ == '__main__':
    run_roicat()
    
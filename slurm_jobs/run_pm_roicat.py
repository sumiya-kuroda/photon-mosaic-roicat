import time
import submitit
from submitit.helpers import CommandFunction
from roicat import pipelines, util, helpers, run_pipeline
from photon_mosaic_roicat.notification import slack_bot
from photon_mosaic_roicat.roicat_helpers import io, tracking_helpers

def run_pm_roicat(pipeline='tracking'):
    # Run photon-mosaic
    executor_pm = submitit.AutoExecutor(folder="./submitit")
    executor_pm.update_parameters(
        slurm_partition="cpu",
        slurm_job_name="pm",
        slurm_time="72:00:00",
        mem_gb=4
    )

    job_pm = executor_pm.submit(CommandFunction(["photon-mosaic", "--jobs", "10", "--rerun-incomplete"])) # no space!
    # job_pm = executor_pm.submit("bash __run_pm.sh") #TODO: call via Python (not implemented)
    print("Submitted job_id:", job_pm.job_id)


    # Run ROICaT
    ## Get subject ids
    config, config_path = io.load_and_process_config(reset_config=False)
    subject_paths = io.get_subject_path(config)

    for subject in subject_paths:
        executor_roicat = submitit.AutoExecutor(folder="./roicat")
        executor_roicat.update_parameters(
            slurm_partition="gpu",
            slurm_job_name="roicat",
            slurm_time="4:00:00",
            mem_gb=128,
            slurm_gres="gpu:1",
            slurm_additional_parameters={
                "dependency": f"afterok:{str(job_pm.job_id)}",
            },
        )
        dir_data, dir_save = tracking_helpers.constuct_roicat_params(config, subject)
        job_roicat = executor_roicat.submit(run_pipeline, pipeline, str(config_path), str(dir_data), str(dir_save))
        print(
            f"Waiting for submission jobs ({job_pm.job_id}) to complete before running collector job ({job_roicat.job_id})."
        )
        # overwrite=True?
        # wait_and_notify(job_pm)


def wait_and_notify(job: submitit.Job, interval=30):
    print(f"Start monitoring job {job.job_id} ...")
    while not job.done():
        print(f"Still running...")
        time.sleep(interval)

    try:
        result = job.result()  # also raises if job failed
        msg = f"✅ Job {job.job_id} completed successfully!"
    except Exception as e:
        msg = f"❌ Job {job.job_id} failed: {e}"

    print(msg)
    slack_bot.notify_slack(msg)

if __name__ == '__main__':
    run_pm_roicat()
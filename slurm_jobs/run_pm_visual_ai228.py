import time
import submitit
from submitit.helpers import CommandFunction
from photon_mosaic_roicat.notification import slack_bot
from photon_mosaic_roicat.slurm_helper import monitor_pm_and_notify
def run_pm():
    # Run photon-mosaic
    executor_pm = submitit.AutoExecutor(folder="../.submitit")
    executor_pm.update_parameters(
        slurm_partition="gpu_lowp",
        slurm_job_name="pm",
        slurm_time="72:00:00",
        mem_gb=4
    )

    job_pm = executor_pm.submit(CommandFunction(["photon-mosaic", "--config", "/ceph/mrsic_flogel/public/projects/SuKu_20250111_FeedbackRF/configPhotonMosaicVisualInference_ai228.yaml", "--jobs", "30", "--latency-wait", "30", "--rerun-incomplete"])) # no space!
    #TODO: call via Python (not implemented)
    print("Submitted job_id:", job_pm.job_id)
    monitor_pm_and_notify(job_pm)

    return job_pm

if __name__ == '__main__':
    run_pm()
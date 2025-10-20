import time
import submitit
from submitit.helpers import CommandFunction

def run_pm():
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
    wait_and_notify(job_pm)


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
    # notify

if __name__ == '__main__':
    run_pm()
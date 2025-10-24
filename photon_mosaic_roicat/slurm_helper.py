import time
import submitit
from photon_mosaic_roicat.notification import slack_bot

def monitor_pm_and_notify(job: submitit.Job, interval=30):
    print(f"Start monitoring job {job.job_id} ...")
    while not job.done():
        print(f"Still running...")
        time.sleep(interval)

    with open(job.paths.stderr, "r") as f:
        lines = f.readlines()
        last_line = lines[-1].strip() if lines else None

    if "successfully" in last_line:
        msg = f"✅ photon-mosaic job {job.job_id} completed successfully!"
    else:
        msg = f"❌ photon-mosaic job {job.job_id} failed. Check the log"

    print(msg)
    slack_bot.notify_slack(msg)
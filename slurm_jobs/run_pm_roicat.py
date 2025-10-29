import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_pm import run_pm
from run_roicat import run_roicat

def run_pm_roicat(pipeline='tracking'):
    # Run photon-mosaic
    job_pm = run_pm()

    # Run ROICaT
    jobs_roicat = run_roicat(after_this_job=job_pm)

if __name__ == '__main__':
    run_pm_roicat()
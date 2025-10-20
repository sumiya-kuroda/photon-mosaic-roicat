import importlib.resources as pkg_resources
import logging
import os
from datetime import datetime
from pathlib import Path
import yaml
import re

# Adapted from photon-mosaic
def ensure_default_config(reset_config=False):
    """Ensure the default config file exists, create if missing or
    reset requested.

    Parameters
    ----------
    reset_config : bool
        Whether to reset the config file to defaults

    Returns
    -------
    Path
        Path to the default config file
    """
    logger = logging.getLogger(__name__)
    default_config_dir = Path.home() / ".photon_mosaic"
    default_config_path = default_config_dir / "roicat_config.yaml"

    if not default_config_path.exists() or reset_config:
        logger.debug("Creating default config file")
        default_config_dir.mkdir(parents=True, exist_ok=True)
        source_config_path = pkg_resources.files("photon_mosaic_roicat").joinpath(
            "roicat_helpers", "roicat_config.yaml"
        )
        with (
            source_config_path.open("rb") as src,
            open(default_config_path, "wb") as dst,
        ):
            dst.write(src.read())

    return default_config_path


def load_and_process_config(reset_config=False):
    """Load configuration file and apply CLI overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments

    Returns
    -------
    tuple[dict, Path]
        Configuration dictionary and path to config file used
    """
    logger = logging.getLogger(__name__)

    # Ensure default config exists
    default_config_path = ensure_default_config(reset_config)

    # Determine which config to use
    logger.debug("Using default config file")
    config_path = default_config_path

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.debug(f"Loaded config from {config_path}:")
    # logger.debug(f"use_slurm: {config.get('use_slurm', 'NOT SET')}")
    # logger.debug(f"slurm config: {config.get('slurm', 'NOT SET')}")

    # Process paths
    processed_data_base = Path(config["processed_data_base"]).resolve()

    # Append derivatives to the processed_data_base if
    # it doesn't end with /derivatives
    if processed_data_base.name != "derivatives":
        processed_data_base = processed_data_base / "derivatives"
    config["processed_data_base"] = str(processed_data_base)

    # Update default config file if it's the one being used
    update_default_config(
        default_config_path, processed_data_base
    )

    return config, config_path

def update_default_config(config_path, processed_data_base):
    """Update the default config file with new base paths while
    preserving comments.

    Parameters
    ----------
    config_path : Path
        Path to the config file to update
    raw_data_base : Path
        Raw data base path
    processed_data_base : Path
        Processed data base path
    """
    with open(config_path, "r") as f:
        config_lines = f.readlines()
    with open(config_path, "w") as f:
        for line in config_lines:
            if line.startswith("processed_data_base:"):
                f.write(f"processed_data_base: {processed_data_base}\n")
            else:
                f.write(line)

def setup_output_directories(processed_data_base):
    """Create necessary output directories for the pipeline.

    Parameters
    ----------
    processed_data_base : Path
        Base path for processed data

    Returns
    -------
    tuple[Path, Path, Path]
        Paths to output directory, logs directory, and configs directory
    """
    logger = logging.getLogger(__name__)

    output_dir = processed_data_base / "photon-mosaic-roicat"
    logger.debug(f"Creating output directory: {output_dir}")

    logs_dir = output_dir / "logs"
    configs_dir = output_dir / "configs"

    # Create directories with explicit permissions (rwxr-xr-x)
    # This ensures SLURM jobs can access these directories
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o755)

    logs_dir.mkdir(exist_ok=True)
    os.chmod(logs_dir, 0o755)

    configs_dir.mkdir(exist_ok=True)
    os.chmod(configs_dir, 0o755)

    return output_dir, logs_dir, configs_dir


def save_timestamped_config(config, configs_dir):
    """Save configuration with timestamp for reproducibility.

    Parameters
    ----------
    config : dict
        Configuration dictionary to save
    configs_dir : Path
        Directory to save config files

    Returns
    -------
    tuple[str, Path]
        Timestamp string and path to saved config file
    """
    import os

    logger = logging.getLogger(__name__)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f"roicat_config_{timestamp}.yaml"
    config_path = configs_dir / config_filename

    with open(config_path, "w") as f:
        logger.debug(f"Saving config to: {config_path}")
        yaml.dump(config, f)

    # Ensure the config file is readable by all users (rw-r--r--)
    # This is critical for SLURM jobs that need to read this file
    # when Snakemake re-runs itself in each SLURM job context
    os.chmod(config_path, 0o644)
    logger.debug("Set config file permissions to 644 for SLURM job access")

    return timestamp, config_path

def get_subject_path(config=None, processed_data_base=None, pattern='sub-'):
    if config is not None:
        processed_data_base = Path(config["processed_data_base"]).resolve()
    elif processed_data_base is not None:
        processed_data_base = Path(config["processed_data_base"]).resolve()

    candidate_datasets = [
        d.name
        for d in processed_data_base.iterdir()
        if d.is_dir() and re.match(pattern, d.name)
    ]

    return candidate_datasets
import logging

import matplotlib.pyplot as pl
import numpy as np
import polars as ps
import torch
import yaml

# setup logging
log_file_path = "./log/tiny.log"
logging.basicConfig(
    format="%(asctime)s - %(name)s: %(message)s",
    level=logging.DEBUG,
    filename=log_file_path,
)


def load_config(path: str) -> dict:
    """Load config from yaml file

    Args:
        path (str): path to yaml file

    Returns:
        dict: dictionary of config
    """
    try:
        logging.info(f"Loading config from {path}")
        config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        logging.info(f"Loaded config: {config}")
    except Exception as e:
        logging.error(f"Error loading config at path {path}: {e}")
        exit(1)

    return config

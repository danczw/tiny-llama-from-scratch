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

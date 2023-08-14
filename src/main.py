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


def load_data(path: str) -> tuple:
    """Load txt file and return

    Args:
        path (str): path to txt file

    Returns:
        tuple: tuple of (ind_to_char, char_to_ind, vocab, lines)
            with ind_to_char: dict of index to character
            with char_to_ind: dict of character to index
            with vocab: list of unique characters
            with lines: string of all characters
    """
    try:
        logging.info(f"Loading data from {path}")

        lines = open(path, "r").read()
        logging.info(f"Loaded {len(lines)} characters")

        vocab = sorted(list(set(lines)))
        logging.info(f"Loaded {len(vocab)} unique characters")

    except Exception as e:
        logging.error(f"Error loading data at path {path}: {e}")
        exit(1)

    # create a mapping of unique chars to integers
    ind_to_char = {i: ch for i, ch in enumerate(vocab)}
    char_to_ind = {ch: i for i, ch in enumerate(vocab)}

    return ind_to_char, char_to_ind, vocab, lines


def encode(s: str, char_ind_map: dict) -> list:
    """simple encoding function using characters and index mapping

    Args:
        s (str): string to encode
        char_ind_map (dict): character to index mapping

    Returns:
        list: list of encoded characters
    """
    return [char_ind_map[c] for c in s]


def decode(ind_list: list, ind_char_map: dict) -> str:
    """simple decoding function using index and character mapping

    Args:
        l (list): list of encoded characters
        ind_char_map (dict): index to character mapping

    Returns:
        str: decoded string
    """
    return "".join([ind_char_map[i] for i in ind_list])

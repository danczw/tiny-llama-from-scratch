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


def get_batches(
    data: torch.tensor,
    context_window: int,
    split: str = "train",
    train_size: float = 0.8,
    batch_size: int = 32,
) -> tuple:
    """Get batches of data

    Args:
        data (torch.tensor): torch tensor of encoded data
        split (str): split type (train, val, test)
        context_window (int): context window size
        train_size (float, optional): train size. Defaults to 0.8.
        batch_size (int, optional): batch size. Defaults to 32.

    Returns:
        tuple: tuple of (x, y) where x is the input and y is the target
    """
    # define validation and test size in percentage
    val_size = (1 - train_size) / 2
    test_size = 1 - train_size - val_size
    data_size = len(data)

    logging.info(f"Train size: {train_size}% - {int(data_size * train_size)}")
    logging.info(f"Val size: {val_size}% - {int(data_size * val_size)}")
    logging.info(f"Test size: {test_size}% - {int(data_size * test_size)}")
    logging.info(f"Data size: {data_size}")

    # split data into train, val, test
    train = data[: int(data_size * train_size)]
    val = data[int(data_size * train_size) : int(data_size * (train_size + val_size))]
    test = data[int(data_size * (train_size + val_size)) :]

    logging.info(f"Actual train size: {train.shape}")
    logging.info(f"Actual val size: {val.shape}")
    logging.info(f"Actual test size: {test.shape}")

    batch_data = train
    if split == "val":
        batch_data = val
    elif split == "test":
        batch_data = test

    # pick random starting point
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i : i + context_window] for i in ix]).long()
    y = torch.stack([batch_data[i + 1 : i + context_window + 1] for i in ix]).long()

    return x, y


def main():
    # load config
    config = load_config(path="./conf/config.yaml")

    # load data
    ind_to_char, char_to_ind, vocab, lines = load_data(path=config["data_file_path"])
    config["vocab_size"] = len(vocab)

    # encode data and create torch data set
    dataset = torch.tensor(encode(s=lines, char_ind_map=char_to_ind), dtype=torch.int8)
    logging.info(f"Encoded data shape: {dataset.shape}")

    # get batches
    X, Y = get_batches(
        data=dataset,
        context_window=config["context_window"],
        split="train",
        train_size=config["train_size"],
        batch_size=config["batch_size"],
    )

    print(
        [
            (decode(X[i].tolist(), ind_to_char), decode(Y[i].tolist(), ind_to_char))
            for i in range(len(X))
        ]
    )


if __name__ == "__main__":
    main()

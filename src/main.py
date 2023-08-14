import logging
import time

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
    data: torch.Tensor,
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
    data_size = len(data)
    val_size = (1 - train_size) / 2
    test_size = 1 - train_size - val_size

    # split data into train, val, test
    train = data[: int(data_size * train_size)]
    val = data[int(data_size * train_size) : int(data_size * (train_size + val_size))]
    test = data[int(data_size * (train_size + val_size)) :]

    logging.debug(f"Data size: {data_size}")
    logging.debug(f"Train size: {train_size}% - {int(data_size * train_size)}")
    logging.debug(f"Val size: {val_size}% - {int(data_size * val_size)}")
    logging.debug(f"Test size: {test_size}% - {int(data_size * test_size)}")

    # set batch data
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


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, dataset: torch.Tensor, config: dict) -> dict:
    """Evaluate loss on train and test set

    Args:
        model (torch.nn.Module): model to evaluate
        dataset (torch.tensor): dataset tensor of encoded data
        config (dict): config dictionary

    Returns:
        dict: dictionary of loss on train and test set
    """
    output = {}
    model.eval()

    # evaluate loss on train and test set
    for split in ["train", "test"]:
        losses = []
        for _ in range(10):
            # get batches and calculate loss for each batch
            xb, yb = get_batches(
                data=dataset,
                context_window=config["context_window"],
                split=split,
                train_size=config["train_size"],
                batch_size=config["batch_size"],
            )
            _, loss = model(xb, yb)
            losses.append(loss.item())
        # calculate mean loss across batches
        output[split] = np.mean(losses)

    logging.debug(f"Loss on train set: {output['train']}")
    logging.debug(f"Loss on test set: {output['test']}")

    model.train()
    return output


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: torch.Tensor,
    config: dict,
    scheduler=None,
) -> ps.DataFrame:
    """Train model

    Args:
        model (torch.nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer
        dataset (torch.tensor): dataset tensor of encoded data
        config (dict): config dictionary
        scheduler (torch.optim.lr_scheduler, optional): learning rate scheduler. Defaults to None.

    Returns:
        ps.DataFrame: dataframe of losses for each epoch
    """
    losses = []
    logging.info(f"Start training for {config['epochs']} epochs")

    for epoch in range(config["epochs"]):
        start_time = time.time()

        # reset gradients
        optimizer.zero_grad()

        # get batches
        xs, ys = get_batches(
            data=dataset,
            context_window=config["context_window"],
            split="train",
            train_size=config["train_size"],
            batch_size=config["batch_size"],
        )

        # forward pass
        logits, loss = model(xs, ys)
        # backward pass
        loss.backwards()
        # update parameters
        optimizer.step()

        if scheduler:
            scheduler.step()

        if epoch % config["log_interval"] == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model=model, dataset=dataset, config=config)
            losses += [x]

            epoch_log = f"Epoch: {epoch}"
            val_log = f"val loss {x['val']:.3f}"
            time_log = f"time {batch_time:.3f}"
            eta_log = f"ETA {batch_time * (config['epochs'] - epoch):.3f}"
            logging.info(" | ".join([epoch_log, val_log, time_log, eta_log]))

            if scheduler:
                logging.info(f"Learning rate: {scheduler.get_lr()}")

    logging.info(f"Training completed for {config['epochs']} epochs")
    logging.info(f"Final loss: {losses[-1]}")

    return ps.DataFrame(losses)


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


if __name__ == "__main__":
    main()

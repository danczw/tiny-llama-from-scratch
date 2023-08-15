import logging
import time

import matplotlib.pyplot as pl
import numpy as np
import polars as ps
import torch
import yaml
from torch import nn
from torch.nn import functional as F

# setup logging
log_file_path = "./log/tiny.log"
formatter = logging.Formatter("%(asctime)s - %(name)s: %(message)s")
logger = logging.getLogger("tinyLLaMa")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(log_file_path, mode="w")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger.addHandler(fh)


def load_config(path: str) -> dict:
    """Load config from yaml file

    Args:
        path (str): path to yaml file

    Returns:
        dict: dictionary of config
    """
    try:
        logger.info(f"Loading config from {path}")
        config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        logger.info(f"Loaded config: {config}")
    except Exception as e:
        logger.error(f"Error loading config at path {path}: {e}")
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
        logger.info(f"Loading data from {path}")

        lines = open(path, "r").read()
        logger.info(f"Loaded {len(lines)} characters")

        vocab = sorted(list(set(lines)))
        logger.info(f"Loaded {len(vocab)} unique characters")

    except Exception as e:
        logger.error(f"Error loading data at path {path}: {e}")
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
        data (torch.Tensor): torch tensor of encoded data
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
    test_size = val_size

    # split data into train, val, test
    train = data[: int(data_size * train_size)]
    val = data[int(data_size * train_size) : int(data_size * (train_size + val_size))]
    test = data[int(data_size * (train_size + test_size)) :]

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
def evaluate_loss(model: nn.Module, dataset: torch.Tensor, config: dict) -> dict:
    """Evaluate loss on train and test set

    Args:
        model (torch.nn.Module): model to evaluate
        dataset (torch.Tensor): dataset tensor of encoded data
        config (dict): config dictionary

    Returns:
        dict: dictionary of loss on train and test set
    """
    output = {}
    model.eval()

    # evaluate loss on train and test set
    for split in ["train", "val"]:
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

    logger.debug(f"Loss on train set: {output['train']:.3f}")
    logger.debug(f"Loss on test set: {output['val']:.3f}")

    model.train()
    return output


class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        """Forward pass of RMSNorm

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: normalized tensor
        """
        # frob norm is not the same as RMS. RMS = 1/sqrt(n) * frob norm
        ff_rms = torch.linalg.norm(x, dim=(1, 2)) * x[0].numel() ** -0.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        return self.scale[: x.shape[1], :].unsqueeze(0) * raw  # type: ignore


class RoPEAttention_wMask(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.w_q = nn.Linear(config["d_model"], config["d_model"], bias=False)
        self.w_k = nn.Linear(config["d_model"], config["d_model"], bias=False)
        self.w_v = nn.Linear(config["d_model"], config["d_model"], bias=False)

        self.multihead = nn.MultiheadAttention(
            config["d_model"], config["num_heads"], dropout=0.1, batch_first=True
        )
        self.R = self.get_rotary_matrix(config["context_window"], config["d_model"])

    def get_rotary_matrix(
        self, context_window: int, embedding_dim: int
    ) -> torch.Tensor:
        """Get rotary matrix

        Args:
            context_window (int): context window size
            embedding_dim (int): embedding dimension

        Returns:
            torch.Tensor: rotary matrix
        """
        # init rotation matrix
        R = torch.zeros(
            (context_window, embedding_dim, embedding_dim), requires_grad=False
        )
        # get rotation matrix for each position
        for pos in range(context_window):
            for i in range(embedding_dim // 2):
                # calc rotation angle
                theta = 10000.0 ** (-2.0 * (i - 1) / embedding_dim)
                m_theta = pos * theta
                # calc position in rotation matrix
                R[pos, 2 * i, 2 * i] = np.cos(m_theta)
                R[pos, 2 * i, 2 * i + 1] = -np.sin(m_theta)
                R[pos, 2 * i + 1, 2 * i] = np.sin(m_theta)
                R[pos, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
        return R

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass of RoPEAttention

        Args:
            x (torch.Tensor): input tensor

        Returns:
            tuple: tuple of (activations, attention weights)
        """
        b, m, d = x.shape

        # get queries, keys, values
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # add rotary embedding to queries and keys
        q_out = (torch.bmm(q.transpose(0, 1), self.R[:m, ...])).transpose(0, 1)
        k_out = (torch.bmm(k.transpose(0, 1), self.R[:m, ...])).transpose(0, 1)
        v_out = (torch.bmm(v.transpose(0, 1), self.R[:m, ...])).transpose(0, 1)

        # pass through multihead attention to get attention weights and activations
        activation, _attn_weights = self.multihead(
            q_out,
            k_out,
            v_out,
            attn_mask=nn.Transformer.generate_square_subsequent_mask(m),
            is_causal=True,
        )

        return activation


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf
    """

    def __init__(
        self,
        size: int,
    ):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)

        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta)

    def forward(self, x) -> torch.Tensor:
        """Forward pass of SwiGLU

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        # calculate swish gate based on https://arxiv.org/pdf/2002.05202v1.pdf
        swish_gate = self.linear_gate(x) * torch.sigmoid(
            self.beta * self.linear_gate(x)
        )
        out = swish_gate * self.linear(x)
        return out


class TinyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # embedding layer
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        # RMS normalization
        self.rms = RMSNorm((config["context_window"], config["d_model"]))
        # RoPE attention
        self.rope_attention = RoPEAttention_wMask(config)

        # linear layer with swiglu activation
        self.linear = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]),
            SwiGLU(config["d_model"]),
        )

        # simple linear layer as last layer
        self.last_linear = nn.Linear(config["d_model"], config["vocab_size"])

        logger.info(f"model params: {sum([m.numel() for m in self.parameters()])}")

    def forward(self, idx: int, targets: torch.Tensor = None) -> tuple:  # type: ignore
        """Forward pass of model

        Args:
            idx (int): index of input
            targets (torch.Tensor, optional): target tensor. Defaults to None.

        Returns:
            tuple: tuple of (logits, loss) if targets is not None else tuple of logits
        """
        # get embeddings
        x = self.embedding(idx)

        # one block of attention
        x = self.rms(x)  # rms pre-normalization
        x = x + self.rope_attention(x)

        x = self.rms(x)  # rms pre-normalization
        x = x + self.linear(x)

        # get logits
        logits = self.last_linear(x)

        # calculate loss if targets is not None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config["vocab_size"]), targets.view(-1)
            )
            return logits, loss

        else:
            return logits


def train(
    model: nn.Module,
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
    logger.info(f"Start training for {config['epochs']} epochs")

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
        loss.backward()
        # update parameters
        optimizer.step()

        # update learning rate
        if scheduler:
            scheduler.step()

        # log batch metrics
        if epoch % config["log_interval"] == 0 or epoch == config["epochs"] - 1:
            batch_time = time.time() - start_time
            x = evaluate_loss(model=model, dataset=dataset, config=config)
            losses += [x]

            epoch_log = f"Epoch: {epoch}"
            val_log = f"val loss {x['val']:.3f}"
            time_log = f"time {batch_time:.3f}"
            eta_log = f"ETA {batch_time * (config['epochs'] - epoch):.3f}"
            logger.info(" | ".join([epoch_log, val_log, time_log, eta_log]))

            if scheduler:
                logger.info(f"Learning rate: {scheduler.get_lr()}")

    logger.info(f"Training completed for {config['epochs']} epochs")
    logger.info(f"Final loss: {losses[-1]}")

    return ps.DataFrame(losses)


def generate(
    model: nn.Module, config: dict, ind_char_map: dict, max_new_tokens: int = 50
) -> list:
    """Generate new text

    Args:
        model (nn.Module): model to generate text
        config (dict): config dictionary
        ind_char_map (dict): index to character mapping
        max_new_tokens (int, optional): max number of new tokens to generate. Defaults to 50.

    Returns:
        list: list of generated text
    """
    idx = torch.zeros(5, 1).long()
    for _ in range(max_new_tokens):
        # call the model
        logits = model(idx[:, -config["context_window"] :])

        # get all the batches (1), last time step, all the logits
        last_time_step_logits = logits[:, -1, :]

        # softmax to get probabilities
        p = F.softmax(last_time_step_logits, dim=-1)

        # sample from the distribution to get next token
        idx_next = torch.multinomial(p, num_samples=1)

        # append the new token to the sequence
        idx = torch.cat((idx, idx_next), dim=-1)
    return [decode(x, ind_char_map) for x in idx.tolist()]


def main():
    # load config
    config = load_config(path="./conf/config.yaml")

    # load data
    ind_to_char, char_to_ind, vocab, lines = load_data(path=config["data_file_path"])
    config["vocab_size"] = len(vocab)

    # encode data and create torch data set
    dataset = torch.tensor(encode(s=lines, char_ind_map=char_to_ind), dtype=torch.int8)
    logger.info(f"Encoded data shape: {dataset.shape}")

    # get batches
    X, Y = get_batches(
        data=dataset,
        context_window=config["context_window"],
        split="train",
        train_size=config["train_size"],
        batch_size=config["batch_size"],
    )

    # create model and optimizer
    model = TinyModel(config=config)
    optimizer = torch.optim.Adam(model.parameters())

    # train model
    loss_df = train(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        config=config,
        scheduler=None,
    )

    # plot train and val loss
    pl.plot(loss_df["train"], label="train")
    pl.plot(loss_df["val"], label="val")
    pl.ylabel("Loss")
    pl.title("Train and Val Loss")
    pl.legend()
    pl.savefig(config["output_path"] + "/loss.png")

    generated_text = generate(
        model=model,
        config=config,
        ind_char_map=ind_to_char,
    )
    logger.info(f"Generated text: {generated_text}")


if __name__ == "__main__":
    main()

<h1 align="center">Tiny LLaMa from Scratch</h1>

<div align="center">
Tiny <a href="https://arxiv.org/pdf/2302.13971.pdf">LLaMa</a> LLM implemented from scratch using PyTorch and Polars.
</div>

Based on and adapted from Brian Kitanos' [Llama from scratch (or how to implement a paper without crying)](https://blog.briankitano.com/llama-from-scratch/)

<br>

----

<br>

# Setup

All configurations are located at `conf/config.yml`. The default configuration is set to run the model on the TinyShakespear dataset. Get a version of TinyShakespear, e.g. from [here](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt). The file should be located under `data/input.txt`.

## Install dependencies

Currently, there is a documented issue with installing `pytorch` via poetry: [Instructions for installing PyTorch #6409 - python-poetry/poetry (github.com)](https://github.com/python-poetry/poetry/issues/6409). Therefore, we recommend to install `pytorch` first via:

```bash
poetry add torch --source torch_cpu
```

Then, install the remaining dependencies via:

```bash
poetry install
```

## Run

Run tiny LLaMa training cycle via:

```bash
poetry run python main.py
```
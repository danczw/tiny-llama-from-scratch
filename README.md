<h1 align="center">Tiny LLaMa from Scratch</h1>

<div align="center">
Tiny LLaMa LLM implemented from scratch using PyTorch and Polars.

<br>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
</div>

<br>

----

<br>

This repository contains a minimal implementation of the [LLaMa](https://arxiv.org/pdf/2302.13971.pdf) language model. It is implemented using PyTorch and Polars and trained on the TinyShakespear dataset. The repo is intended as a minimal example for a LLaMa implementation for educational purposes.

Based, adapted and extended on Brian Kitanos' `Llama from scratch (or how to implement a paper without crying)` ([Blog Version](https://blog.briankitano.com/llama-from-scratch/) | [GitHub Version](https://github.com/bkitano/llama-from-scratch)).

<br>

----

<br>

# Setup

All configurations are located at `conf/config.yml`. The default configuration is set to run the model on the TinyShakespear dataset. Get a version of TinyShakespear, e.g. from [here](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt). The file should be located under `data/input.txt`.

## Install dependencies

Install dependencies via:

```bash
poetry install
```

Note:
Currently, there is a documented issue with installing `pytorch` via poetry: [Instructions for installing PyTorch #6409 - python-poetry/poetry (github.com)](https://github.com/python-poetry/poetry/issues/6409), forcing installation for specific PyTorch features (e.g., CPU only version) to resolve dependencies using the wheel URL, which takes up to several minutes.

## Run

Run tiny LLaMa training cycle via:

```bash
poetry run python main.py
```

Logs are written to `log/`, while loss visualizations are written to `data/loss.png`.
# LLMICL (Huawei Noah's Ark Paris)

## Overview
This repository contains an adaptation of the official implementation of the paper:

   [LLMs learn governing principles of dynamical systems, revealing an in-context neural scaling law](http://arxiv.org/abs/2402.00795)

This implementation is used for the following paper:

   [Can LLMs predict the convergence of Stochastic Gradient Descent?](https://openreview.net/forum?id=FraikHzMu9)

## Directory structure 
An overview of the repository's structure and contents (inside /src/llmicl/):

- `interfaces/`: Contains classes for the ICLTrainer. Objects of type ICLTrainer have methods to update the LLM context with a time series, call the LLM, collect the predicted PDFs, compute statistics, visualize the results, build the associated Markov chain kernel, etc.
- `legacy/`: The legacy code.
- `matrix_completion/`: utils for the OT matrix completion techniques.
- `rl_helpers/`: utils for the RL time series.


## Installation

- create a conda environment (or pyvirtualenv):
```
conda create -n LLMICL python=3.9
```
```
python3.9 -m venv venv
```
- activate the environment:
```
conda activate LLMICL
```
```
source venv/bin/activate
```
- install the package
```
pip install -e .
```

- (for developers, install the optional dependencies)
```
pip install -e .[dev]
```

## Getting started

see the notebook: src/llmicl/basic_example.ipynb
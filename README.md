# A reproduction of Growing Neural Cellular Automata

Play around with the trained models and read about the alorithms at [Reproducing Growing Neural Cellular Automata](https://jensjepsen.github.io/growing-neural-cellular-automata/).

This repository contains code to reproduce the work by **Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson and Michael Levin**, which can be found at [Growing Neural Cellular Automata
](https://distill.pub/2020/growing-ca/).


The repo consists of two parts:
* PyTorch code to train the models and export them to ONNX
* A frontend in VueJS+TypeScript to serve and interact with the exported ONNX models, entirely in the browser.

## Training models
Setup venv and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

and to train all models, run:

```bash
python train.py train-all
```

or a single model:

```
python train.py train --image-num 0
```
where `--image-num` is the index of the emoji in [emoji.png](./images/emoji.png)

to see all training options run:
```bash
python train.py train --help
```

or

```bash
python train.py train-all --help
```

## Running and building frontend
Install dependencies:

```
cd frontend
npm install .
```

Serve frontend locally:

```
cd frontend
npm run serve
```

## Running tests
```bash
python -m pytest
```
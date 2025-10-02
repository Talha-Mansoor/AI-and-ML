# Mini Project 2

This project uses Python 3.12.

## Project instructions

Project instructions can be found [here](doc/instructions.pdf).

## Installation

The Python virtual environment for this project can be created as before.
A dedicated Python virtual environment should be created for this respository.
Dependencies should be installed by running `pip install -r requirements.txt`.

## File structure

This repository contains the following files:

```
.
├── README.md
├── data                         # Dataset downloaded from Canvas. DO NOT ADD THIS FOLDER TO GIT.
│   └── formula
│       ├── test_img.npy
│       ├── test_labels.npy
│       ├── train_img.npy
│       └── train_labels.npy
├── doc
│   ├── figure
│   │   ├── formula_000.png
│   │   ├── formula_001.png
│   │   ├── formula_002.png
│   │   ├── formula_003.png
│   │   └── formula_004.png
│   ├── main.tex
│   └── report.tex
├── p2
│   ├── __init__.py
│   └── formula.py               # Starter code for loading dataset.
├── requirements.txt
└── test
    └── test_data.py             # Unit test for testing dataset loading.
```

### Downloading the dataset

The dataset can be downloaded as a .zip file from [this link](https://usaskca1-my.sharepoint.com/:u:/g/personal/xla804_usask_ca/EQU3VgzahwNIlTYK2E7YqHABlbLigjHtKtddoct0AZb17w?e=rq6paK).
You can then extract this .zip file to match the file structure above.

Do not add the folder `./data` to git.
The git index will be overwhelmed with the relatively large file sizes.

## Running training loops

Both baseline and improved models should be implemented using PyTorch and each model should be optimized with a gradient descent algorithm.
This code can be fully implemented in `p2/formula.py`.

The baseline model training loop should be run from the command line with

```
python -m p2.formula train-baseline
```

The improved model training loop should be run from the command line with

```
python -m p2.formula train-improved
```

Logging numbers to console is sufficient for this project and probably the easiest approach.

This repository also contains debug configurations for both training loops in `.vscode/launch.json`.

Alternatively, Jupyter notebooks can be provided as well for training both models.
The Jupyter notebooks can be submitted without any cell output content to reduce file size.
However, these notebooks must execute correctly for the implementation to be considered reproducible.
How to reproduce all experiments should be documented in the project report.

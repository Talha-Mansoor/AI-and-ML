# Mini Project 3

This project uses Python 3.12.

## Project instructions

Project instructions can be found [here](doc/instructions.pdf).

## Dataset download

The dataset can be downloaded [here](https://usaskca1-my.sharepoint.com/:u:/g/personal/xla804_usask_ca/EWwJXLaHc2tCq8EvgNtTU1wBCb-8je5T4Nk7f0gMOY4WNg?e=cfax33).

## Installation

The Python virtual environment for this project can be created as before.
A dedicated Python virtual environment should be created for this respository.
Dependencies should be installed by running `pip install -r requirements.txt`.

## File structure

This repository contains the following files:

```
.
├── KMeansImageExample.ipynb   # Image visualization example code.
├── README.md
├── data
│   ├── mnist_imgs.npy         # MNIST dataset files
│   └── mnist_labels.npy       # MNIST dataset files
├── doc
│   └── figure
├── kmeans
│   ├── __init__.py
│   └── kmeans.py              # K-Means implementation. You need to implement the algorithm in this module.
├── requirements.txt
└── test
    └── test_kmeans.py         # Unit test for K-Means clustering result class.
```

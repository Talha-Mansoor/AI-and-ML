Here’s a **README.md** you can place at the root of your repository. It documents all three projects (`iris_knn`, `formula`, and `kmeans`) with context, setup, and usage instructions.

---

# CMPT 423 Projects Repository

This repository contains three different mini projects implemented for **CMPT 423 (Machine Learning / AI course)**. Each project is organized into its own folder with corresponding code, instructions, and documentation.

---

## 📂 Project 1: Iris KNN Classifier

**Goal:** Implement a **k-Nearest Neighbors (kNN)** classifier on the classic **Iris dataset**.

### Features

* Implements helper functions for distance computation, neighbor search, and confusion matrix.
* Uses the Iris dataset with train/test split.
* Predicts species (`versicolor`, `virginica`, `setosa`) based on sepal and petal measurements.

### Files

* `iris_knn.py` – Python implementation of the kNN classifier.
* `IrisKNN.ipynb` – Jupyter notebook for experimentation and visualization.

### Usage

```bash
python iris_knn.py
```

Or open the Jupyter notebook:

```bash
jupyter notebook IrisKNN.ipynb
```

---

## 📂 Project 2: Formula Recognition (Mini Project 2)

**Goal:** Recognize **arithmetic formulas from images** using deep learning (PyTorch). Dataset contains synthetic images of formulas such as `2+0`, `3/3`, `4-1`, etc.

### Features

* **Baseline model:** Logistic regression using linear layers.
* **Improved model:** Convolutional Neural Network (CNN).
* Dataset derived from **EMNIST** digits and operator symbols.
* Supports reproducibility and performance comparison.

### Files

* `formula.py` – Contains dataset loader, baseline model, CNN model, training & evaluation functions.
* `instructions.pdf` – Assignment instructions for Mini Project 2.

### Usage

Train the baseline model:

```bash
python -m formula train-baseline
```

Train the improved CNN model:

```bash
python -m formula train-improved
```

Compare performance of both models:

```bash
python -m formula performance-diff
```

---

## 📂 Project 3: K-Means Clustering on MNIST Digits (Mini Project 3)

**Goal:** Implement **K-Means clustering** on a subset of the MNIST dataset (7,000 images) and analyze distortion across different cluster sizes.

### Features

* Custom **KMeans** class implemented in NumPy.
* Support for repeated runs to avoid poor centroid initialization.
* Saves clustering results (centroids, memberships, distortion).
* Includes helper functions for distortion computation and centroid visualization.
* Final deliverable includes a report analyzing distortion vs `k`, and centroid visualizations.

### Files

* `kmeans.py` – Implementation of the K-Means algorithm with CLI options.
* `instructions.pdf` – Assignment instructions for Mini Project 3.

### Usage

Run clustering:

```bash
python -m kmeans --filename clustering.json --k 10 --max-iterations 100 --epsilon 0.001 --repeats 5
```

---

## ⚙️ Requirements

* Python 3.8+
* Libraries:

  * `numpy`
  * `pandas`
  * `torch`
  * `scipy`
  * `click`
  * `matplotlib` (for visualization in notebooks)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📑 Reports

Each project includes a LaTeX-formatted report (`report.pdf`) stored in its project folder:

* Project 2: Formula Recognition
* Project 3: K-Means Clustering

---

## 👨‍💻 Authors

Developed as part of **CMPT 423-820** coursework (Winter 2025).

---

Do you want me to **merge this into a single README.md file** with code block formatting, or create **separate README files inside each project folder** as well?


Medical References:
1. None — DOI: file-467Snh1cUYSEjPxP8mSyfY
2. None — DOI: file-Bnadjn6dyzx2aWSp8yEKEr
3. None — DOI: file-AW8E4JrTsWbZyVFs5AXUWr
4. None — DOI: file-RZi343n2UamLxYpGi6nMQx
5. None — DOI: file-2QEjWyexSFsg3U2TeTqwHX
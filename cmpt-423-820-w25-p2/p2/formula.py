import os

import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import sem
from torch.utils.data import DataLoader, TensorDataset

# To run the code just run in terminal

# for baseline model
# python -m formula train-baseline

# for improved model
# python -m formula train-improved


# check if gpu is available and set the device for gpu acceleration otherwise cpu will be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

"""
Below are the parameters we chose to improve our model. Change them to desired
values as needed for reproducibility. These parameters were chosen for the following reasons:

1. NUM_OF_EPOCHS: Chosen to train long enough for good performance without overfitting.
2. LEARNING_RATE: Set to make learning stable and efficient without being too fast or too slow.
3. CURR_BATCH_SIZE: Picked to balance training speed and how well the model generalizes.
4. OPTIMIZER TYPE: Selected to help the model learn faster and more effectively. Currently set
to SGD Optimizer for baseline and ADAM for improved model.
To Change the Optimizer change the optimization_method argument in train_baseline and train_improved functions.

TRAIN_SHUFFLE_SETTING should be set to False if testing for reproducibility to
ensure consistent data order across runs. Otherwise, set to True for better training performance.

"""
NUM_OF_EPOCHS = 10
LEARNING_RATE = 0.001
CURR_BATCH_SIZE = 64
TRAIN_SHUFFLE_SETTING = False


# ==========================
# DATA LOADING & PREPROCESSING
# ==========================
"""
As mentioned in project instructions, the dataset consists of images and labels representing arithmetic formulas
The images are converted from 8-bit (0-255) to float values (0-1) for easier processing.

Attributes:
    train_imgs (np.ndarray): Normalized training images
    test_imgs (np.ndarray): Normalized testing images
    train_labels_str (np.ndarray): String-based labels for training set
    test_labels_str (np.ndarray): String-based labels for testing set

"""


class FormulaData:
    """Load and access the formula dataset"""

    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), "../data/formula")

        train_imgs_uint8 = np.load(os.path.join(data_dir, "train_img.npy"))
        test_imgs_uint8 = np.load(os.path.join(data_dir, "test_img.npy"))
        self.train_imgs = train_imgs_uint8.astype(np.float32) / 255.0
        self.test_imgs = test_imgs_uint8.astype(np.float32) / 255.0

        train_labels_fn = os.path.join(data_dir, "train_labels.npy")
        test_labels_fn = os.path.join(data_dir, "test_labels.npy")
        self.train_labels_str = np.load(train_labels_fn)
        self.test_labels_str = np.load(test_labels_fn)


def label_to_indices(labels_str):
    """Convert formula labels (e.g., '2a3') to numerical indices

    first and last characters are numbers (0-9).
    middle character represents an arithmetic operator (+, -, *, /)

     Args:
        labels_str (list of str): List of string-based formula labels

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Three arrays representing:
            - first digit (0-9)
            - operator (addition, subtraction, multiplication, division)
            - second digit (0-9)
    """
    num_dict = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
    }
    operator_dict = {"a": 0, "s": 1, "m": 2, "d": 3}
    y1 = np.array([num_dict[label[0]] for label in labels_str])
    y2 = np.array([operator_dict[label[1]] for label in labels_str])
    y3 = np.array([num_dict[label[2]] for label in labels_str])
    return y1, y2, y3


def load_and_process_data(batch_size=64):
    """
    Load and prepare the dataset for training and testing.

    - Loads images and labels
    - Converts them to PyTorch tensors
    - Wraps them in DataLoaders for batch processing

    Args:
        batch_size (int, optional): Number of images per batch. Default is 64

    Returns:
        Tuple[DataLoader, DataLoader]: DataLoaders for training and testing

    """
    data = FormulaData()
    train_y1, train_y2, train_y3 = label_to_indices(data.train_labels_str)
    test_y1, test_y2, test_y3 = label_to_indices(data.test_labels_str)

    # Convert to PyTorch tensors
    train_imgs_tensor = torch.tensor(data.train_imgs, dtype=torch.float32)
    test_imgs_tensor = torch.tensor(data.test_imgs, dtype=torch.float32)
    train_y1_tensor = torch.tensor(train_y1, dtype=torch.long)
    train_y2_tensor = torch.tensor(train_y2, dtype=torch.long)
    train_y3_tensor = torch.tensor(train_y3, dtype=torch.long)
    test_y1_tensor = torch.tensor(test_y1, dtype=torch.long)
    test_y2_tensor = torch.tensor(test_y2, dtype=torch.long)
    test_y3_tensor = torch.tensor(test_y3, dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(
        train_imgs_tensor, train_y1_tensor, train_y2_tensor, train_y3_tensor
    )
    test_dataset = TensorDataset(
        test_imgs_tensor, test_y1_tensor, test_y2_tensor, test_y3_tensor
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=TRAIN_SHUFFLE_SETTING
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


# ==========================
# BASELINE MODEL (LOGISTIC REGRESSION)
# ==========================
class LinearModel(nn.Module):
    """
    Baseline Logistic Regression Model.

    this model applies a linear transformation to the input image features and produces three separate outputs for predicting:

    - The first digit (0-9)
    - The operator (+, -, *, /)
    - The second digit (0-9)

    Azttributes:
        fc_y1 (nn.Linear): Fully connected layer for first digit
        fc_y2 (nn.Linear): Fully connected layer for operator
        fc_y3 (nn.Linear): Fully connected layer for second digit
    """

    def __init__(self, input_size):
        super().__init__()
        self.fc_y1 = nn.Linear(input_size, 10)  # First digit classification
        self.fc_y2 = nn.Linear(input_size, 4)  # Operator classification
        self.fc_y3 = nn.Linear(input_size, 10)  # Second digit classification

    def forward(self, batch):
        batch = batch.view(batch.size(0), -1)  # Flatten images
        y1_logits = self.fc_y1(batch)
        y2_logits = self.fc_y2(batch)
        y3_logits = self.fc_y3(batch)
        return y1_logits, y2_logits, y3_logits


# ==========================
# IMPROVED MODEL (CNN)
# ==========================
class CNNModel(nn.Module):
    """
    Improved Convolutional Neural Network Model.

    This model uses convolutional layers to extract features from images before classification

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer
        conv2 (nn.Conv2d): Second convolutional layer
        conv3 (nn.Conv2d): Third convolutional layer
        pool (nn.MaxPool2d): Reduces image size while keeping important features
        fc1 (nn.Linear): Fully connected layer before classification
        fc_y1 (nn.Linear): Output layer for first digit
        fc_y2 (nn.Linear): Output layer for operator
        fc_y3 (nn.Linear): Output layer for second digit

    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Reduce size by half
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc_y1 = nn.Linear(256, 10)
        self.fc_y2 = nn.Linear(256, 4)
        self.fc_y3 = nn.Linear(256, 10)

    def forward(self, batch):
        batch = batch.view(-1, 3, 28, 28)  # Reshape to image format
        x = self.pool(torch.relu(self.conv1(batch)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        y1_logits = self.fc_y1(x)
        y2_logits = self.fc_y2(x)
        y3_logits = self.fc_y3(x)
        return y1_logits, y2_logits, y3_logits


# ==========================
# TRAINING FUNCTION
# ==========================


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=NUM_OF_EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=CURR_BATCH_SIZE,
    optimization_method="adam",
):
    """
    Train the given model using cross entropy loss

    Training involves:
    - Forward pass: Predicting labels from input images
    - Loss calculation: Checking how far predictions are from the actual labels
    - Backpropagation: Adjusting weights to reduce loss
    - Optimization: Updating model parameters to improve accuracy

    Args:
        model (torch.nn.Module): Model to train
        train_loader (DataLoader): Training data
        test_loader (DataLoader): Testing data

        num_epochs (int, optional): Number of training cycles. Default is 20
        learning_rate (float, optional): Step size for weight updates. Default is 0.001
        batch_size (int, optional): Number of samples per batch. Default is 64
        optimization_method (str): The method used for optimization. Can be Adam or SGD

    Returns:
        float: Final accuracy on the test set
    """
    loss_fn = nn.CrossEntropyLoss()
    if optimization_method == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9
        )

    # Move model to the selected device (GPU if available)
    model = model.to(device)

    print(
        "\n[Parameters] Epochs:",
        num_epochs,
        " Learning Rate:",
        learning_rate,
        " Batch Size:",
        batch_size,
        " Optimizer:",
        optimization_method.upper(),
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch, y1, y2, y3 in train_loader:
            # Move data to device
            batch = batch.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)

            y1_logits, y2_logits, y3_logits = model(batch)
            y1_loss = loss_fn(y1_logits, y1)
            y2_loss = loss_fn(y2_logits, y2)
            y3_loss = loss_fn(y3_logits, y3)
            total_loss = y1_loss + y2_loss + y3_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch + 1}] Loss: {total_loss.item()}")

    accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# ==========================
# EVALUATION FUNCTION
# ==========================
def evaluate_model(model, test_loader):
    """
    Evaluate model accuracy on the test set.

    The model makes predictions on the test set, and the accuracy is computed based on the number of correctly predicted formulas.

    Args:
        model (torch.nn.Module): trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test images.

    Returns:
        float: Model Accuracy as a Percentage..
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y1, y2, y3 in test_loader:
            # Move data to device
            x = x.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)

            y1_logits, y2_logits, y3_logits = model(x)
            y1_pred = torch.argmax(y1_logits, dim=1)
            y2_pred = torch.argmax(y2_logits, dim=1)
            y3_pred = torch.argmax(y3_logits, dim=1)
            correct_preds = (
                ((y1_pred == y1) & (y2_pred == y2) & (y3_pred == y3))
                .sum()
                .item()
            )
            correct += correct_preds
            total += y1.size(0)

    accuracy = 100 * correct / total
    return accuracy


# ==========================
# COMMAND LINE EXECUTION
# ==========================
@click.group()
def main():
    """Main CLI group."""
    pass


@main.command()
def train_baseline():
    """Train the baseline model"""
    train_loader, test_loader = load_and_process_data()
    print("\nTraining Baseline Model (Logistic Regression)...")
    train_model(
        LinearModel(28 * 28 * 3),
        train_loader,
        test_loader,
        num_epochs=NUM_OF_EPOCHS,
        optimization_method="sgd",
    )


@main.command()
def train_improved():
    """Train the improved model"""
    train_loader, test_loader = load_and_process_data()
    print("\nTraining Improved Model (CNN)...")
    train_model(
        CNNModel(),
        train_loader,
        test_loader,
        num_epochs=NUM_OF_EPOCHS,
        optimization_method="adam",
    )


@main.command(name="performance-diff")
def performance_diff():
    """
    Check if improved model is better using statistical significance method. Comparng the performance of the two models.

    This runs multiple experiments with the best of hyperparameters for both models:
    - Learning Rates
    - Batch sizes
    - Number of Epochs
    - Optimization Method

    The results are written to a file with:
    - Mean Accuracy
    - Standard Error
    - Confidence intervals
    """

    # Experiment parameters
    baseline_lr = 0.001
    baseline_epochs = 20
    baseline_batch = 32
    baseline_optimization = "sgd"

    improved_lr = 0.001
    improved_epochs = 20
    improved_batch = 64
    improved_optimization = "adam"

    # Open a file to output performance stats
    stats_output = open("performance_stats.txt", "a+", encoding="utf-8")

    baseline_accuracies = []
    improved_accuracies = []
    for i in range(10):
        # Train and evaluate the baseline model
        train_loader, test_loader = load_and_process_data(
            batch_size=baseline_batch
        )
        print("\nTraining Baseline Model (Logistic Regression)...")
        baseline_acc = train_model(
            LinearModel(28 * 28 * 3),
            train_loader,
            test_loader,
            num_epochs=baseline_epochs,
            batch_size=baseline_batch,
            learning_rate=baseline_lr,
            optimization_method=baseline_optimization,
        )
        baseline_accuracies.append(baseline_acc)

        # Train and evaluate the improved model
        train_loader, test_loader = load_and_process_data(
            batch_size=improved_batch
        )
        print("\nTraining Improved Model (CNN)...")
        improved_acc = train_model(
            CNNModel(),
            train_loader,
            test_loader,
            num_epochs=improved_epochs,
            learning_rate=improved_lr,
            batch_size=improved_batch,
            optimization_method=improved_optimization,
        )
        improved_accuracies.append(improved_acc)

    stats_output.write(f"performance comparison using best hyperparameters:\n")
    stats_output.write(
        f"Baseline:[ epochs={baseline_epochs} ] -- [ learning rate={baseline_lr} ] -- [ batch size={baseline_batch} ] -- [ optimization={baseline_optimization} ]\n"
    )
    stats_output.write(
        f"Improved:[ epochs={improved_epochs} ] -- [ learning rate={improved_lr} ] -- [ batch size={improved_batch} ] -- [ optimization={improved_optimization} ] \n"
    )
    stats_output.write(f"\tBaseline Accuracies = {baseline_accuracies}\n")
    stats_output.write(f"\tImproved Accuracies = {improved_accuracies}\n")

    # Calculate statistics

    # Calculate Means
    baseline_mean = np.mean(baseline_accuracies)
    improved_mean = np.mean(improved_accuracies)

    # Calculate the SEMs
    baseline_sem = sem(baseline_accuracies)
    improved_sem = sem(improved_accuracies)

    # Calculate the interval bounds
    baseline_lower = baseline_mean - baseline_sem
    baseline_upper = baseline_mean + baseline_sem
    improved_lower = improved_mean - improved_sem
    improved_upper = improved_mean + improved_sem

    # Write the output to file
    stats_output.write(f"\tBaseline Mean Accuracy: {baseline_mean:.3f}\n")
    stats_output.write(f"\tImproved Mean Accuracy: {improved_mean:.3f}\n")
    stats_output.write(f"\tBaseline SEM: {baseline_sem:.4f}\n")
    stats_output.write(f"\tImproved SEM: {improved_sem:.4f}\n")
    stats_output.write(
        f"\tBaseline 68% Confidence Interval: [{baseline_lower:.3f}, {baseline_upper:.3f}]\n"
    )
    stats_output.write(
        f"\tImproved 68% Confidence Interval: [{improved_lower:.3f}, {improved_upper:.3f}]\n\n"
    )
    stats_output.flush()
    stats_output.close()


if __name__ == "__main__":
    main()

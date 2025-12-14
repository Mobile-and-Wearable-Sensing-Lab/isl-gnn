import torch
import torch.nn as nn
import torch.optim as optim
from lstm import SimpleLSTM, TwoHandedSimpleLSTM
from dataset import SingleHandH5Dataset_MobileLastN

from tqdm import tqdm

import matplotlib.pyplot as plt
from datetime import datetime


def create_dataset(input_path, label_file_path, window_length, is_lstm, padding="default", debug=False):
    dataset = SingleHandH5Dataset_MobileLastN(
        root_dir=input_path,
        label_file=label_file_path,
        padding=padding,
        n=window_length,
        is_lstm=is_lstm,
        debug=debug
    )

    return dataset


def create_datasets(input_dir, metadata_dir, label_file, window_length, is_lstm, padding="default", debug=False):
    dataset_list = []
    dataset_types = ["train", "val", "test"]

    for dataset_type in dataset_types:
        dataset = create_dataset(
            input_path=f"{input_dir}/{dataset_type}",
            label_file_path=f"{metadata_dir}/{label_file}",
            window_length=window_length,
            is_lstm=is_lstm,
            padding=padding,
            debug=debug
        )
        dataset_list.append(dataset)

    return dict(zip(dataset_types, dataset_list))


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=16):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader


def create_dataloaders(datasets, batch_size, shuffle=True, num_workers=16):
    dataloader_list = []
    dataloader_types = ["train", "val", "test"]

    for dataloader_type in dataloader_types:
        dataloader = create_dataloader(
            dataset=datasets[dataloader_type],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        dataloader_list.append(dataloader)

    return dict(zip(dataloader_types, dataloader_list))


def initialize_model(metadata_dir, model_type, num_input_features, num_coordinates, num_signs, load_pretrained, model_weights_file, device_type="cpu"):
    model = None
    model_weights_path = f"{metadata_dir}/{model_weights_file}"

    if model_type == "SimpleLSTM":
        model = SimpleLSTM(num_input_features, num_coordinates, num_signs, load_pretrained, model_weights_path, device_type)
    elif model_type == "TwoHandedSimpleLSTM":
        model = TwoHandedSimpleLSTM(num_input_features * 2, num_input_features, num_coordinates, num_signs, load_pretrained, model_weights_path, device_type)
    else:
        raise Exception(f"Model type {model_type} not implemented")

    print(model)
    return model


def initialize_loss_fn(loss_fn_type):
    if loss_fn_type == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    else:
        raise Exception(f"Loss functions type {loss_fn_type} not implemented")


def initialize_optimizer(optimizer_type, model, lr, alpha=0.9):
    if optimizer_type == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr, alpha=alpha)
    elif optimizer_type == "Adam":
        return optim.Adam(model.parameters(), lr=lr)


def train_one_epoch(model, freeze_parameters, dataloader, criterion, optimizer, device_type, epoch_idx):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"----------TRAIN EPOCH {epoch_idx + 1} STARTED----------")
    for inputs, labels in tqdm(dataloader):
        # Move inputs and labels to the device
        inputs, labels = inputs.to(device_type), labels.to(device_type).float()

        # Zero the gradients
        optimizer.zero_grad()

        if freeze_parameters:
            model.freeze_parameters()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels.argmax(dim=1))
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate the running loss
        running_loss += loss.item() * inputs.size(0)

        # Calculate accuracy: convert outputs to probabilities
        predicted = (torch.softmax(outputs, dim=1))
        correct += (predicted.argmax(dim=1) == labels.argmax(dim=1)).sum().item()  # Count correct multi-class matches
        total += labels.size(0)

    accuracy = correct / total
    avg_loss = running_loss / len(dataloader.dataset)

    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}, Correct: {correct}, Total: {total}\n\n")

    return avg_loss, accuracy, correct, total


def evaluate(model, dataloader, criterion, device_type):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device_type), labels.to(device_type).float()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels.argmax(dim=1))
            running_loss += loss.item() * inputs.size(0)

            # Calculate accuracy: convert outputs to probabilities
            predicted = (torch.softmax(outputs, dim=1))
            correct += (predicted.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = running_loss / len(dataloader.dataset)

    return avg_loss, accuracy, correct, total


def save_model_weights(output_dir, model_type, model, label=""):
    # output_path = f"{output_dir}/models/{model_type}"
    output_path = f"{output_dir}"

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

    torch.save(model.state_dict(), f"{output_path}/{label}{dt_string}.pth")
    print(f"Model saved at: {output_path}/{label}{dt_string}.pth")

    scripted_model = torch.jit.script(model)
    scripted_model.save(f"{output_path}/{label}scripted_{dt_string}.pth")
    print(f"Scripted model saved at: {output_path}/{label}scripted_{dt_string}.pth")


def save_metrics_to_file(output_dir, train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy, filename="metrics_summary_"):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

    output_path = f"{output_dir}/{filename}{dt_string}.txt"

    with open(output_path, "w") as f:
        f.write("Training, Validation, and Test Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Train Loss     : {train_loss:.4f}\n")

        if val_loss is not None:
            f.write(f"Validation Loss: {val_loss:.4f}\n")

        f.write(f"Test Loss      : {test_loss:.4f}\n\n")
        f.write(f"Train Accuracy     : {train_accuracy:.4f}\n")

        if val_accuracy is not None:
            f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")

        f.write(f"Test Accuracy      : {test_accuracy:.4f}\n")


def generate_loss_plot(output_dir, model_type, num_epochs, train_loss_arr, val_loss_arr, filename=""):
    # output_path = f"{output_dir}/plots/{model_type}"
    output_path = f"{output_dir}"

    x = [i + 1 for i in range(num_epochs)]
    plt.plot(x, train_loss_arr, 'g', label='train')
    plt.plot(x, val_loss_arr, 'r', label='val')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    plt.savefig(f"{output_path}/{filename}{dt_string}.png")


def generate_acc_plot_across_folds(output_dir, fold_results, filename=""):
    # output_path = f"{output_dir}/plots/{model_type}"
    output_path = f"{output_dir}"

    # Extract data for plotting
    folds = list(fold_results.keys())
    num_folds = len(folds)
    train_acc = [fold_results[f]["train_acc"] for f in folds]
    test_acc = [fold_results[f]["test_acc"] for f in folds]

    # Plot accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(folds, train_acc, marker='o', linestyle='-', label='Train Accuracy')
    plt.plot(folds, test_acc, marker='^', linestyle='-.', label='Test Accuracy')

    # Labels and title
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title(f'Model Accuracy Across {num_folds} Folds')
    plt.xticks(folds)
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    plt.legend()
    plt.grid(True)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    plt.savefig(f"{output_path}/{filename}{dt_string}.png")

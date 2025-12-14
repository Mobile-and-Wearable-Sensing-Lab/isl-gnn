import argparse
import os

import torch

from train_utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a model on the ISL dataset.")

    parser.add_argument('--input_dir', type=str, help="The root path to the input directory.", default="")
    parser.add_argument('--metadata_dir', type=str, help="The root path to the metadata directory.", default="")
    parser.add_argument('--output_dir', type=str, help="The root path to the output directory.", default="")
    parser.add_argument('--label_file', type=str,
                        help="The name of the label .txt file present in the input directory.", default="isl_one_handed_labels.txt")

    parser.add_argument('--window_length', type=int, default=60)
    parser.add_argument('--is_lstm', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_type', type=str, default="SimpleLSTM")
    parser.add_argument('--num_input_features', type=int, default=21)
    parser.add_argument('--num_coordinates', type=int, default=2)
    parser.add_argument('--num_signs', type=int, default=15)
    parser.add_argument('--load_pretrained', type=bool, default=False)
    parser.add_argument('--freeze_parameters', type=bool, default=False)
    parser.add_argument('--model_weights_file', type=str, default="")
    parser.add_argument('--loss_fn_type', type=str, default="CrossEntropyLoss")
    parser.add_argument('--optimizer_type', type=str, default="Adam")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=50)

    args = parser.parse_args()

    input_dir = f"/in/{args.input_dir}" if args.input_dir != "" else "/in"
    metadata_dir = f"/in1/{args.input_dir}" if args.input_dir != "" else "/in1"
    output_dir = f"/out/{args.output_dir}" if args.output_dir != "" else "/out"
    label_file = args.label_file
    window_length = args.window_length
    is_lstm = args.is_lstm
    batch_size = args.batch_size
    model_type = args.model_type
    num_input_features = args.num_input_features
    num_coordinates = args.num_coordinates
    num_signs = args.num_signs
    load_pretrained = args.load_pretrained
    freeze_parameters = args.freeze_parameters
    model_weights_file = args.model_weights_file
    loss_fn_type = args.loss_fn_type
    optimizer_type = args.optimizer_type
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs

    os.makedirs(output_dir, exist_ok=True)

    datasets = create_datasets(input_dir, metadata_dir, label_file, window_length, is_lstm)
    dataloaders = create_dataloaders(datasets, batch_size)

    model = initialize_model(metadata_dir, model_type, num_input_features, num_coordinates, num_signs, load_pretrained, model_weights_file)
    criterion = initialize_loss_fn(loss_fn_type)
    optimizer = initialize_optimizer(optimizer_type, model, learning_rate)

    train_loss_arr = []
    val_loss_arr = []

    train_acc_arr = []
    val_acc_arr = []

    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device_type)

    print(f"=========RUNNING ON {device_type}==========")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]

        train_loss, train_accuracy, train_correct, train_total = train_one_epoch(model, freeze_parameters, train_loader, criterion, optimizer, device_type, epoch)
        val_loss, val_accuracy, val_correct, val_total = evaluate(model, val_loader, criterion, device_type)

        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f} | correct, total | {val_correct, val_total}")
        print(f"----------TRAIN EPOCH {epoch + 1} COMPLETE----------\n")

        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)

        train_acc_arr.append(train_accuracy)
        val_acc_arr.append(val_accuracy)

    save_model_weights(output_dir, model_type, model)

    # Evaluate on the test set
    model.eval()
    test_loader = dataloaders["test"]
    test_loss, test_accuracy, correct, total = evaluate(model, test_loader, criterion, device_type)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f} | correct, total | {correct, total}")

    avg_train_loss = sum(train_loss_arr[-5:]) / 5
    avg_val_loss = sum(val_loss_arr[-5:]) / 5

    avg_train_acc = sum(train_acc_arr[-5:]) / 5
    avg_val_acc = sum(val_acc_arr[-5:]) / 5

    save_metrics_to_file(output_dir, avg_train_loss, avg_val_loss, test_loss, avg_train_acc, avg_val_acc, test_accuracy)
    generate_loss_plot(output_dir, model_type, num_epochs, train_loss_arr, val_loss_arr)

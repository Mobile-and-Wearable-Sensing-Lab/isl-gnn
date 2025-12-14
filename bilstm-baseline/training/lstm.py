import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self, num_input_features, num_coordinates, num_signs, load_pretrained=False, model_weights_file="", device_type="cpu"):
        super(SimpleLSTM, self).__init__()
        
        # First bidirectional LSTM layer with return_sequences=True equivalent
        self.lstm1 = nn.LSTM(
            input_size=num_input_features * num_coordinates,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        
        # Second bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=128 * 2,  # 128 hidden units * 2 for bidirection
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(128 * 2, num_signs)

        if load_pretrained:
            self.load_weights(model_weights_file, num_signs, device_type)

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        
        # Only take the last output for classification
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification layer
        x = self.fc(x)
        
        return x

    def load_weights(self, model_weights_file, num_signs, device_type):
        self.fc = nn.Linear(128 * 2, 563)
        self.load_state_dict(torch.load(model_weights_file, map_location=torch.device(device_type)))
        self.fc = nn.Linear(128 * 2, num_signs)
        print("Loaded model weights")

    def freeze_parameters(self):
        for param in self.lstm1.parameters():
            param.requires_grad = False

        for param in self.lstm2.parameters():
            param.requires_grad = False


class TwoHandedSimpleLSTM(nn.Module):
    def __init__(self, num_two_handed_input_features, num_one_handed_input_features, num_coordinates, num_signs, load_pretrained=False, model_weights_file="", device_type="cpu"):
        super(TwoHandedSimpleLSTM, self).__init__()

        self.fc = nn.Linear(num_two_handed_input_features * num_coordinates, num_one_handed_input_features * num_coordinates)

        self.simple_lstm = SimpleLSTM(num_one_handed_input_features, num_coordinates, num_signs, load_pretrained, model_weights_file, device_type)

        if load_pretrained:
            self.load_weights(model_weights_file, num_signs, device_type)

    def forward(self, x):
        # First Two-handed transformer LSTM layer
        x = self.fc(x)

        # Second Condensed information processor LSTM layer
        x = self.simple_lstm.forward(x)

        return x

    def load_weights(self, model_weights_file, num_signs, device_type):
        print("Loaded model weights")

    def freeze_parameters(self):
        for param in self.simple_lstm.lstm1.parameters():
            param.requires_grad = False

        for param in self.simple_lstm.lstm2.parameters():
            param.requires_grad = False

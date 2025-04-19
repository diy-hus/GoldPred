import torch
import torch.nn as nn
import numpy as np


class GoldLSTM(nn.Module):

    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(GoldLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(1)


# Load model
device = torch.device("cpu")
model = GoldLSTM().to(device)
model.load_state_dict(
    torch.load("model/gold_lstm_model.pt", map_location=device))
model.eval()


def forecast_next_days(scaled_seq, steps=3):
    sequence = torch.tensor(
        scaled_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
    predictions = []

    for _ in range(steps):
        with torch.no_grad():
            pred = model(sequence).item()
        predictions.append(pred)

        # Update sequence
        new_point = torch.tensor([[pred]], dtype=torch.float32).to(device)
        sequence = torch.cat((sequence[:, 1:, :], new_point.unsqueeze(0)),
                             dim=1)

    return predictions

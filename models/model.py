import torch.nn as nn

class CovidPredictor(nn.Module):
    def __init__(self, n_features: int, n_hidden: int = 128, n_layers: int = 2, dropout: float = 0.2):
        super(CovidPredictor, self).__init__()

        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

        self.regressor = nn.Linear(n_hidden, 1)

    def forward(self, x):
        self.lstm.flatten_parameters()

        _, (hidden, _) = self.lstm(x)

        out = hidden[-1]

        return self.regressor(out)
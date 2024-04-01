import torch
import torch.nn as nn


class CausalTemporalLSTM(nn.Module):
    def __init__(self, num_features=124, 
                    hidden_units= 3, #was 128
                    #initial_offset = -2,
                    num_layers = 2, 
                    out_dims = 6):
        super(CausalTemporalLSTM, self).__init__()
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size= int(num_features/4),
            hidden_size=hidden_units,
            batch_first=True,
            num_layers= num_layers,
            bidirectional=False,
        )
        self.linear1 = nn.Linear(in_features=self.num_features, out_features=int(num_features/4))
        self.linear2 = nn.Linear(in_features=self.hidden_units, out_features=out_dims)

    def forward(self, x):

        x = self.linear1(x)
        x, _ = self.lstm(x)
        output = self.linear2(x)
        # Apply sigmoid activation function
        output = torch.sigmoid(output)
        
        return output.squeeze()
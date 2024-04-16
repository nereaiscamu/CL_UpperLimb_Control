import torch
import torch.nn as nn


class CausalTemporalLSTM(nn.Module):
    def __init__(self, num_features=124, 
                    hidden_units= 3, #was 128
                    #initial_offset = -2,
                    num_layers = 2, 
                    input_size = 50,
                    out_dims = 6, 
                    dropout_1 = 0.3, 
                    dropout_2 = 0.3):
        super(CausalTemporalLSTM, self).__init__()
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size= self.input_size,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers= num_layers,
            bidirectional=False,)
            
        self.linear1 = nn.Linear(in_features=self.num_features, out_features= self.input_size)
        self.linear2 = nn.Linear(in_features=self.hidden_units, out_features=out_dims)

        self.dropout1 = nn.Dropout(p= dropout_1) #trial.suggest_float('dropout_1', 0.1, 0.9)
        self.dropout2 = nn.Dropout(p= dropout_2) 

    def forward(self, x):

        x = self.linear1(x)
        x = self.dropout1(x)
        x, _ = self.lstm(x)
        x = self.dropout2(x)
        output = self.linear2(x)
        # Apply sigmoid activation function
        output = torch.sigmoid(output)
        
        return output.squeeze()
    

class CausalRNN(nn.Module):
    def __init__(self, input_size = 124, hidden_size = 10, num_layers = 1,
                 output_dim = 3, dropout = 0.3, ):
    
        super(CausalRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.rnn = nn.RNN(
            input_size = self.input_size, 
            hidden_size = self.hidden_size, 
            num_layers = self.num_layers, 
            nonlinearity='tanh', bias= True, 
            batch_first= True, dropout=0.0, 
            bidirectional=False,)

        self.linear = nn.Linear(in_features=self.hidden_size, out_features= self.output_dim)
        self.dropout = nn.Dropout(p= dropout) #trial.suggest_float('dropout_1', 0.1, 0.9)

    def forward(self, x):

        x, _ = self.rnn(x)
        x = self.dropout(x)
        output = self.linear(x)
        # Apply sigmoid activation function
        output = torch.sigmoid(output)
        
        return output.squeeze()
    
import torch
import torch.nn as nn
import torch.nn.functional as F



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
    

class Causal_Simple_RNN(nn.Module):
    def __init__(self, num_features=124, 
                hidden_units= 3, #was 128
                num_layers = 2, 
                out_dims = 6,
                dropout = 0.5):
        super(Causal_Simple_RNN, self).__init__()
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.out_dims = out_dims
        self.dropout = dropout

        self.linear = nn.Linear(in_features=self.hidden_units, out_features= self.out_dims)

        self.rnn = nn.RNN(
            input_size = self.num_features, 
            hidden_size = self.hidden_units, 
            num_layers = self.num_layers, 
            nonlinearity='tanh', bias= True, 
            batch_first= True, dropout=0.0, 
            bidirectional=False,)  

        self.selu = nn.SELU()
    
        self.dropout = nn.Dropout(p= dropout) #trial.suggest_float('dropout_1', 0.1, 0.9)
        
        # Flatten the parameters
        self.rnn.flatten_parameters()

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.selu(x) 
        output = self.linear(x)

        return output.squeeze()
    

#### This model is meant to be used with hnets.
def create_state_dict(param_names, param_values):
    s_d = {}
    for n,v in zip(param_names, param_values):
        s_d[n] = v
    return s_d



class RNN_Main_Model(nn.Module):
    def __init__(self, hnet_output, 
                 num_features = 124, 
                hidden_size= 3, 
                num_layers = 2, 
                out_dims = 6,
                dropout = 0.5,
                bias = True,
                LSTM_ = False):
        
        super(RNN_Main_Model, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hnet_output = hnet_output
        self.bias = bias
        self.out_features = out_dims
        self.LSTM_ = LSTM_
        self.dropout_value = dropout

        self.dropout = nn.Dropout(p= self.dropout_value) #trial.suggest_float('dropout_1', 0.1, 0.9)

        # Define recurrent layer
        self.rnn = nn.RNN(self.num_features, self.hidden_size, self.num_layers, self.bias, batch_first = True, bidirectional = False)
        names_p = [name for name, _ in self.rnn.named_parameters()]
        self.hnet_output_dict = create_state_dict(names_p,hnet_output[2:] )

        # Define recurrent layer (LSTM)
        if self.LSTM_:
            self.rnn = nn.LSTM(self.num_features, self.hidden_size, self.num_layers, self.bias, batch_first = True, bidirectional = False)
            names_p = [name for name, _ in self.rnn.named_parameters()]
            self.hnet_output_dict = create_state_dict(names_p,hnet_output[2:] )      

        self.selu = nn.SELU()      

    def forward(self, x, hx=None):
        # Forward pass
        if hx is None:
            if self.LSTM_:
                h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size, device=x.device) * 0.1
                c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size, device=x.device) *0.1 # Initialize cell state
                hx = (h0, c0)
            else:
                hx = torch.randn(self.num_layers, x.size(0), self.hidden_size, device=x.device) * 0.1
        
        # Perform RNN operation
        x, _  = torch.func.functional_call(self.rnn, self.hnet_output_dict, (x, hx))
        x = self.dropout(x)
        x = self.selu(x) 
        output =  F.linear(x, self.hnet_output[0], bias=self.hnet_output[1])
        
        return output.squeeze() 


class Task_Recog_Model(nn.Module):
    def __init__(self, num_features=124, 
                hidden_units= 3, #was 128
                num_layers = 2, 
                output_size = 6):
        super(Task_Recog_Model, self).__init__()
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size = self.num_features, 
            hidden_size = self.hidden_units, 
            num_layers = self.num_layers, 
            nonlinearity='tanh', bias= True, 
            batch_first= True, dropout=0.0, 
            bidirectional=False,)  
        
        self.fc = nn.Linear(hidden_units, output_size)

        # Flatten the parameters
        self.rnn.flatten_parameters()

    def forward(self, x):
        x, _ = self.rnn(x)
        out = self.fc(x[:, -1, :])

        return out.squeeze()
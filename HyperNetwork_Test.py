from typing import Optional, Tuple
import torch
from torch import nn
from labml_helpers.module import Module
from labml_nn.lstm import LSTMCell


class HyperLSTMCell(Module):

    def __init__(self, input_size: int, embedding_size: int, hidden_size: int, hyper_size: int, n_z: int):
        """
        `input_size` is the size of the input $x_t$,
        `hidden_size` is the size of the LSTM, and
        `hyper_size` is the size of the smaller LSTM that alters the weights of the larger outer LSTM.
        `n_z` is the size of the feature vectors used to alter the LSTM weights.

        We use the output of the smaller LSTM to compute $z_h^{i,f,g,o}$, $z_x^{i,f,g,o}$ and
        $z_b^{i,f,g,o}$ using linear transformations.
        We calculate $d_h^{i,f,g,o}(z_h^{i,f,g,o})$, $d_x^{i,f,g,o}(z_x^{i,f,g,o})$, and
        $d_b^{i,f,g,o}(z_b^{i,f,g,o})$ from these, using linear transformations again.
        These are then used to scale the rows of weight and bias tensors of the main LSTM.
        """

        super().__init__()
        self.hyper = LSTMCell(embedding_size, hyper_size, layer_norm=True)
        # LSTMCell class takes x: torch.Tensor, h: torch.Tensor, c: torch.Tensor and returns h_next and c_next
        self.z_h = nn.Linear(hyper_size, 4 * n_z)
        self.z_x = nn.Linear(hyper_size, 4 * n_z)
        self.z_b = nn.Linear(hyper_size, 4 * n_z, bias=False)

        d_h = [nn.Linear(n_z, hidden_size, bias=False) for _ in range(4)]
        self.d_h = nn.ModuleList(d_h)

        d_x = [nn.Linear(n_z, hidden_size, bias=False) for _ in range(4)]
        self.d_x = nn.ModuleList(d_x)

        d_b = [nn.Linear(n_z, hidden_size) for _ in range(4)]
        self.d_b = nn.ModuleList(d_b)


        # Defining weight matrices Wh ^(i,f,g,o)
        self.w_h = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, hidden_size)) for _ in range(4)])
        self.w_x = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, input_size)) for _ in range(4)])

        #Defining layer normalization
        self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
        self.layer_norm_c = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, e: torch.Tensor, #added e for the task embeddings
                h: torch.Tensor, c: torch.Tensor,
                h_hat: torch.Tensor, c_hat: torch.Tensor):
        # Defining input of the hypernetwork

        # Computing cell and hidden states
        h_hat, c_hat = self.hyper(e, h_hat, c_hat)  # --> changed x by e to make the 
                                                    # hypernetwork output depend only on task embeddings.

        # Computing the embeddings
        z_h = self.z_h(h_hat).chunk(4, dim=-1)
        z_x = self.z_x(h_hat).chunk(4, dim=-1)
        z_b = self.z_b(h_hat).chunk(4, dim=-1)

        # Compute d_h * self.w_h[i] separately
        dhw_h = [torch.matmul(self.d_h[i], self.w_h[i]) for i in range(4)]

        # Compute d_x * self.w_x[i] separately
        dhw_x = [torch.matmul(self.d_x[i], self.w_x[i]) for i in range(4)]

        return dhw_h, dhw_x, z_b

class MainLSTMCell(Module):
    def __init__(self, input_size: int, embedding_size:int, hidden_size: int, hyper_size: int, n_z: int):
        self.hyper = HyperLSTMCell(input_size, embedding_size, hidden_size, hyper_size)

    def forward(self, x: torch.Tensor, e: torch.Tensor, #added e for the task embeddings
                h: torch.Tensor, c: torch.Tensor,
                h_hat: torch.Tensor, c_hat: torch.Tensor):

        dhw_h, dhw_x, z_b = self.hyper(e, h_hat, c_hat)
        # We calculate i, f, g and o in a loop:
        ifgo = []
        for i in range(4):
            y = d_h * torch.einsum('ij,bj->bi', self.w_h[i], h) + \
                d_x * torch.einsum('ij,bj->bi', self.w_x[i], x) + \
                self.d_b[i](z_b[i])

            ifgo.append(self.layer_norm[i](y))
        
        i, f, g, o = ifgo

        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)

        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return h_next, c_next, h_hat, c_hat

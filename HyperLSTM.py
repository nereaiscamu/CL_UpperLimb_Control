""" 
### Implementation from https://nn.labml.ai/hypernetworks/hyper_lstm.html

Check complete implementation here: 
https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/hypernetworks/hyper_lstm.py


### Date: 12/04/2024


INTRO:

In the basic form, a Dynamic HyperNetwork has a smaller recurrent network that generates a 
feature vector corresponding to each parameter tensor of the larger recurrent network. Let's 
say the larger network has some parameter Wh the smaller network generates a feature vector zh
and we dynamically compute Wh as a linear transformation of zh. For instance Wh=⟨ Whz, zh ⟩ where Whz 
is a 3-d tensor parameter and ⟨.⟩ is a tensor-vector multiplication. zh is usually a linear
transformation of the output of the smaller recurrent network.

WEIGHT SCALING INSTEAD OF COMPUTING:
Large recurrent networks have large dynamically computed parameters. 
These are calculated using linear transformation of feature vector Z.
When Wh has shape: NhxNh, then Whz has shape NhxNhxNz.

To overcome this, we compute the weight parameters of the recurrent network by 
dynamically scaling each row of a matrix of same size.

d(z) = Whz * zh --> so row 0 of Wh matrix becomes: d0(z)* Whd0

where Whd is a Nh x Nh parameter matrix.
We can then further optimize by computing Wh*h as d(z) · (Whd * h), where · is a element-wise multiplication.


"""
from typing import Optional, Tuple
import torch
from torch import nn
from labml_helpers.module import Module
from labml_nn.lstm import LSTMCell




class HyperLSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, hyper_size: int, n_z: int):
        super().__init__()
        self.hyper = LSTMCell(hidden_size + input_size, hyper_size, layer_norm=True)
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

    def forward(self, x: torch.Tensor,
                h: torch.Tensor, c: torch.Tensor,
                h_hat: torch.Tensor, c_hat: torch.Tensor):
        # Defining input of the hypernetwork
        x_hat = torch.cat((h, x), dim=-1)

        # Computing cell and hidden states
        h_hat, c_hat = self.hyper(x_hat, h_hat, c_hat)

        # Computing the embeddings
        z_h = self.z_h(h_hat).chunk(4, dim=-1)
        z_x = self.z_x(h_hat).chunk(4, dim=-1)
        z_b = self.z_b(h_hat).chunk(4, dim=-1)

        # We calculate i, f, g and o in a loop:
        ifgo = []
        for i in range(4):
            d_h = self.d_h[i](z_h[i])
            d_x = self.d_x[i](z_x[i])
            y = d_h * torch.einsum('ij,bj->bi', self.w_h[i], h) + \
                d_x * torch.einsum('ij,bj->bi', self.w_x[i], x) + \
                self.d_b[i](z_b[i])

            ifgo.append(self.layer_norm[i](y))
        
        i, f, g, o = ifgo

        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)

        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return h_next, c_next, h_hat, c_hat
    

class HyperLSTM(Module):
    # Create a network of n_layers of HyperLSTM.
    def __init__(self, input_size: int, hidden_size: int, hyper_size: int, n_z: int, n_layers: int):
        super().__init__()

        # Store sizes to initialize state
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.hyper_size = hyper_size


        # Create cells for each layer. Note that only the first layer gets the input directly. 
        # Rest of the layers get the input from the layer below
        self.cells = nn.ModuleList([HyperLSTMCell(input_size, hidden_size, hyper_size, n_z)] +
                                   [HyperLSTMCell(hidden_size, hidden_size, hyper_size, n_z) for _ in
                                    range(n_layers - 1)])
        
        # x has shape [n_steps, batch_size, input_size] and
        # state is a tuple of h,c, h_hat, c_hat. h,c have shape [batch_size, 
        # hidden_size] and  h_hat and c_hat have shape [batch_size, hyper_size].

        def forward(self, x: torch.Tensor,
            state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None):

            n_steps, batch_size = x.shape[:2]

            # Initialize the state with zeros if None
            if state is None:
                h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
                c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
                h_hat = [x.new_zeros(batch_size, self.hyper_size) for _ in range(self.n_layers)]
                c_hat = [x.new_zeros(batch_size, self.hyper_size) for _ in range(self.n_layers)]

            else:
                (h, c, h_hat, c_hat) = state

            # Reverse stack the tensors to get the states of each layer
            # You can just work with the tensor itself but this is easier to debug
            h, c = list(torch.unbind(h)), list(torch.unbind(c))
            h_hat, c_hat = list(torch.unbind(h_hat)), list(torch.unbind(c_hat))

            # Collect the outputs of the final layer at each step
            out = []
            for t in range(n_steps):
                # Input to the first layer is the input itself
                inp = x[t]

                # Loop through the layers
                for layer in range(self.n_layers):
                    # Get the state of the layer
                    h[layer], c[layer], h_hat[layer], c_hat[layer] = \
                    self.cells[layer](inp, h[layer], c[layer], h_hat[layer], c_hat[layer])

                    # Input to the next layer is the state of this layer
                    inp = h[layer]

                # Collect the output h of the final layer
                out.append(h[-1])

            # Stack the outputs and states
            out = torch.stack(out)
            h = torch.stack(h)
            c = torch.stack(c)
            h_hat = torch.stack(h_hat)
            c_hat = torch.stack(c_hat)

            return out, (h, c, h_hat, c_hat)









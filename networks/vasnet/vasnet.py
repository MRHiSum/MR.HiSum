__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"


import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.vasnet.layer_norm import LayerNorm



class SelfAttention(nn.Module):

    def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()

        self.apperture = apperture
        self.ignore_itself = ignore_itself

        self.m = input_size
        self.output_size = output_size

        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        self.drop50 = nn.Dropout(0.5)

    def forward(self, x, mask):
        bs = x.shape[0]
        n = x.shape[1]  # sequence length
        dim = x.shape[2]
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)

        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1,2))
        
        # for batch training
        if mask is not None:
            mask = mask.unsqueeze(-1)
            mask_t = mask.transpose(2,1)
            attention_mask = torch.matmul(mask.float(), mask_t.float()).bool()
            logits[~attention_mask] = -1e9 #float('-Inf')

        if self.ignore_itself:
            # Zero the diagonal activations (a distance of each frame with itself)
            logits[torch.eye(n).byte()] = -float("Inf")

        if self.apperture > 0:
            # Set attention to zero to frames further than +/- apperture from the current one
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")

        att_weights_ = nn.functional.softmax(logits, dim=-1)
        if torch.isnan(att_weights_).any():
            print("[ERROR] NaN att_weights_", att_weights_.shape, att_weights_)
            exit()
        weights = self.drop50(att_weights_)
        y = torch.matmul(V.transpose(1,2), weights).transpose(1,2)
        y = self.output_linear(y)

        return y, att_weights_



class VASNet(nn.Module):

    def __init__(self, hidden_dim=1024):
        super(VASNet, self).__init__()

        self.m = 1024 # cnn features size
        self.hidden_size = hidden_dim

        self.att = SelfAttention(input_size=self.m, output_size=self.hidden_size)
        self.ka = nn.Linear(in_features=self.m, out_features=1024)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)


    def forward(self, x, mask=None):
        bs = x.shape[0]
        m = x.shape[2] # Feature size

        x = x.view(bs, -1, m)
        y, att_weights_ = self.att(x, mask=mask)
        y = y + x
        y = self.drop50(y)
        y = self.layer_norm_y(y)

        # Frame level importance score regression
        # Two layer NN
        y = self.ka(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)
        y = self.kd(y)
        y = self.sig(y)
        y = y.view(bs, -1)
        return y, att_weights_



if __name__ == "__main__":
    pass
import torch
import torch.nn as nn


from networks.sl_module.transformer import Transformer
from networks.sl_module.score_net import ScoreFCN

class SL_module(nn.Module):

    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio):
        super(SL_module, self).__init__()
        
        self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio)
        self.score_model = ScoreFCN(emb_dim=1024)
        
    def forward(self, x, mask):

        transformed_emb = self.transformer(x)
        score = self.score_model(transformed_emb).squeeze(-1)
        
        score = torch.sigmoid(score)

        return score, 0.0

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys(): 
            self.transformer.load_state_dict(state_dict['transformer'])
            self.score_model.load_state_dict(state_dict['score_model'])
        else:
            super(SL_module, self).load_state_dict(state_dict)
import torch
from torch import nn, optim
import torch.nn.functional as F

from .MRI_block import MRI_block_att
from .revin import RevIN


class MRIformer(nn.Module):
    def __init__(self, Input_len, out_len, num_id, num_hi, muti_head,num_layer,dropout,IF_Chanel):
        """
        Input_len: History length
        out_len: Future length
        num_id: number of variable
        num_hi: Number of hidden
        muti_head: number of muti_head attention
        num_layer: number of MRI block
        dropout: dropout rate
        IF_Chanel: Whether to adopt the channel independent modeling strategy
        """
        super(MRIformer, self).__init__()

        self.RevIN = RevIN(num_id)
        ###encorder
        self.MRI_block_1 = MRI_block_att(Input_len, num_id, num_hi, muti_head, dropout,IF_Chanel)
        self.laynorm_1 = nn.LayerNorm([num_id,Input_len])

        ###decorder
        self.num_layer = num_layer
        self.output = nn.Conv1d(in_channels = Input_len, out_channels=out_len, kernel_size=1)

    def forward(self, x):
        # Input [B,H,N]: B is batch size. N is the number of variables. H is the history length
        # Output [B,L,N]: B is batch size. N is the number of variables. L is the future length
        
        x = self.RevIN(x, 'norm').transpose(-2, -1)

        for i in range(self.num_layer):
            x = x + self.MRI_block_1(x)
            x = self.laynorm_1(x)

        x = self.output(x.transpose(-2, -1))
        x = self.RevIN(x, 'denorm')
        return x

import torch
from torch import nn
from typing import Callable
import numpy as np

from hyperparameters import *


class StockBot(nn.Module):
    def __init__(
        self,
        past_history: int = PAST_HISTORY,
        forward_look: int = FORWARD_LOOK,
        stack_depth: int = STACK_DEPTH,
        layer_units: int = LAYER_UNITS,

        bias:bool = BIAS,
        hidden_size:int = HIDDEN_SIZE,
        dropout:int = DROPOUT
    ):
        super().__init__()

        self.past_history = past_history
        self.forward_look = forward_look
        self.stack_depth = stack_depth
        self.layer_units = layer_units

        self.layer_norm = nn.LayerNorm((past_history, past_history))
        self.lstm = nn.LSTM(
                input_size=past_history,
                hidden_size=past_history,
                num_layers=stack_depth,
                bias=bias,
                batch_first=True)
        self.ffn1 = nn.Linear(
                in_features=past_history,
                out_features=1,
                bias=bias
                )
        self.ffn2 = nn.Linear(
                in_features=past_history,
                out_features=1,
                bias=bias
                )


    def forward(self, x):
        x = self.layer_norm(x)
        x, (h_n, c_n) = self.lstm(x)
        x = self.ffn1(x)
        x = torch.squeeze(x)
        x = self.ffn2(x)
        x = torch.squeeze(x)

        return x
        
if __name__ == "__main__":
    sb = StockBot()
    print(sb)

# PLACE TORCH MODELS HERE
import torch.nn as nn
import torchxrayvision as xrv

class DenseNetBaseline(nn.Module):

    def __init__(self):
        super(DenseNetBaseline, self).__init__()
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")

    def forward(self, input_data):
        return self.model(input_data)
# PLACE TORCH MODELS HERE
import torch.nn as nn
import torchxrayvision as xrv

class DenseNetBaseline(nn.Module):

    def __init__(self):
        super(DenseNetBaseline, self).__init__()
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, 14)
        self.model.op_threshs = None
        self.model.apply_sigmoid = False

    def forward(self, input_data):
        return self.model(input_data)

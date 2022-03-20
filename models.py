# PLACE TORCH MODELS HERE
import torch.nn as nn
import torchxrayvision as xrv
import timm

class DenseNetBaseline(nn.Module):

    def __init__(self, num_classes=14):
        super(DenseNetBaseline, self).__init__()
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)
        self.model.op_threshs = None
        self.model.apply_sigmoid = False

    def forward(self, input_data):
        return self.model(input_data)


class TimmModels(nn.Module):

    def __init__(self, model_name, num_classes=14, pretrained=True):
        super(TimmModels, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # in_features = self.model.classifier.in_features
        # self.model.classifier = nn.Linear(in_features, num_classes)
        self.model.reset_classifier(num_classes)
        
    def forward(self, input_data):
        return self.model(input_data)

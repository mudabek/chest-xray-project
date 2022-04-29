from torch import nn
import torch
import torch.nn.functional as F

from convnext.global_pool import GlobalPool
from convnext.attention_map import AttentionMap


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class Classifier(nn.Module):

    def __init__(self, cfg=None):
        super(Classifier, self).__init__()
        self.global_pool = GlobalPool('PCAM')
        self.expand = 1
        self.num_classes = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self._init_classifier()
        self._init_bn()
        self._init_attention_map()

    def _init_classifier(self):
        # for index, num_class in enumerate(self.cfg.num_classes):
        for index, num_class in enumerate(self.num_classes):
            setattr(
                self,
                "fc_" + str(index),
                nn.Conv2d(
                    560 * self.expand,
                    num_class,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True))

            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def _init_bn(self):
        # for index, num_class in enumerate(self.cfg.num_classes):
        for index, num_class in enumerate(self.num_classes):
            setattr(self, "bn_" + str(index),
                        nn.BatchNorm2d(560 * self.expand))

    def _init_attention_map(self):
        setattr(self, "attention_map", AttentionMap(560))

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, x):
        # (N, C, H, W)
        feat_map = x
        # [(N, 1), (N,1),...]
        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = list()
        for index, num_class in enumerate(self.num_classes):
            feat_map = self.attention_map(feat_map)

            classifier = getattr(self, "fc_" + str(index))
            # (N, 1, H, W)
            logit_map = None
            logit_map = classifier(feat_map)
            logit_maps.append(logit_map.squeeze())
            # (N, C, 1, 1)
            feat = self.global_pool(feat_map, logit_map)

            bn = getattr(self, "bn_" + str(index))
            feat = bn(feat)
            feat = F.dropout(feat, p=0, training=self.training)
            # (N, num_class, 1, 1)

            logit = classifier(feat)
            # (N, num_class)
            logit = logit.squeeze(-1).squeeze(-1)
            logits.append(logit)

        return (logits, logit_maps)
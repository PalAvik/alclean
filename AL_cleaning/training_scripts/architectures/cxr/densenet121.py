import torch
import torchvision
from torch import nn

from InnerEyeDataQuality.configs.config_node import ConfigNode


class Network(nn.Module):
    """
    DenseNet121 as implement in torchvision
    """

    def __init__(self, config: ConfigNode) -> None:
        super().__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=config.train.pretrained, progress=False)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, config.dataset.n_classes)
        self.projection = self.densenet121.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.densenet121(x)
        return x

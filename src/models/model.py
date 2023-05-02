import torch.nn as nn
import torch.hub


class PokeModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PokeModel, self).__init__()
        self.model = torch.hub.load(
            'pytorch/vision:v0.10.0', 'vgg11', weights=pretrained)
        self.model.fc = torch.nn.Linear(self.model.classifier[6].in_features, num_classes)
        
        

    def forward(self, x):
        x = self.model(x)
        return x

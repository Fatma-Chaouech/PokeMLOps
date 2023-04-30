import torch.nn as nn
import torch.hub

class PokeModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PokeModel, self).__init__()
        self.features = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', weights=pretrained)
        self.classifier = self.features.classifier
        self.fc = nn.Linear(self.classifier[6].in_features, num_classes)
        self.classifier[6] = self.fc

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
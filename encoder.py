import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, images):
        features = self.resnet(images)  # (batch_size, 2048, 7, 7)
        return features.view(
            features.size(0), -1, features.size(1)
        )  # (batch_size, 49, 2048)

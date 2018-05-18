import torch.nn as nn
from .BBBdistributions import Normal
from .BBBConvlayer import BBBConv2d
import torch.nn.init as init


class BBBCNN(nn.Module):
    def __init__(self, num_tasks):
        # create AlexNet with probabilistic weights
        super(BBBCNN, self).__init__()

        # FEATURES
        self.conv1 = BBBConv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv1a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = BBBConv2d(64, 192, kernel_size=5, padding=2)
        self.conv2a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = BBBConv2d(192, 384, kernel_size=3, padding=1)
        self.conv3a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(384),
        )
        self.conv4 = BBBConv2d(384, 256, kernel_size=3, padding=1)
        self.conv4a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(256),
        )
        self.conv5 = BBBConv2d(256, 256, kernel_size=3, padding=1)
        self.conv5a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # CLASSIFIER
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_tasks)
        )

        layers = [self.conv1, self.conv1a, self.conv2, self.conv2a, self.conv3, self.conv3a, self.conv4, self.conv4a,
                  self.conv5, self.conv5a, self.classifier]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'probforward') and callable(layer.probforward):
                x, _kl, = layer.probforward(x)
                kl += _kl
            elif layer is self.classifier:
                x = x.view(-1, 256 * 6 * 6)
                x = layer(x)
            else:
                x = layer(x)
        logits = x
        return logits, kl

    def load_prior(self, state_dict):
        d_q = {k: v for k, v in state_dict.items() if "q" in k}
        for i, layer in enumerate(self.layers):
            if type(layer) is BBBConv2d:
                layer.pw = Normal(mu=d_q["layers.{}.qw_mean".format(i)],
                                  logvar=d_q["layers.{}.qw_logvar".format(i)])

                #layer.pb = Normal(mu=d_q["layers.{}.qb_mean".format(i)], logvar=d_q["layers.{}.qb_logvar".format(i)])

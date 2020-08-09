import torch
import torch.nn as nn

class DCGAN(nn.Module):
    '''
    The DCGAN Network
    '''

    def __init__(self):
        super(DCGAN, self).__init__()

        self.conv_trans_1 = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        # Shape [2 x 2]

        self.conv_trans_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        # Shape [4 x 4]

        self.conv_trans_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        # Shape [7 x 7]

        self.conv_trans_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        # Shape [14 x 14]

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        # Shape [28 x 28]

    def forward(self, inputs):
        x = self.conv_trans_1(inputs)
        x = self.conv_trans_2(x)
        x = self.conv_trans_3(x)
        x = self.conv_trans_4(x)
        return self.output_layer(x)
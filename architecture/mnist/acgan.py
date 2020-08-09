import torch
import torch.nn as nn

class ACGAN(nn.Module):
    '''
    The ACGAN network
    '''
    def __init__(self):
        super(ACGAN, self).__init__()

        # For labels
        self.label_embedding = nn.Embedding(10, 100)
        self.linear = nn.Sequential(nn.Linear(100, 128 * 4 ** 2))
        # 4 ** 2 will come handy as we reshape the layer to (batch_size, 128, 4, 4)
        # as the first Transposed Conv layer is skipped.

        self.norm_layer = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        # Shape [4 x 4]

        self.conv_trans_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        # Shape [7 x 7]

        self.conv_trans_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        # Shape [14 x 14]

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        # Shape [28 x 28]
            

    def forward(self, inputs, labels):
        x = torch.mul(self.label_embedding(labels), inputs)
        x = self.linear(x)
        x = x.view(x.shape[0], 128, 4, 4)
        x = self.norm_layer(x)
        x = self.conv_trans_1(x)
        x = self.conv_trans_2(x)
        return self.output_layer(x)

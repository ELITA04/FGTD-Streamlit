import torch
import torch.nn as nn

class GAN(nn.Module):
    '''
    The GAN Network
    '''

    def __init__(self):
        super(GAN, self).__init__()

        self.linear_1 = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2)
        )
        # Out 256

        self.linear_2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        # Out 512

        self.linear_3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
        )
        # Out 1024

        self.output_layer = nn.Sequential(
            nn.Linear(1024, 28 ** 2),
            nn.Tanh()
        )
        # Out 784

    def forward(self, inputs):
        x = self.linear_1(inputs)
        x = self.linear_2(x)
        x = self.linear_3(x)
        generated_image = self.output_layer(x)
        return generated_image.view(generated_image.size(0), 1, 28, 28)
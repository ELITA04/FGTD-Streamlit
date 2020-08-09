import torch
import torch.nn as nn

class CGAN(nn.Module):
    '''
    The CGAN Network
    '''

    def __init__(self):
        super(CGAN, self).__init__()

        self.label_emb = nn.Embedding(10, 10)

        # First layer input -> (features + 10)
        self.linear_1 = nn.Sequential(
            nn.Linear(100 + 10, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Out 128

        self.linear_2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256, 0.8)
        )
        # Out 256

        self.linear_3 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512, 0.8)
        )
        # Out 512

        self.linear_4 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024, 0.8)
        )
        # Out 1024

        self.output_layer = nn.Sequential(
            nn.Linear(1024, 28 ** 2),
            nn.Tanh()
        )
        # Out 784

    def forward(self, noise, labels):
        #Concat label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        x = self.linear_1(gen_input)
        x = self.linear_2(x)
        x = self.linear_3(x)
        x = self.linear_4(x)
        output_image = self.output_layer(x)
        output_image = output_image.view(output_image.size(0), 1, 28, 28)
        return output_image
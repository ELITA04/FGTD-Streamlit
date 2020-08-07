import torch
import torch.nn as nn

class Generator(nn.Module):
    '''
    The Generator Network
    '''
    def __init__(self):
        super(Generator, self).__init__()
        self.image_size = 256
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_dim = 768
        self.projected_embed_dim = 256
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        self.ngf = self.image_size // 16

        self.projection = nn.Sequential(
            nn.Linear(in_features = self.embed_dim, out_features = self.projected_embed_dim),
            nn.BatchNorm1d(num_features = self.projected_embed_dim),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True)
        )

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 32, 4, 1, 0, bias = False),
            nn.BatchNorm2d(self.ngf * 32),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            # state size (ngf*4) x 4 x 4
            
            nn.ConvTranspose2d(self.ngf * 32, self.ngf * 16, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32

            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64

            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()

        )
        self.optimG = torch.optim.Adam(self.parameters(), lr = 0.0002, betas = (0.5, 0.999))

    def forward(self, noise, text_embeddings):
        encoded_text = self.projection(text_embeddings)
        concat_input = torch.cat([noise, encoded_text], dim = 1).unsqueeze(2).unsqueeze(2)
        output = self.layer(concat_input)
        return output

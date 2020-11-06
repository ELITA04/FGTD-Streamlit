import torch
import torch.nn as nn
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class Self_Attn(nn.Module):
    """
    Self attention layer
    """

    def __init__(self, in_dim):
        super().__init__()

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)

        attn = self.softmax(energy)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attn.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out, attn


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class SAGAN(nn.Module):
    """
    Generator architecture
    """

    def __init__(
        self,
        noise_size=100,
        embedding_size=768,
        reduced_dim_size=256,
        num_channels=3,
        image_size=128,
        attn=True,
        feature_size=64,
    ):
        super(SAGAN, self).__init__()
        self.attn = attn
        self.noise_dim = noise_size

        assert image_size in [
            64,
            128,
            256,
        ], "Can only generate images for 64, 128 and 256 dimensions."
        self.image_size = image_size

        ########## Embedding Layer ##########
        self.encode_layer = nn.Sequential(
            nn.Linear(embedding_size, reduced_dim_size),
            nn.BatchNorm1d(reduced_dim_size),
            nn.ReLU(),
            nn.Linear(reduced_dim_size, noise_size),
            nn.BatchNorm1d(noise_size),
            nn.ReLU(),
        )

        # Size 1 -> 4
        self.layer1 = nn.Sequential(
            SpectralNorm(
                nn.ConvTranspose2d(
                    in_channels=noise_size, out_channels=feature_size * 8, kernel_size=4
                )
            ),
            nn.BatchNorm2d(feature_size * 8),
            nn.ReLU(),
        )

        # Size 4 -> 8
        self.layer2 = nn.Sequential(
            SpectralNorm(
                nn.ConvTranspose2d(
                    in_channels=feature_size * 8,
                    out_channels=feature_size * 4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.BatchNorm2d(feature_size * 4),
            nn.ReLU(),
        )

        # Size 8 -> 16
        self.layer3 = nn.Sequential(
            SpectralNorm(
                nn.ConvTranspose2d(
                    in_channels=feature_size * 4,
                    out_channels=feature_size * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.BatchNorm2d(feature_size * 2),
            nn.ReLU(),
        )

        # Attn1 layer turn 128 dims -> 128 dims
        self.attn1 = Self_Attn(feature_size * 2)

        # Size 16 -> 32
        self.layer4 = nn.Sequential(
            SpectralNorm(
                nn.ConvTranspose2d(
                    in_channels=feature_size * 2,
                    out_channels=feature_size * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.BatchNorm2d(feature_size * 2),
            nn.ReLU(),
        )

        # Attn2 layer turn 64 dims -> 64 dims
        self.attn2 = Self_Attn(feature_size * 2)

        final_size = feature_size * 2

        if image_size >= 128:
            # Size 32 -> 64
            self.layer5 = nn.Sequential(
                SpectralNorm(
                    nn.ConvTranspose2d(
                        in_channels=feature_size * 2,
                        out_channels=feature_size * 2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                ),
                nn.BatchNorm2d(feature_size * 2),
                nn.ReLU(),
            )

            final_size = feature_size * 2

            # Attn3 layer turn 64 dims -> 64 dims
            self.attn3 = Self_Attn(feature_size * 2)

            if image_size == 256:
                # Size 64 -> 128
                self.layer6 = nn.Sequential(
                    SpectralNorm(
                        nn.ConvTranspose2d(
                            in_channels=feature_size * 2,
                            out_channels=feature_size,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        )
                    ),
                    nn.BatchNorm2d(feature_size),
                    nn.ReLU(),
                )

                final_size = feature_size

        # Output last_dim -> last_dim * 2
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(final_size, 3, 4, 2, 1), nn.Tanh()
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

    def forward(self, noise, text_embeddings):
        encoded_text = self.encode_layer(text_embeddings)
        # Concatenating the noise and encoded text
        concat_input = torch.mul(noise, encoded_text).view(-1, self.noise_dim, 1, 1)

        x = self.layer1(concat_input)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.attn:
            x, _ = self.attn1(x)
        x = self.layer4(x)
        if self.attn:
            x, _ = self.attn2(x)
        if self.image_size >= 128:
            x = self.layer5(x)
            if self.attn:
                x, _ = self.attn3(x)
            if self.image_size == 256:
                x = self.layer6(x)

        x = self.output_layer(x)
        return x

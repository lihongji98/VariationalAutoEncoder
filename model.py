import torch
import torch.nn.functional as F
from torch import nn

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim = 200, z_dim = 20):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        #decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encoder(self, x):
        h = self.img_2hid(x)
        h = self.relu(h)
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decoder(self, z):
        h = self.relu(self.z_2hid(z))
        img = torch.sigmoid(self.hid_2img(h))
        return img

    def forward(self, x):
        mu, sigma = self.encoder(x)
        epsilon = torch.randn_like(sigma)
        z_reparamatrized = mu + sigma * epsilon
        x_reconstructed = self.decoder(z_reparamatrized)
        return x_reconstructed, mu, sigma

if __name__ == "__main__":
    x = torch.randn(4, 28 * 28)
    vae = VariationalAutoEncoder(input_dim = 784)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)







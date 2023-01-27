from imports import *

"""
Variational Encoder network
"""
class VariationalEncoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        depths =   (16,8,4,2,1)
        # define layers
        self.flatten = nn.Flatten()
        self.initial_linear = nn.Linear(input_size*input_size, input_size*depths[0])
        self.encoder = nn.ModuleList([nn.Linear(input_size*depths[i], input_size*depths[i+1]) for i in range(len(depths)-1)])

        self.encoder_mu = nn.Linear(input_size*depths[-1], latent_size)
        self.encoder_logvar = nn.Linear(input_size*depths[-1], latent_size)
        # self.latent_linear = nn.Linear(input_size*depths[-1], latent_size)
        self.relu = nn.ReLU()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.flatten(x)
        x = self.initial_linear(x)
        for encode in self.encoder:
            x = self.relu(encode(x))
        # x = self.relu(self.latent_linear(x))
        mu, logvar = self.encoder_mu(x), self.encoder_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

"""
Decoder network
"""
class Decoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        depths =   (1,2,4,8,16)
        # define layers
        self.flatten = nn.Flatten()
        self.final_layer = nn.Linear(input_size*depths[-1], input_size*input_size)
        self.decoder = nn.ModuleList([nn.Linear(input_size*depths[i], input_size*depths[i+1]) for i in range(len(depths)-1)])
        self.latent_linear = nn.Linear(latent_size, input_size*depths[0])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.latent_linear(x))
        for encode in self.decoder:
            x = self.relu(encode(x))
        x = self.final_layer(x)
        x = self.sigmoid(x)
        return x

"""
Autoencoder network
"""
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = VariationalEncoder(input_size, latent_size=2)
        self.decoder = Decoder(input_size, latent_size=2)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        p_x = self.decoder(z)
        return p_x, mu, logvar

def VAE_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# from torchsummary import summary
# model = Autoencoder(28)
# summary(model, (1, 28, 28))
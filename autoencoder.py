from imports import *

"""
Encoder network
"""
class Encoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        depths =   (4,2,1)
        # define layers
        self.flatten = nn.Flatten()
        self.initial_linear = nn.Linear(input_size*input_size, input_size*depths[0])
        self.linear = nn.ModuleList([nn.Linear(input_size*depths[i], input_size*depths[i+1]) for i in range(len(depths)-1)])
        self.latent_linear = nn.Linear(input_size*depths[-1], latent_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.initial_linear(x)
        for encode in self.linear:
            x = self.relu(encode(x))
        x = self.relu(self.latent_linear(x))
        return x

"""
Decoder network
"""
class Decoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()
        depths =   (1,2,4)
        # define layers
        self.flatten = nn.Flatten()
        self.final_layer = nn.Linear(input_size*depths[-1], input_size*input_size)
        self.linear = nn.ModuleList([nn.Linear(input_size*depths[i], input_size*depths[i+1]) for i in range(len(depths)-1)])
        self.latent_linear = nn.Linear(latent_size, input_size*depths[0])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.latent_linear(x))
        for encode in self.linear:
            x = self.relu(encode(x))
        x = self.final_layer(x)
        x = self.relu(x)
        return x

"""
Autoencoder network
"""
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = Encoder(input_size, latent_size=2)
        self.decoder = Decoder(input_size, latent_size=2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# from torchsummary import summary
# model = Autoencoder(28)
# summary(model, (1, 28, 28))
from imports import *
from scipy.stats import norm

def create_datagen_figure(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = 12
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    figure = np.zeros((28*n, 28*n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = torch.tensor([[xi, yi]]).to(device)
            x_decoded = model.decoder(z_sample.float())
            digit = x_decoded.view(28,28).detach().cpu().numpy()
            figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = digit
    plt.imshow((figure*255.).astype('uint8'))
    plt.axis("off")
    plt.savefig('vae.png', transparent=True, bbox_inches='tight', pad_inches = 0)
    return figure

vae = torch.load("vae_model_5")
create_datagen_figure(vae)
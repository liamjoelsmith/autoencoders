from imports import *

# def plot_latent(autoencoder, data, num_batches=100):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     for i, (x, y) in enumerate(data):
#         z = autoencoder.encoder(x.to(device))
#         z = z.to('cpu').detach().numpy()
#         plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
#         if i > num_batches:
#             plt.colorbar()
#             break
#         plt.xlim(0,20)
#         plt.ylim(0,20)
#         plt.savefig("latent.png", bbox_inches='tight')

# autoencoder = torch.load("autoencoder_model_15")
# data = torch.utils.data.DataLoader(
#         torchvision.datasets.MNIST('./data',
#         transform=torchvision.transforms.ToTensor(), download=True),
#         batch_size=32, shuffle=False)
# plot_latent(autoencoder, data)

def visualise_latent_space(autoencoder, r0=(0, 7.5), r1=(0, 7.5), num_images=12):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_width = 28
    img = np.zeros((num_images*image_width, num_images*image_width))
    for i, yi in enumerate(np.linspace(*r1, num_images)):
        for j, xi in enumerate(np.linspace(*r0, num_images)):
            z = torch.Tensor([[xi, yi]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(num_images-1-i)*image_width:(num_images-1-i+1)*image_width, j*image_width:(j+1)*image_width] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    plt.axis("off")
    plt.savefig('reconstructed.png', transparent=True, bbox_inches='tight', pad_inches = 0)

autoencoder = torch.load("autoencoder_model_15")
visualise_latent_space(autoencoder)

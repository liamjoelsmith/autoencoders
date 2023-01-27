from imports import *

def visualise_latent_space(autoencoder, r0=(0, 7.5), r1=(0, 7.5), num_images=12):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_width = 28
    img = np.zeros((num_images*image_width, num_images*image_width))
    for i, y in enumerate(np.linspace(*r1, num_images)):
        for j, x in enumerate(np.linspace(*r0, num_images)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(num_images-1-i)*image_width:(num_images-1-i+1)*image_width, j*image_width:(j+1)*image_width] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    plt.axis("off")
    plt.savefig('reconstructed.png', transparent=True, bbox_inches='tight', pad_inches = 0)

autoencoder = torch.load("autoencoder_model_15")
visualise_latent_space(autoencoder)
from imports import *
from autoencoder import *
from dataset import *

plt.style.use('ggplot')

"""
Train the model and regularly save and plot the respective losses
"""
def train(trainloader):
    model = Autoencoder(input_size=28)
    # use a GPU whenever possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    epochs = 41

    # keep track of training and validation losses
    loss_vals = []

    for epoch in range(epochs):
        model.train(True)
        running_loss = 0.0
        for index, (x, y) in tqdm(enumerate(trainloader), total=len(trainloader)):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = loss_fn(torch.flatten(x_hat), torch.flatten(x))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'  [{epoch+1}, {index + 1:5d}] loss: {running_loss / len(trainloader):.4f}', flush=True)
        loss_vals.append(running_loss / len(trainloader))
        # val_vals.append(validate(model, valloader))

        # save model and update loss figure every 5 epochs
        if epoch % 5 == 0:
            torch.save(model, "autoencoder_model_" + str(epoch))

            # create loss graph
            fig, ax = plt.subplots()
            plt.plot(loss_vals, label="Training Loss")
            ax.set_ylabel('Loss')
            ax.set_title('Autoencoder Training Error')
            ax.set_xlabel('Epoch')
            ax.legend(loc="upper right")
            # plt.show()
            plt.savefig("loss" + str(epoch) + ".png") 

if __name__ == "__main__":
    print("Output-------------------------------------")
    print("Loading Dataset...", flush=True)
    train_dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
        transform=torchvision.transforms.ToTensor(), download=True),
        batch_size=32, shuffle=False)
    print("Loaded Dataset", flush=True)
    print("Training.....", flush=True)
    train(train_dataloader)
    print("Finished Training", flush=True)

import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from model import VariationalAutoEncoder
from torchvision import transforms
from torch.utils.data import DataLoader

#configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 28 * 28
H_DIM = 200
Z_DIM = 20
epochs = 20
batch_size = 32
lr = 3e-4 # Karpathy constant

# Dataset Loading
dataset = datasets.MNIST(root = "datasets/",
                            train = True,
                            transform = transforms.ToTensor(),
                            download = False)
train_loader = DataLoader(dataset = dataset,
                            batch_size = batch_size,
                            shuffle = True)
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr = lr)
loss_fn = nn.MSELoss(reduction = "sum")

#training
min_loss = torch.inf

for epoch in range(epochs):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        # forward
        x = x.to(device).view(x.shape[0], INPUT_DIM)
        x_reconstructed, mu, sigma = model(x)

        # compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = (-0.5) * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        # backprop
        loss = kl_div + reconstruction_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description("epoch:{}/{} step:{}/{}".format(epoch+1, epochs, i+1, len(train_loader)))
        loop.set_postfix(loss = loss.item())
    if loss.item() < min_loss:
        min_loss = loss.item()
        print("save model, mininum loss -> ", min_loss)
        torch.save(model.state_dict(),"model.pth")








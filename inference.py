import torch
from model import VariationalAutoEncoder
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

def inference(digit, num_example, model, dataset):
    images = []
    idx = 0
    for x,y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encoding_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encoder(images[d].view(1, 784))
        encoding_digit.append((mu, sigma))

    mu, sigma = encoding_digit[digit]
    for example in range(num_example):
        epsilon = torch.randn_like(sigma)
        z = mu + epsilon * sigma
        out = model.decoder(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"generated_numbers/{digit}_{example}.png")

if __name__ == "__main__":
    model = VariationalAutoEncoder(input_dim = 784, h_dim = 200, z_dim = 20)
    model.load_state_dict(torch.load("model.pth"))

    dataset = datasets.MNIST(root = "datasets/",
                            train = True,
                            transform = transforms.ToTensor(),
                            download = False)

    for idx in tqdm(range(10)):
        inference(idx, num_example = 2, model = model, dataset = dataset)
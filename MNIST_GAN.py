"""This is a file to learn a GAN on MNIST digits

This file is based on 
    https://deeplearning4j.org/generative-adversarial-network
and also some inspiration from
    https://github.com/prcastro/pytorch-gan/blob/master/MNIST%20GAN.ipynb
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid, save_image

# hyperparameters
batch_size = 100
learning_rate = 2e-4
noise_length = 100
print_step = 200
test_side = 4

image_size = 784
image_shape = (1, 28, 28)

# check whether we can use cuda or not
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

data_dir = "./MNIST"
train_data = datasets.MNIST(
                        root=data_dir,
                        train=True,
                        download=True,
                        transform=transform)

# create pytorch dataset
mnist_dataset = torch.utils.data.DataLoader(
                                        dataset=train_data,
                                        batch_size=batch_size,
                                        shuffle=True
                                    )


# for batch_img, batch_label in mnist_dataset:
#     pass


class Generator(nn.Module):
    def __init__(self, input_size, image_size, image_shape):
        super(Generator, self).__init__()
        self.image_shape = image_shape
        self.main = nn.Sequential(
            nn.Linear(input_size, 256), nn.LeakyReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 512), nn.LeakyReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, 1024), nn.LeakyReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, image_size)
        )

    def forward(self, z_noise):
        return self.main(z_noise).view(-1, *self.image_shape)


class Discriminator(nn.Module):
    def __init__(self, img_size, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.main = nn.Sequential(
            nn.Linear(img_size, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, img):
        return self.main(img.view(-1, self.image_size))


def get_noise_sampler():
    return lambda m, n: torch.rand(m, n)


def train():
    D = Discriminator(image_size, image_size).to(device)
    G = Generator(noise_length, image_size, image_shape).to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)

    epochs = 100
    loss_function = nn.BCELoss()
    noise_sampler = get_noise_sampler()
    test_noise = noise_sampler(test_side**2, noise_length).to(device)

    real_loss_buffer = []
    fake_loss_buffer = []
    g_loss_buffer = []
    for e in range(epochs):
        for i, (batch_img, batch_label) in enumerate(mnist_dataset):
            batch_img = batch_img.to(device)
            batch_label = batch_label.to(device)

            # Train D
            D.zero_grad()
            real_score = D(batch_img)
            real_loss = loss_function(real_score, torch.ones_like(real_score))
            real_loss_buffer.append(real_loss)
            real_loss.backward()
            d_optimizer.step()

            D.zero_grad()
            noise_data = noise_sampler(batch_size, noise_length).to(device)
            fake_data = G(noise_data)
            fake_score = D(fake_data.t())
            fake_loss = loss_function(fake_score, torch.zeros_like(fake_score))
            fake_loss_buffer.append(fake_loss)
            fake_loss.backward()
            d_optimizer.step()

            # Train G
            G.zero_grad()
            noise_data = noise_sampler(batch_size, noise_length).to(device)
            fake_data = G(noise_data)
            g_score = D(fake_data.t())
            g_loss = loss_function(g_score, torch.ones_like(g_score))
            g_loss_buffer.append(g_loss)
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % print_step == 0:
                print("Epoch {}, iteration {}".format(e+1, i+1))
                test_images = G(test_noise)
                save_image(test_images,
                           "mnist_test_{}_{}.png".format(e+1, i+1),
                           nrow=test_side,
                           padding=10)



if __name__ == "__main__":
    train()
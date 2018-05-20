"""Simplest possible GAN

Code based on https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f

"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

input_size = 10
hidden_size = 50
output_size = 1
learning_rate = 2e-4
batch_size = 100
test_num = 30000

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  #
    # Gaussian


def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n) # Uniform-dist data into generator, _NOT_ Gaussian


class Generator(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(*[
            nn.Linear(input_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, output_size)
        ])

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(*[
            nn.Linear(input_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, 1), nn.Sigmoid()
        ])

    def forward(self, input):
        return self.main(input)


def make_histogram(data_real, data_fake, i, n_bins=100):
    fig = plt.figure()
    axes = plt.subplot(111)
    n, bins, patches = plt.hist(data_real, n_bins, normed=1, facecolor='green',
                                alpha=0.75)
    n, bins, patches = plt.hist(data_fake, bins, normed=1, facecolor='blue',
                                alpha=0.75)
    axes.set_xlim(-2, 5)
    axes.set_ylim(0, 1)
    fig.savefig("iteration_{}.png".format(i+1))
    plt.close(fig)


def main():
    G = Generator(input_size, hidden_size, output_size)
    D = Discriminator(batch_size, hidden_size, output_size)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)

    iterations = 30000
    d_loop = 1
    mu = 2
    sigma = 1
    print_step = 200

    loss_function = nn.BCELoss()

    data_sampler = get_distribution_sampler(mu, sigma)
    test_data = data_sampler(test_num)

    noise_sampler = get_generator_input_sampler()

    real_loss_buffer = []
    fake_loss_buffer = []
    g_loss_buffer = []
    for i in range(iterations):
        # train D
        for _ in range(d_loop):
            D.zero_grad()

            real_data = data_sampler(batch_size)
            real_score = D(real_data)
            real_loss = loss_function(real_score, torch.ones_like(real_score))
            real_loss_buffer.append(real_loss)
            real_loss.backward()
            d_optimizer.step()

            D.zero_grad()
            noise_data = noise_sampler(batch_size, input_size)
            fake_data = G(noise_data)
            fake_score = D(fake_data.t())
            fake_loss = loss_function(fake_score, torch.zeros_like(fake_score))
            fake_loss_buffer.append(fake_loss)
            fake_loss.backward()

            d_optimizer.step()

        # train G
        G.zero_grad()

        noise_data = noise_sampler(batch_size, input_size)
        fake_data = G(noise_data)
        g_score = D(fake_data.t())
        g_loss = loss_function(g_score, torch.ones_like(g_score))
        g_loss_buffer.append(g_loss)
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % print_step == 0:
            print("{:>6} -> D_real_loss {:.6f}  D_fake_loss {:.6f}  "
                  "G_fake_loss {:.6f}".format(
                i+1,
                real_loss_buffer[-1],
                fake_loss_buffer[-1],
                g_loss_buffer[-1]
            ))
            fake_data = torch.Tensor([])
            for _ in range(test_num // batch_size):
                noise_data = noise_sampler(batch_size, input_size)
                fake_data = torch.cat((fake_data, G(noise_data).t()), 1)
            fake_data = fake_data[:, :test_num]
            make_histogram(test_data.data[0,:].numpy(), fake_data.data[
                0,:].numpy()
                           , i)

    print("End of training")
    print("Show errors")
    fig0 = plt.figure(0)
    plt.subplot(311)
    plt.plot(real_loss_buffer)
    plt.title("D real loss")

    plt.subplot(312)
    plt.plot(fake_loss_buffer)
    plt.title("D fake loss")

    plt.subplot(313)
    plt.plot(g_loss_buffer)
    plt.title("G fake loss")
    plt.show()
    # fig0.close()

    print("Show distributions")
    num_samples = 2000
    fig1 = plt.figure(1)
    plt.subplot(121)
    real_data = data_sampler(num_samples).view(num_samples)
    n, bins, patches = plt.hist(real_data, 50, normed=1, facecolor='g',
                                alpha=0.75)
    plt.title("Real distribution")

    plt.subplot(122)
    fake_data = []
    for k in range(num_samples // batch_size):
        fake_data.extend    (G(noise_sampler(batch_size, input_size)).view(
            batch_size).data)
    n, bins, patches = plt.hist(fake_data, 50, normed=1, facecolor='g',
                                alpha=0.75)
    plt.title("Fake distribution")
    plt.show()
    print("end")





if __name__ == '__main__':
    main()
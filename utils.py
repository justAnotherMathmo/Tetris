import math
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import params

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# Means that program doesn't stop to show the graph
plt.ion()


def curr_eps(steps):
    # return 0
    return params.EPS_END + (params.EPS_START - params.EPS_END) * math.exp(-1. * steps / params.EPS_DECAY)


def create_state(grid, current_piece, next_piece):
    fgrid = np.zeros((20, 10))
    for i, row in enumerate(grid):
        for j in range(3, 13):
            fgrid[i][j - 3] = 1.0 if row & (1 << j) else 0.0
    return fgrid


def get_screen(grid, device):
    screen = create_state(grid, None, None)

    # Resize and add a batch dimension
    tensor = torch.from_numpy(screen).unsqueeze(0).unsqueeze(0)
    # Push to floats on GPU
    return tensor.type(torch.FloatTensor).to(device)


def plot_durations(episode_durations, lines_cleared, eps, save=None):
    fig = plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    plt.plot(np.array(lines_cleared) * 50)
    plt.plot(np.array(eps) * 200)

    # Take 200 episode averages and plot them too
    if len(durations_t) >= 200:
        means = durations_t.unfold(0, 200, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(199), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

    if save is not None:
        fig.savefig(save, bbox_inches='tight')

    if is_ipython:
        display.clear_output(wait=True)
        plt.show()
    else:
        plt.show()


if params.TENSORBOARD_LOGGING:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

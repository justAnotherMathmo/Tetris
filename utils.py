import math
import torch
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import params

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# Means that program doesn't stop to show the graph
plt.ion()


def _raw_eps(steps):
    return params.EPS_END + (params.EPS_START - params.EPS_END) * math.exp(-1. * steps / params.EPS_DECAY)


def curr_eps(steps):
    # return 0
    raw_eps = _raw_eps(steps)
    if not params.CYCLE_RANDOMNESS:
        return raw_eps
    else:
        decay_cycle = (params.EPS_DECAY * 10)
        return raw_eps + 0.2 * _raw_eps((steps + (decay_cycle // 2)) % decay_cycle)


# def create_state(grid, next_piece):
#     fgrid = np.zeros((20, 10))
#     for i, row in enumerate(grid):
#         for j in range(3, 13):
#             fgrid[i][j - 3] = 1.0 if row & (1 << j) else 0.0
#     return fgrid


def create_state(grid, next_piece):
    fgrid = np.zeros((20, 10))
    pgrid = np.zeros((4, 4))
    for i, row in enumerate(grid):
        for j in range(3, 13):
            fgrid[i][j - 3] = 1.0 if row & (1 << j) else 0.0
    if next_piece is not None:
        for i, row in enumerate(next_piece):
            for j in range(4):
                pgrid[i][j] = 1.0 if row & (1 << j) else 0.0
    return fgrid, pgrid


def get_screen(grid, device, piece=None):
    screen, piece = create_state(grid, piece)

    # Resize and add a batch dimension
    tensor = torch.from_numpy(screen).unsqueeze(0).unsqueeze(0)
    if piece is not None:
        out_piece = torch.from_numpy(piece).unsqueeze(0).unsqueeze(0)
    else:
        out_piece = None
    # Push to floats on GPU
    return tensor.type(torch.FloatTensor).to(device), out_piece.type(torch.FloatTensor).to(device)


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

#
# if params.TENSORBOARD_LOGGING:
#     from torch.utils.tensorboard import SummaryWriter
#     writer = SummaryWriter()


def tblogger(name, value, writer, is_random=False):
    if not params.TENSORBOARD_LOGGING:
        return
    if not is_random or (random.random() > 0.95):
        writer.add_scalar(name, value)

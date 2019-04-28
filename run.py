import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from tetris.Tetris import Tetris, O

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = Tetris(pieces=[O])
env.lockout_rate = 0
env.start_drop_rate = 1
env.reset()
NUM_STATES = len(env.actions)

steps_done = 0


def curr_eps(steps):
    return 0
#     return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / EPS_DECAY)


def create_state(grid, current_piece, next_piece):
    fgrid = np.zeros((20, 10))
    for i, row in enumerate(grid):
        for j in range(3, 13):
            fgrid[i][j - 3] = 1.0 if row & (1 << j) else 0.0
    return fgrid


def get_screen(grid, human=False):
    screen = create_state(grid, None, None)

    # Resize and add a batch dimension
    tensor = torch.from_numpy(screen).unsqueeze(0).unsqueeze(0)
    # Push to floats on GPU
    return tensor.type(torch.FloatTensor).to(device)

# Get screen size so that we can initialize layers correctly based on shape
init_screen = get_screen(env.get_grid())
_, _, screen_height, screen_width = init_screen.shape

load_net_prefix = './models/tetrisBotHackedConv2v'
load_net_number = 0
net_to_load = f'{load_net_prefix}{load_net_number}'
try:
    policy_net = torch.load(net_to_load)
    policy_net.eval()
    target_net = torch.load(net_to_load)
    target_net.eval()
    print(f'{net_to_load} loaded...')
except:
    policy_net = DQN(screen_height, screen_width).to(device)
    target_net = DQN(screen_height, screen_width).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    print(f'Fell back to creating a new net...')

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(1000000)


def select_action(state, deterministic=False):
    global steps_done
    sample = random.random()
    eps_threshold = curr_eps(steps_done)
    steps_done += 1
    if sample > eps_threshold and not deterministic:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net.eval()(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(NUM_STATES)]], device=device, dtype=torch.long)


episode_durations = []
lines_cleared = []
eps_values = []


def plot_durations(save=None):
    fig = plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    plt.plot(np.array(lines_cleared) * 50)
    #     plt.plot(np.array(eps_values) * 500)
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

    if save is not None:
        fig.savefig(save, bbox_inches='tight')

    if is_ipython:
        display.clear_output(wait=True)
        plt.show()


def compute_loss_single(state, action, next_state, reward):
    return _compute_loss(state, action, next_state, reward, batch_size=1)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    next_state_batch = torch.cat(batch.next_state)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    loss = _compute_loss(state_batch, action_batch, next_state_batch, reward_batch)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def _compute_loss(_state, _action, _next_state, _reward, batch_size=BATCH_SIZE):
    state_action_values = policy_net(_state).gather(1, _action)
    next_state_values = target_net(_next_state)[0][policy_net(_next_state).argmax(1)[0]].detach()
    expected_state_action_values = (next_state_values * GAMMA) + _reward
    return F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))


def did_piece_fall(this_env):
    return this_env.placing_frame


def get_height(this_env):
    return np.argmin(this_env._grid[3:] != 57351)


def create_reward(this_env, block_placed, action, is_done,
                  old_height, old_lines, include_height=True, include_score=True):
    if not block_placed:
        # Punish a little for doing something that isn't the empty move, or down
        if action == 0:
            return 0
    if is_done:
        return -10.0

    total_reward = 0
    if include_height:
        this_height = get_height(this_env)
        if this_height > old_height:
            # Punish a little more the closer you are to the top
            total_reward += (1 + this_height / 10) * (old_height - this_height) / 4

    line_diff = env._total_lines_cleared - old_lines
    if include_score and line_diff != 0:
        total_reward += 2 ** (line_diff + 1)

    return total_reward


def train(num_episodes=1000, human=False):
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        height, lines = 0, 0
        env.reset()
        last_state = get_screen(env.get_grid(), human=human)
        state = get_screen(env.get_grid(), human=human)
        hole_count = 0
        hole_reward = 0
        tower_count = 0
        tower_reward = 0
        if not human:
            state_array = [last_state] * MULTISTEP_PARAM
            reward_array = [0] * MULTISTEP_PARAM

            reward_sum = 0
            array_pos = 0
            next_array_pos = 1
            warmup = 1
        for t in count():

            # Select and perform an action
            action = select_action(state, deterministic=human)
            # Can only perform an action once every three frames anyway...
            state, _, done = env.step(action.item())
            piece_fell = did_piece_fall(env)
            #             if not done:
            #                 state, _, done = env.step(0)
            #                 piece_fell = (piece_fell or did_piece_fall(env))
            #             if not done:
            #                 state, _, done = env.step(0)
            #                 piece_fell = (piece_fell or did_piece_fall(env))

            # Observe new state
            state = get_screen(state, human)

            if not human:
                state_array[array_pos] = state

                reward_single = create_reward(env, piece_fell, action, done, height, lines)
                reward_sum = (MULISTEP_GAMMA * reward_sum) + reward_single - (MULISTEP_GAMMA ** MULTISTEP_PARAM) * \
                             reward_array[array_pos]
                reward_array[array_pos] = reward_single
                reward_sum = torch.tensor([reward_sum], device=device).type(torch.float)

                # Store the transition in memory
                if warmup > MULTISTEP_PARAM:
                    #                     with torch.no_grad():
                    #                         loss = compute_loss_single(state_array[next_array_pos], action, state, reward_sum) ** ((1 - curr_eps(steps_done)) / 2 + 0.05)
                    #                     memory.push(state_array[next_array_pos], action, state, reward_sum, bias=np.array([loss.cpu()])[0])
                    memory.push(state_array[next_array_pos], action, state, reward_sum)

                # Perform one step of the optimization (on the target network)
                if (warmup + 1) % TRAIN_RATE == 0:
                    optimize_model()
                if done:
                    episode_durations.append(t + 1)
                    lines_cleared.append(lines)
                    eps_values.append(curr_eps(steps_done))
                    plot_durations('latestConv.png')
                    break

            else:
                if done:
                    break

            # Set up params for next cycle
            height = get_height(env)
            lines = env._total_lines_cleared
            last_state = state
            if not human:
                array_pos = (array_pos + 1) % MULTISTEP_PARAM
                next_array_pos = (next_array_pos + 1) % MULTISTEP_PARAM
                warmup += 1

        if not human:
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())


def watch_model(rounds=1000):
    with torch.no_grad():
        train(rounds, human=True)


idx = 6
while True:
    train(5000)
    torch.save(policy_net, f'{load_net_prefix}{idx}')
    idx += 1

watch_bot_tetris(action_func, pieces=[O])
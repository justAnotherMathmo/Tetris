# Misc Python Imports
import random
import numpy as np
import itertools

# Game Imports
from tetris.Tetris import Tetris, O

# Local Imports
import params
import model
import utils
import memory
import reward
import loss


# Torch Imports & initialisation
import torch
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Game initialisation
env = Tetris(pieces=[O])
env.lockout_rate = 0
env.start_drop_rate = 1
env.reset()
num_states = len(env.actions) - 2  # Get rid of hard + soft drop for now...

# Get screen size so that we can initialize layers correctly based on shape
init_screen = utils.get_screen(env.get_grid(), device)
_, _, screen_height, screen_width = init_screen.shape

# Attempt to load a net - if not make a new one
load_net_prefix = './models/resConvNoisy'
load_net_number = 0
net_to_load = f'{load_net_prefix}{load_net_number}'
try:
    policy_net = torch.load(net_to_load)
    policy_net.eval()
    target_net = torch.load(net_to_load)
    target_net.eval()
    print(f'{net_to_load} loaded...')
except FileNotFoundError:
    policy_net = model.DQN(screen_height, screen_width, num_states).to(device)
    target_net = model.DQN(screen_height, screen_width, num_states).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    print(f'Fell back to creating a new net...')

# Set up optimizer and memory
optimizer = optim.Adam(policy_net.parameters(), lr=params.LEARNING_RATE)
model_memory = memory.ReplayMemory(params.MEMORY_SIZE)


def select_action(state, eps_threshold):
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # # found, so we pick action with the larger expected reward.
            # writer.add_scalar('Q0', policy_net.eval()(state)[0, 0])
            return policy_net.eval()(state)[0].max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(num_states)]], device=device, dtype=torch.long)


# Yes yes, I'm a terrible person - I'll clean this up later
steps_done = 0
episode_durations = []
lines_cleared = []
eps_values = []


def train(num_episodes=1000, human=False):
    global steps_done

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        height, old_height, lines, old_lines = 0, 0, 0, 0
        env.reset()
        last_state = utils.get_screen(env.get_grid(), device)
        state = utils.get_screen(env.get_grid(), device)

        # if not human:
        state_array = [last_state] * params.MULTISTEP_PARAM
        reward_array = [0] * params.MULTISTEP_PARAM
        reward_sum = 0
        array_pos = 0
        next_array_pos = 1
        warmup = 0

        for t in itertools.count():
            steps_done += 1
            # Select and perform an action
            if human:
                eps = 0
            else:
                eps = utils.curr_eps(steps_done)
            action = select_action(state, eps)
            state, _, done = env.step(action.item())
            piece_fell = reward.did_piece_fall(env)

            # Observe new state
            state = utils.get_screen(state, device)

            if not human:
                state_array[array_pos] = state

                # Create reward
                height = reward.get_height(env)
                lines = env._total_lines_cleared
                reward_single = reward.create_reward(piece_fell, action, done, height, old_height, lines, old_lines)
                reward_sum = (params.MULISTEP_GAMMA * reward_sum) + reward_single - (params.MULISTEP_GAMMA ** params.MULTISTEP_PARAM) * reward_array[array_pos]
                reward_array[array_pos] = reward_single
                reward_sum = torch.tensor([reward_sum], device=device).type(torch.float)

                # Store the transition in memory
                if warmup > params.MULTISTEP_PARAM:
                    model_memory.push(state_array[next_array_pos], action, state, reward_sum)

                # Perform one step of the optimization (on the target network)
                if (warmup + 1) % params.TRAIN_RATE == 0:
                    loss.optimize_model(optimizer, model_memory, policy_net, target_net)
                if done or t > 5000:
                    # 5000 here just to stop us playing forever...
                    episode_durations.append(t + 1)
                    lines_cleared.append(lines)
                    eps_values.append(eps)
                    utils.plot_durations(episode_durations, lines_cleared, eps_values, save='latest.png')
                    break

            else:
                if done:
                    break

            # Set up params for next frame
            old_height = height
            old_lines = lines
            if not human:
                array_pos = (array_pos + 1) % params.MULTISTEP_PARAM
                next_array_pos = (next_array_pos + 1) % params.MULTISTEP_PARAM
                warmup += 1

        if not human:
            # Update the target network, copying all weights and biases in DQN
            if i_episode % params.TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())


def watch_model(rounds=1000):
    with torch.no_grad():
        train(rounds, human=True)


# Train and save the model at intervals
idx = 0
while True:
    train(5000)
    torch.save(policy_net, f'{load_net_prefix}{idx}')
    idx += 1

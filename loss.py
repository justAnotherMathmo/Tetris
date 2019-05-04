# Local imports
import params
import memory

# Torch Imports
import torch
import torch.nn.functional as F


def compute_loss_single(state, action, next_state, reward, policy_net, target_net,):
    return _compute_loss(state, action, next_state, reward, policy_net, target_net, batch_size=1)


def optimize_model(optimizer, model_memory, policy_net, target_net):
    if len(model_memory) < params.BATCH_SIZE:
        return

    transitions = model_memory.sample(params.BATCH_SIZE)
    batch = memory.Transition(*zip(*transitions))
    next_state_batch = torch.cat(batch.next_state)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    loss = _compute_loss(state_batch, action_batch, next_state_batch, reward_batch, policy_net, target_net)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def _compute_loss(_state, _action, _next_state, _reward, policy_net, target_net, batch_size=params.BATCH_SIZE):
    state_action_values, expected_state = policy_net(_state)
    state_action_values = state_action_values.gather(1, _action)

    # Compute value loss (traditional DQN)
    next_state_values = target_net(_next_state)[0].gather(1, policy_net(_next_state)[0].argmax(1).view(-1, 1)).detach()

    expected_state_action_values = (next_state_values * params.GAMMA) + _reward.unsqueeze(1)
    value_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Compute state loss
    state_loss = F.binary_cross_entropy(expected_state, _next_state)

    return value_loss + state_loss * params.STATE_VALUE_LOSS_RATIO

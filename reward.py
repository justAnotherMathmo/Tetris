import numpy as np
import params


def did_piece_fall(env):
    return env.placing_frame


def get_height(env):
    return np.argmin(env._grid[3:] != 57351)


def create_reward(block_placed, action, is_done, new_height, old_height, new_lines, old_lines,
                  include_height=True, include_score=True, include_death=True):
    if not block_placed:
        # Punish a little for doing something that isn't the empty move, or down
        if action in [0, 5, 6]:
            return 0
        else:
            return params.MOVEMENT_COST

    if include_death and is_done:
        return -10.0

    total_reward = 0
    if include_height and (new_height > old_height):
        # Punish a little more the closer you are to the top
        total_reward += (1 + new_height / 10) * (old_height - new_height) / 4

    line_diff = new_lines - old_lines
    if include_score and line_diff != 0:
        total_reward += 2 ** (line_diff + 1)

    return total_reward

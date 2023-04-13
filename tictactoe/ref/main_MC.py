import numpy as np
from env import TicTacToeEnv
from MC import random_policy, first_visit_mc_policy_evaluation

env = TicTacToeEnv()
action_value = first_visit_mc_policy_evaluation(env, random_policy, episodes=50000)

initial_state = env.reset()
empty_cells = np.where(initial_state == 0)

# Get action values for all empty cells
action_values = [action_value[empty_cells[0][i], empty_cells[1][i], 3 * empty_cells[0][i] + empty_cells[1][i]] for i in range(len(empty_cells[0]))]
best_action_index = np.argmax(action_values)
best_action = (empty_cells[0][best_action_index], empty_cells[1][best_action_index])

print(f"Best first move for the random policy: {best_action}")

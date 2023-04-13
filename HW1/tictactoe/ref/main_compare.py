import numpy as np
import random
from env import TicTacToeEnv
from MC import first_visit_mc_policy_evaluation
from TD import td_zero
import matplotlib.pyplot as plt

def random_policy(env):
    valid_actions = np.where(env.board.flatten() == 0)[0]
    return random.choice(valid_actions)




total_episode=5000

env = TicTacToeEnv()

# Monte Carlo
mc_action_value,mc_action_value_change  = first_visit_mc_policy_evaluation(env, random_policy, episodes=total_episode)
print(mc_action_value[:,:,0])
initial_state = env.reset()
empty_cells = np.where(initial_state == 0)
mc_action_values = [mc_action_value[empty_cells[0][i], empty_cells[1][i], 3 * empty_cells[0][i] + empty_cells[1][i]] for i in range(len(empty_cells[0]))]
mc_best_action_index = np.argmax(mc_action_values)
mc_best_action = (empty_cells[0][mc_best_action_index], empty_cells[1][mc_best_action_index])

# TD(0)
td_action_value,td_action_value_change  = td_zero(env, random_policy, episodes=total_episode)
td_action_values = [td_action_value[empty_cells[0][i], empty_cells[1][i], 3 * empty_cells[0][i] + empty_cells[1][i]] for i in range(len(empty_cells[0]))]
td_best_action_index = np.argmax(td_action_values)
td_best_action = (empty_cells[0][td_best_action_index], empty_cells[1][td_best_action_index])

print(f"Best first move for the random policy (Monte Carlo): {mc_best_action}")
print(f"Best first move for the random policy (TD(0)): {td_best_action}")

# Plot the action value change
plt.figure(figsize=(10,5))
plt.plot(mc_action_value_change, label='MC')
plt.plot(td_action_value_change, label='TD(0)')
plt.xlabel('Episodes')
plt.ylabel('Action Value Change')
plt.legend()
plt.show()
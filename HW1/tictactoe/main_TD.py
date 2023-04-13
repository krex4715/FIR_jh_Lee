import matplotlib.pyplot as plt
from env import TicTacToeEnv
import random
import numpy as np
from copy import deepcopy


def available_actions(board):
    actions = []
    for i in range(9):
        row, col = i // 3, i % 3
        if board[row, col] == 0:
            actions.append(i)
    return actions

def state_to_tuple(board):
    return tuple(board.flatten())





def main_TD(env, alpha = 0.1, total_eps = 10000, rendering=False):
    done = False

    if rendering==True:
        fig, ax = plt.subplots()
        plt.ion()

    td_convergence = []
    prev_q_table = deepcopy(env.q_table)
    prev_q_table[(0, 0, 0, 0, 0, 0, 0, 0, 0)] = np.zeros(9) # initialize q_table for empty board
    prev_q_table = prev_q_table[(0, 0, 0, 0, 0, 0, 0, 0, 0)]
    for _ep in range(total_eps):
        print("Episode: ", _ep, "/", total_eps, end="\r")
        observation = env.reset()

        # run episode
        while True:
            actions = available_actions(env.board)
            action = random.choice(actions)
            next_observation, reward, done, information = env.step(action)

            if rendering:
                env.render(ax=ax)
                plt.pause(0.5)

            # TD(0) update
            state_tuple = state_to_tuple(observation)
            next_state_tuple = state_to_tuple(next_observation)

            if state_tuple not in env.q_table:
                env.q_table[state_tuple] = np.zeros(9)
            if next_state_tuple not in env.q_table:
                env.q_table[next_state_tuple] = np.zeros(9)

            env.q_table[state_tuple][action] += alpha * (reward + 1.0 * np.max(env.q_table[next_state_tuple]) - env.q_table[state_tuple][action])

            observation = next_observation

            if reward == 1:
                if rendering:
                    print(f"Player {env.current_player} wins!")
                break
            elif env.is_full():
                if rendering:
                    print("It's a draw!")
                break

            if done:
                break
            
        # TD(0) Convergence check
        diff = np.sum(np.abs(env.q_table[(0, 0, 0, 0, 0, 0, 0, 0, 0)] - prev_q_table))
        td_convergence.append(diff)
        prev_q_table = env.q_table[(0, 0, 0, 0, 0, 0, 0, 0, 0)].copy()



    if rendering:
        plt.ioff()
        plt.show()
        
    return td_convergence




if __name__ == "__main__":
    env = TicTacToeEnv()
    td_convergence = main_TD(env, alpha = 0.1, total_eps = 1000, rendering=False)

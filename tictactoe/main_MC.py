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






def main_MC(env, total_eps = 10000, rendering=False):
    done = False


    if rendering==True:
        fig, ax = plt.subplots()
        plt.ion()
    mc_convergence = []
    prev_q_table = deepcopy(env.q_table)
    prev_q_table[(0, 0, 0, 0, 0, 0, 0, 0, 0)] = np.zeros(9) # initialize q_table for empty board
    prev_q_table = prev_q_table[(0, 0, 0, 0, 0, 0, 0, 0, 0)]
    for _ep in range(total_eps):
        print("Episode: ", _ep, "/", total_eps, end="\r")
        observation = env.reset()
        episode_history = []

        # run episode
        while True:
            actions = available_actions(env.board)
            action = random.choice(actions)
            next_observation, reward, done, information = env.step(action)
            episode_history.append((observation, action, reward))
            observation = next_observation

            if rendering:
                env.render(ax=ax)
                plt.pause(0.5)

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

        G = 0
        # calculate Action-Value Function first-visit MC q(s,a)
        visited_pairs = set()
        for t in reversed(range(len(episode_history))):
            state, action, reward = episode_history[t]
            G = reward + 1.0 * G

            # for memory efficiency, convert state to tuple
            state_tuple = state_to_tuple(state)
            sa_pair = (state_tuple, action)

            # First-visit MC
            if sa_pair not in visited_pairs:
                visited_pairs.add(sa_pair)

                if state_tuple not in env.q_table:
                    env.q_table[state_tuple] = np.zeros(9)
                    env.visit_counts[state_tuple] = np.zeros(9)

                env.visit_counts[state_tuple][action] += 1
                alpha = 1.0 / env.visit_counts[state_tuple][action]
                env.q_table[state_tuple][action] += alpha * (G - env.q_table[state_tuple][action])

        # MC convergence check
        diff = np.sum(np.abs(env.q_table[(0, 0, 0, 0, 0, 0, 0, 0, 0)] - prev_q_table))
        mc_convergence.append(diff)
        prev_q_table = env.q_table[(0, 0, 0, 0, 0, 0, 0, 0, 0)].copy()

    init_q = env.q_table[(0, 0, 0, 0, 0, 0, 0, 0, 0)]
    print("initial state q_table : ", init_q)

    # grid position of q_table
    max = np.argmax(init_q)
    row, col = max // 3, max % 3
    print(f"Best First Move : {row} ,{col}")

    if rendering:
        plt.ioff()
        plt.show()

    return mc_convergence,init_q.copy()
    
    






if __name__=="__main__":
    env=TicTacToeEnv()
    mc_convergence,_ = main_MC(env, total_eps=10000, rendering=False)
    
    plt.figure(figsize=(10, 5))
    plt.plot(mc_convergence)
    plt.show()
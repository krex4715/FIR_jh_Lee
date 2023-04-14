import matplotlib.pyplot as plt
from env import TicTacToeEnv
import random
import numpy as np
from copy import deepcopy
import matplotlib.colors as mcolors



def available_actions(board):
    actions = []
    for i in range(9):
        row, col = i // 3, i % 3
        if board[row, col] == 0:
            actions.append(i)
    return actions

def state_to_tuple(board):
    return tuple(board.flatten())

def get_symmetric_states(state):
    state_2d = np.array(state).reshape(3, 3)
    symmetric_states = [state_2d]

    # Rotatation 90, 180, 270 degrees
    for _ in range(3):
        state_2d = np.rot90(state_2d)
        symmetric_states.append(state_2d)

    # Mirroring
    mirrored_state = np.fliplr(state_2d)
    symmetric_states.append(mirrored_state)
    for _ in range(3):
        mirrored_state = np.rot90(mirrored_state)
        symmetric_states.append(mirrored_state)

    return [tuple(s.flatten()) for s in symmetric_states]






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

                # if state_tuple not in env.q_table:
                #     env.q_table[state_tuple] = np.zeros(9)
                #     env.visit_counts[state_tuple] = np.zeros(9)

                if state_tuple not in env.q_table:
                    symmetric = get_symmetric_states(state)
                    symmetric_found = False
                    for s in symmetric:
                        if s in env.q_table:
                            state_tuple = s
                            symmetric_found = True
                            break
                    if not symmetric_found:
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
    mc_convergence,mc_init_q = main_MC(env, total_eps=50000, rendering=False)
    
    
    mc_init_q_2d = np.array(mc_init_q).reshape(3, 3)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # First Graph
    ax1.set_title('Initial Q-Table')
    min_val = mc_init_q_2d.min()
    max_val = mc_init_q_2d.max()
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    cax = ax1.matshow(mc_init_q_2d, cmap='coolwarm', norm=norm)
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f"{mc_init_q_2d[i, j]:.5f}", ha="center", va="center", color="w")
    ax1.set_xticks(np.arange(-0.5, 2.5, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, 2.5, 1), minor=True)
    ax1.grid(which="minor", color="black", linewidth=2)
    ax1.set_xticks([])
    ax1.set_yticks([])
        

    # Second Graph
    # only first 5000 episodes to show convergence
    ax2.set_title('Convergence of MC')
    ax2.plot(mc_convergence[0:5000] ,label="MC")
    ax2.legend()
    plt.show()
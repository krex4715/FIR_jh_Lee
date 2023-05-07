import matplotlib.pyplot as plt
from env import TicTacToeEnv
import random


env = TicTacToeEnv()

fig, ax = plt.subplots()
plt.ion()
rendering=True

done = False

def available_actions(board):
    actions = []
    for i in range(9):
        row, col = i // 3, i % 3
        if board[row, col] == 0:
            actions.append(i)
    return actions



while not done:
    actions = available_actions(env.board)
    action = random.choice(actions)

    _, reward, done, _ = env.step(action)
    if rendering == True:
        env.render(ax=ax)
        plt.pause(0.5)

    if reward == 1:
        print(f"Player {env.current_player} wins!")
        break
    elif env.is_full():
        print("It's a draw!")
        break

if rendering==True:
    plt.ioff()
    plt.show()

import matplotlib.pyplot as plt
from env import TicTacToeEnv

env = TicTacToeEnv()

fig, ax = plt.subplots()
plt.ion()
rendering=True

done = False
while not done:
    action = env.action_space.sample()

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

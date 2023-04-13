import gym
import numpy as np
import matplotlib.pyplot as plt

class TicTacToeEnv(gym.Env):
    def __init__(self):
        self.q_table = {}
        self.visit_counts = {}
        self.board = np.zeros((3, 3))
        self.action_space = gym.spaces.Discrete(9)
        self.reset()

    def step(self, action):
        row, col = action // 3, action % 3

        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            if self.check_win(self.current_player):
                reward = 1
                information = "Win"
                done = True
            elif self.is_full():
                reward = 0
                information = "Draw"
                done = True
            else:
                reward = 0
                done = False
                information = "Continue"
                self.current_player = -self.current_player
        else:
            reward = 0
            done = False

        return self.board.copy(), reward, done, information

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        return self.board.copy()

    def render(self, mode='human', ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.clear()
        ax.imshow(self.board, cmap='coolwarm', vmin=-1, vmax=1)

        for x in range(3):
            for y in range(3):
                symbol = 'X' if self.board[y, x] == 1 else 'O' if self.board[y, x] == -1 else ''
                ax.text(x, y, symbol, fontsize=30, ha='center', va='center')

        ax.set_xticks([])
        ax.set_yticks([])

    def check_win(self, player):
        for row in range(3):
            if np.all(self.board[row, :] == player) or np.all(self.board[:, row] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def is_full(self):
        return np.all(self.board != 0)

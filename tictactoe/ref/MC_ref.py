import numpy as np

class MonteCarloAgent:
    def __init__(self, num_episodes=5000, discount_factor=1.0):
        self.q_values = np.zeros((3, 3, 3, 9))
        self.returns = {(i, j, k, a): [] for i in range(3) for j in range(3) for k in range(3) for a in range(9)}
        self.discount_factor = discount_factor
        self.num_episodes = num_episodes

    def evaluate_policy(self, env):
        for episode in range(self.num_episodes):
            episode_history = []
            state = env.reset()

            while True:
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                episode_history.append((state, action, reward))
                state = next_state

                if done:
                    break

            G = 0
            for t in reversed(range(len(episode_history))):
                state, action, reward = episode_history[t]
                G = self.discount_factor * G + reward

                s_idx = tuple(np.where(state == 1, 1, np.where(state == -1, 0, 2)).flatten())

                if (state, action) not in [(x[0], x[1]) for x in episode_history[:t]]:
                    self.returns[s_idx + (action,)].append(G)
                    self.q_values[s_idx + (action,)] = np.mean(self.returns[s_idx + (action,)])

    def best_first_move(self):
        initial_state = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2])
        q_values_initial_state = self.q_values[tuple(initial_state)].reshape(3, 3)
        best_move = np.unravel_index(np.argmax(q_values_initial_state), q_values_initial_state.shape)
        return best_move

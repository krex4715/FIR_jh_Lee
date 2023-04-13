import numpy as np

def td_zero(env, policy, episodes, alpha=0.1, gamma=1):
    action_value = np.zeros((3, 3, 3, 9))
    mse_per_episode = []
    for _ in range(episodes):
        observation = env.reset()
        done = False

        while not done:
            action = policy(env)
            next_observation, reward, done, _ = env.step(action)

            r, c = np.unravel_index(action, (3, 3))
            empty_cell = np.where(observation[:, :, 2] == 1)
            if not done:
                next_action = policy(env)
                next_r, next_c = np.unravel_index(next_action, (3, 3))
                td_error = reward + gamma * action_value[next_r, next_c, empty_cell, next_action] - action_value[r, c, empty_cell, action]
            else:
                td_error = reward - action_value[r, c, empty_cell, action]

            action_value[r, c, empty_cell, action] += alpha * td_error

            observation = next_observation
        
    return action_value


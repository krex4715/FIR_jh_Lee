import numpy as np
import matplotlib.pyplot as plt

def value_iteration(P_h, theta=1e-30, gamma=1.0):
    # define the value function: 1~99 state, 0 and 100 Terminal Dummy state
    V = np.zeros(101)
    V[100] = 1

    while True:
        delta = 0
        for s in range(1, 100):
            v = V[s]
            
            # define the action
            actions = range(1, min(s, 100 - s) + 1)
            action_values = []
            for a in actions:
                # next state is determined by the action, gambler's capital+action
                s_win = s + a
                s_lose = s - a
                # actionvalue update
                action_value = P_h * (gamma * V[s_win]) + (1 - P_h) * (gamma * V[s_lose])
                action_values.append(action_value)
            V[s] = np.max(action_values)
            delta = max(delta, abs(V[s] - v))
        
        if delta < theta:
            break
    
    policy = np.zeros(100)
    for s in range(1, 100):
        actions = range(1, min(s, 100 - s) + 1)
        action_values = []
        for a in actions:
            s_win = s + a
            s_lose = s - a
            action_value = P_h * (gamma * V[s_win]) + (1 - P_h) * (gamma * V[s_lose])
            action_values.append(action_value)
        policy[s] = actions[np.argmax(action_values)]
        
    return V[:-1], policy

def plot_results(P_h):
    V, policy = value_iteration(P_h)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].plot(V)
    ax[0].set_xlabel('Capital')
    ax[0].set_ylabel('Value estimates')
    ax[0].set_title(f'Value estimates for P_h = {P_h}')

    ax[1].plot(policy)
    ax[1].set_xlabel('Capital')
    ax[1].set_ylabel('Final policy (stake)')
    ax[1].set_title(f'Final policy for P_h = {P_h}')

    plt.show()

plot_results(0.4)  # 예시로 P_h 값을 0.4로 설정합니다.

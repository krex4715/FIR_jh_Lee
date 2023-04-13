import numpy as np
import matplotlib.pyplot as plt

def optimal_value_iteration(P_h, theta=1e-10, gamma=1.0):
    # define the value function: 1~99 state, 0 and 100 Terminal Dummy state
    V = np.zeros(101)
    V[100] = 1

    while True:
        delta = 0
        for s in range(1, 100):
            v = V[s]
            
            # define the action
            actions = range(1, min(s, 100 - s) + 1)
            
            # bellman optimality equation
            state_values = []
            for a in actions:
                # next state is determined by the action, gambler's capital+action
                s_win = s + a
                s_lose = s - a
                # statevalue update
                state_value = P_h * (gamma * V[s_win]) + (1 - P_h) * (gamma * V[s_lose])
                state_values.append(state_value)
            V[s] = np.max(state_values)
            delta = max(delta, abs(V[s] - v))

        if delta < theta:
            break
    
    policy = np.zeros(100)
    for s in range(1, 100):
        actions = range(1, min(s, 100 - s) + 1)
        state_values = []
        for a in actions:
            s_win = s + a
            s_lose = s - a
            state_value = P_h * (gamma * V[s_win]) + (1 - P_h) * (gamma * V[s_lose])
            state_values.append(state_value)
        policy[s] = actions[np.argmax(state_values)]
        
    return V[:-1], policy





def expected_value_iteration(P_h, theta=1e-30, gamma=1.0):
    V = np.zeros(101)
    V[100] = 1
    policy = np.ones((100, 101)) * (1 / 101)  # Initialize policy with a uniform distribution.

    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(1, 100): # s = 0, 100 is dummy state
                v = V[s]
                actions = range(1, min(s, 100 - s) + 1)
                state_values = []
                for a in actions:
                    s_win = s + a
                    s_lose = s - a
                    state_value = P_h * (gamma * V[s_win]) + (1 - P_h) * (gamma * V[s_lose])
                    state_values.append(state_value)
                V[s] = np.sum(np.array(state_values) * policy[s][1 : len(actions) + 1])
                delta = max(delta, abs(V[s] - v))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in range(1, 100):
            old_action = policy[s].copy()
            actions = range(1, min(s, 100 - s) + 1)
            state_values = []
            for a in actions:
                s_win = s + a
                s_lose = s - a
                state_value = P_h * (gamma * V[s_win]) + (1 - P_h) * (gamma * V[s_lose])
                state_values.append(state_value)

            max_index = np.argmax(state_values)
            policy[s] = np.zeros(101)
            policy[s][max_index + 1] = 1

            if not np.array_equal(old_action, policy[s]):
                policy_stable = False

        if policy_stable:
            break

    return V[:-1], policy




def optimal_plot_results(P_h, theta=1e-30):
    V, policy = optimal_value_iteration(P_h, theta=theta)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].plot(V)
    ax[0].set_xlabel('Capital')
    ax[0].set_ylabel('State values')
    ax[0].set_title(f'State values for P_h = {P_h}')

    ax[1].bar(range(100),policy)
    ax[1].set_xlabel('Capital')
    ax[1].set_ylabel('Final policy (stake)')
    ax[1].set_title(f'Final policy for P_h = {P_h}')

    plt.show()





def plot_expected_results(P_h,theta=1e-30):
    V, policy = expected_value_iteration(P_h,theta=theta)
    final_policy = np.argmax(policy, axis=1)
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].plot(V)
    ax[0].set_ylabel('State values')
    ax[0].set_title(f'State values for P_h = {P_h}')

    ax[1].bar(range(1,100), final_policy[1:])
    ax[1].set_xlabel('Capital')
    ax[1].set_ylabel('Final policy (stake)')
    ax[1].set_title(f'Final policy for P_h = {P_h}')

    plt.show()








optimal_plot_results(0.25,theta=1e-3)  # Example

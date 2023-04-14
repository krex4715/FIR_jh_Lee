# Solving Tic Tac Toe Game


I have implemented Tic Tac Toe Game with MC and TD Policy Evaluation.


### Monte Carlo Policy Evaluation
By using Monte Carlo Policy Evaluation, I can get the optimal First Action of the game.


### Temporal Difference Policy Evaluation
By using Temporal Difference Policy Evaluation, I can compare the convergence speed of the two methods.



### Symmetric States of Tic Tac Toe

Rotating and mirroring the board will result in the same game state. Therefore, we can reduce the number of states by grouping the symmetric states together. The following function returns all the symmetric states of a given state.

```python
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
```




## How to run

### Convergence Speed Comparison of TD and MC
```bash
python main.py
``` 


### Monte Carlo Policy Evaluation
```bash
python main_MC.py
```

### Temporal Difference Policy Evaluation
```bash
python main_TD.py
```


***


![img1](./img/TicTacToe.png)
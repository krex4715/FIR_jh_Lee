import main_MC,main_TD
import matplotlib.pyplot as plt
from env import TicTacToeEnv
import numpy as np
import matplotlib.colors as mcolors




if __name__=="__main__":
    env = TicTacToeEnv()
    print('--------Searching Best Init Action By using MC Evaluation--------')
    mc_convergence,mc_init_q = main_MC.main_MC(env, total_eps = 50000, rendering=False)
    print('--------MC,TD_Covnergence Comparison-----------------------------')
    td_convergence = main_TD.main_TD(env, alpha = 0.2, total_eps = 100, rendering=False)
    print('plotting convergence: for Comparison, only plot the first 100 episodes--')
    print('TD is faster than MC, but MC is more accurate than TD')
    
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
    ax2.set_title('Convergence Speed Comparison MC,TD')
    ax2.plot(mc_convergence[:100] ,label="MC")
    ax2.plot(td_convergence[:100] ,label="TD")
    ax2.legend()
    plt.show()
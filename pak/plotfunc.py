import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm

plt.rcParams['font.sans-serif']=['Microsoft YaHei']

def plot_SimpleRegret(result_list,n_calls,true_minimum=None, yscale=None, title="Simple regret plot"): 

    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel(r"Simple regret")
    ax.grid()
    if yscale is not None:
        ax.set_yscale(yscale)
    colors = cm.hsv(np.linspace(0.25, 1.0, len(result_list)))

    for results, color in zip(result_list, colors):
        name, results = results
        iterations = range(1, n_calls + 1)
        regrets = [[np.min(r[:i])-true_minimum for i in iterations]
                for r in results]
        mu = np.mean(regrets, axis=0)
        #std = np.std(regrets, axis=0)
        plt.plot(iterations, mu, c=color, lw=1, label=name)
        #plt.fill_between(iterations, mu + std, mu - std, alpha=0.2, color='#9FAEB2', lw=0)
    ax.legend(loc="best")
    plt.savefig(title+'_sim.png')
    return ax

def plot_InstantRegret(result_list,n_calls,true_minimum=None, yscale=None, title="Instant regret plot"): 

    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel(r"Instant regret")
    ax.grid()
    if yscale is not None:
        ax.set_yscale(yscale)
    colors = cm.hsv(np.linspace(0.25, 1.0, len(result_list)))

    minindex = []

    for results, color in zip(result_list, colors):
        name, results = results
        iterations = range(0, n_calls)
        regrets = [[r[i]-true_minimum for i in iterations]
                for r in results]
        mu = np.mean(regrets, axis=0)
        plt.plot(iterations, mu, c=color, lw=1, label=name)
        minindex.append((np.argmin(mu),np.min(mu)+true_minimum))
    ax.legend(loc="best")
    plt.savefig(title+'_ins.png')
    return minindex
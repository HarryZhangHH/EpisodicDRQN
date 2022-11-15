from matplotlib import pyplot as plt 
from matplotlib import ticker
import numpy as np
import matplotlib

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
plt.style.use('seaborn')

def relu(x):
    return np.maximum(0.0, x)

def single_line_plot(x, y, label=None, title=None, xlabel=None, ylabel=None, fname=None):
    fig, ax = plt.subplots()
    ax.plot(x, y, label=label)
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel)
    plt.title(title, fontweight="bold")
    if label is not None:
        ax.legend(frameon=True)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax.yaxis.set_major_formatter(formatter) 
    plt.tight_layout()
    fig.savefig(f'../code/images/market/{fname}.png', dpi=fig.dpi)
    plt.show()

def multiple_lines_plot(x, y, labels, title=None, xlabel=None, ylabel=None, fname=None):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.title(title, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.yaxis.get_offset_text()
    ax.legend(labels=labels, frameon=True)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax.yaxis.set_major_formatter(formatter) 
    plt.tight_layout()
    fig.savefig(f'../code/images/market/{fname}.png', dpi=fig.dpi)
    plt.show() 

def multi_axs_line_plot(N, x, y, title, xlabel, ylabel, subtitle=None, fname=None):
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1))
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.suptitle(title, fontweight="bold")
    if N%2 == 0:
        fig, axs = plt.subplots(N/2,2)
        for n in range(N):
            axs[int(n/2), n%2].plot(x[n], y[n]) 
            axs[int(n/2), n%2].set_title(subtitle[n])
            axs[int(n/2), n%2].set_xlabel(xlabel[n])
            axs[int(n/2), n%2].set_ylabel(ylabel[n])
            axs[int(n/2), n%2].yaxis.set_major_formatter(formatter) 
    else:
        fig, axs = plt.subplots(1,N)
        for n in range(N):
            axs[n].plot(x[n], y[n]) 
            axs[n].set_xlabel(xlabel[n])
            axs[n].set_ylabel(ylabel[n])
            axs[n].yaxis.set_major_formatter(formatter) 
    fig.savefig(f'../images/market/{fname}.png', dpi=fig.dpi)
    plt.show() 


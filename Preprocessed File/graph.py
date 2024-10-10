from termcolor import colored
import os
import io
import numpy as np
import matplotlib.pylab as plt
from numpy import array
import scipy
import scipy.signal
from PIL import Image
from matplotlib.pyplot import plot, scatter, show
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
plt.style.use('classic')

import statsmodels.api as sm


def bland_altman_plot_paper(data1, data2, name = " ", *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    print(md + 1.96*sd, md - 1.96*sd)
    
    fig, ax = plt.subplots(figsize=(5,3.5))
    # fig, ax = plt.subplots(figsize=(5,3))
    fig.set_facecolor("white")
    # plt.title('Bland-Altman Plot')
#    plt.legend()
    plt.scatter(mean, diff, *args, **kwargs)

    plt.axhline(md + 1.96*sd, color='g', label="md + 1.96*sd", linestyle='--')
    plt.axhline(md,           color='r', label="md",           linestyle='--')
    plt.axhline(md - 1.96*sd, color='b', label="md - 1.96*sd", linestyle='--')
    labels = ["md + 1.96*sd", "md", "md - 1.96*sd"]
    handles, _ = ax.get_legend_handles_labels()
    # Slice list to remove first handle
    plt.legend(handles = handles[:], labels = labels, fontsize = 10)
    plt.xlabel("Average Hemoglobin(g/dL)",fontsize=10)
    plt.ylabel("Difference Hemoglobin(g/dL)",fontsize=10)
    plt.savefig("H:/Glucose/Preprocessed File/Imgs/"+ str(name)+".pdf", bbox_inches='tight', dpi=320)
    plt.show()
  
def act_pred_plot_paper(y, predicted, R_2=None, mae=None, name = ""):
    fig, ax = plt.subplots(figsize=(5,3.5))
    fig.set_facecolor("white")
    ax.text(y.min()-1.5, y.max()-1, '$MAE =$ %0.3f' %(np.mean(mae)))
    ax.text(y.min()-1.5, y.max()+.5, '$R^2 =$ %.3f' %(np.mean(R_2)))
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r-', lw=1)
    ax.set_xlabel('Reference Hemoglobin(g/dL)')
    ax.set_ylabel('Estimated Hemoglobin(g/dL)')
    plt.savefig("H:/Glucose/Preprocessed File/Imgs/"+ str(name)+".pdf", bbox_inches='tight', dpi=320)
    plt.show() 
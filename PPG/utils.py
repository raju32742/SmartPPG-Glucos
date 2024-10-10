

#----------------------------------------------------------------
# imports
#----------------------------------------------------------------
from termcolor import colored
import os
import io
import numpy as np
import matplotlib.pylab as plt
from numpy import array
import scipy
import scipy.signal
from .config import config
from PIL import Image
from matplotlib.pyplot import plot, scatter, show
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path
#-------------------------------------------------------------------
def max_first_three(lst):
    '''
        * Use to find first three PPG wave
        arg:
            lst = list 
    '''
    ranks = sorted( [(x,i) for (i,x) in enumerate(lst)], reverse=True )
    values = []
    posns = []
    for x,i in ranks:
        if x not in values:
            values.append( x )
            posns.append( i )
            if len(values) == 3:
                break
            
    return values, posns

#-------------------------------------------------------------------
def sort_descending_order(lst):
    '''
        * Use to sort descending order of PPG wave
        arg:
            lst = list 
    '''
    ranks = sorted( [(x,i) for (i,x) in enumerate(lst)], reverse=True )
    values = []
    posns = []
    for x,i in ranks:
        if x not in values:
            values.append( x )
            posns.append( i )
            
    return values, posns
#------------------------------------------------------------------
def avgNestedLists(nested_vals):
    """
    Averages a 2-D array and returns a 1-D array of all of the columns
    averaged together, regardless of their dimensions.
    """
    output = []
    maximum = 0
    for lst in nested_vals:
        if len(lst) > maximum:
            maximum = len(lst)
    for index in range(maximum): # Go through each index of longest list
        temp = []
        for lst in nested_vals: # Go through each list
            if index < len(lst): # If not an index error
                temp.append(lst[index])
        output.append(np.nanmean(temp))
    return output
#-------------------------------------------------------------------
def tolerant_mean(arrs):
    """
    Averages a 2-D array and returns a 1-D array of all of the columns
    averaged together and error, regardless of their dimensions.
    """
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

#-------------------------------------------------------------------

#-------------------------------------------------------------------
def plot_time_series(sinagl, out_img_dir, color = "r", label="name", fig_save=False, verbose=1):
    '''
        plot series of input signal
        args:
            singal  =   1-D list of float value
    '''
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(figsize=(10,6))
    plt.plot(sinagl, color, linewidth=1, label=label)
    
    plt.xlim(0, len(sinagl))
    plt.legend()
    plt.tight_layout()
    if fig_save:
        _path=create_dir(out_img_dir, "ppg_imgs")
        plt.savefig(os.path.join(_path, label+".png"), dpi = 100)
    if verbose:
        plt.show()

#-------------------------------------------------------------------
def plot_certain_ppg(sinagl, xmin, xmax, out_img_dir, color = "r", label="name", fig_save=False, verbose=1):
    '''
        plot series of input signal
        args:
            singal  =   1-D list of float value
    '''
    xi = list(range(xmin, xmax))
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(figsize=(8,6))
    plt.plot(xi, sinagl,  color, linewidth=1, label=label) 
    plt.xlabel('Index')
    plt.ylabel('Magnitude') 
    plt.title('PPG Signal')
    plt.grid(False)
    plt.legend() 
    plt.tight_layout()
    if fig_save:
        _path=create_dir(out_img_dir, "ppg_imgs")
        plt.savefig(os.path.join(_path, label+".png"), dpi = 100)
    if verbose:
        plt.show()

#-------------------------------------------------------------------
def plot_final_ppg(sinagl, color = "r", label="name"):
    '''
        plot series of input signal
        args:
            singal  =   1-D list of float value
    '''
    plt.rcParams.update({'figure.max_open_warning': 0})
    fig = Figure(figsize=(8, 6), dpi=100)
    plt.plot(sinagl, color, linewidth=1, label=label)
    plt.xlim(0, len(sinagl))
    plt.legend()
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    return im

#-------------------------------------------------------------------
def plot_sample_series(sinagl, color = "r", label="name", verbose=1):
    '''
        plot series of input signal
        args:
            singal  =   1-D list of float value
    '''
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(figsize=(8,6))
    plt.plot(sinagl, color, linewidth=1, label=label)
    
    plt.xlim(0, len(sinagl))
    plt.legend()
    plt.tight_layout()
    if verbose:
        plt.show()
#-------------------------------------------------------------------
def refilter(_sig):
    '''
    * source: https://stackoverflow.com/a/51997184/5424617

        Finally, if you want to better see the contribution of 
        some specific frequency components without the interference 
        from spectral leakage from other frequency component, 
        you may want to consider pre-filtering your time-domain 
        signal before computing the FFT. For example, if you want to 
        eliminate the effect of the constant signal bias, the slow ~0.1Hz 
        variation and the noises with frequency greater than 10Hz 
        you might use something like the following:
    '''
    b,a = scipy.signal.butter(config.ORDER[1], [0.25/10, 5/10], btype='bandpass')
    y = scipy.signal.filtfilt(b,a,scipy.signal.detrend(np.ravel(_sig), type='constant'), padlen=len(_sig)-1)
    return y
#-------------------------------------------------------------------
def FFT(_signal):
    '''
    * Apply FFT on signal
    '''
    fast_FFT = scipy.fft.fft(_signal)
    return fast_FFT.real

#--------------------------------------------------------------------
def plot_peak_detect_series(series, maxtab, mintab, out_img_dir, label="name", fig_save=False, verbose=1):
    '''
        plot peak of series (ppg)
        args:
            series              =   1-D list of float value
            [maxtab, mintab]    =   [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
                                    maxima and minima ("peaks") in the vector V.
    
    '''
    pos_maxtab = array(maxtab)[:,0]
    val_maxtab = array(maxtab)[:,1]
    pos_mintab = array(mintab)[:,0]
    val_mintab = array(mintab)[:,1]

    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.style.use('ggplot') # nicer plots
    np.random.seed(52102) 
    plt.figure(figsize=(8, 6))
    plot(series)

    scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
    scatter(array(mintab)[:,0], array(mintab)[:,1], color='green')
    for i, txt in enumerate(pos_maxtab):
        plt.annotate(txt, (pos_maxtab[i], val_maxtab[i]))
        
    for i, txt in enumerate(pos_mintab):
        plt.annotate(txt, (pos_mintab[i], val_mintab[i]))
    if fig_save:
        _path=create_dir(out_img_dir, "peak detected ppg_imgs")
        plt.savefig(os.path.join(_path, label+".png"), dpi = 100)
    if verbose:
        plt.show()
#----------------------------------------------------------------------

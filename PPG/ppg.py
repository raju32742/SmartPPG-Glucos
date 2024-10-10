
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array
import numpy as np
from scipy.signal import butter, filtfilt, sosfilt
from scipy.signal import argrelmax, argrelmin
from .config import config
from .utils import LOG_INFO, max_first_three, sort_descending_order, avgNestedLists

class PPG:
    def __init__(self, data):
        '''
            args:
                data : avg. value of red channel value (1-D list).
        '''
        self.data = data
        # error check -- type
        assert type(self.data)==list,"data(r_mean channel) is not a list"

    def bandPass(self):
        '''
            * apply band-pass filter for generating PPG signal

            returns:
                band-pass filter: it's a like of reverse-ppg singal.

        '''
        b, a = butter(config.ORDER[2], [(config.BPM_L / 60) * (2 / config.FPS), (config.BPM_H / 60) * (2 / config.FPS)], btype='band')
        print("Order", config.ORDER[2])
        filtered_data = filtfilt(b, a, self.data) # This is a reverse-filtered PPG.
        return filtered_data

    def bandPassSos(self):
        '''
            * apply band-pass (sosfilt) filter for generating PPG signal

            returns:
                band-pass filter: it's a like of reverse-ppg singal.

        '''
        sos = butter(config.ORDER[2], [(config.BPM_L / 60) * (2 / config.FPS), (config.BPM_H / 60) * (2 / config.FPS)], analog=False, btype='band', output='sos')
        filtered_data = sosfilt(sos, self.data) # This is a reverse-filtered PPG.
        return filtered_data
    
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

    def butter_bandpass_filter(self):
        lowcut = (config.BPM_L / 60) * (2 / config.FPS)
        highcut = (config.BPM_H / 60) * (2 / config.FPS)
        fs = config.FPS
        order = config._ORDER
        sos = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, self.data)
        return y
    
    def normPass(self):
        '''
            * normal calculation for PPG
        '''
        return 0

    def peakdet(self, v, delta, x = None):
        """
        Converted from MATLAB script at http://billauer.co.il/peakdet.html
        
        Returns two arrays
        
        function [maxtab, mintab]=peakdet(v, delta, x)
        %PEAKDET Detect peaks in a vector
        %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
        %        maxima and minima ("peaks") in the vector V.
        %        MAXTAB and MINTAB consists of two columns. Column 1
        %        contains indices in V, and column 2 the found values.
        %      
        %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
        %        in MAXTAB and MINTAB are replaced with the corresponding
        %        X-values.
        %
        %        A point is considered a maximum peak if it has the maximal
        %        value, and was preceded (to the left) by a value lower by
        %        DELTA.
        
        % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
        % This function is released to the public domain; Any use is allowed.
        
        """
        maxtab = []
        mintab = []
        
        if x is None:
            x = arange(len(v))
        
        v = asarray(v)
        
        if len(v) != len(x):
            sys.exit('Input vectors v and x must have same length')
        
        if not isscalar(delta):
            sys.exit('Input argument delta must be a scalar')
        
        if delta <= 0:
            sys.exit('Input argument delta must be positive')
        
        mn, mx = Inf, -Inf
        mnpos, mxpos = NaN, NaN
        
        lookformax = True
        
        for i in arange(len(v)):
            this = v[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                mnpos = x[i]
            
            if lookformax:
                if this < mx-delta:
                    maxtab.append((mxpos, mx))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn+delta:
                    if mn < 0.0: # <-- change condition
                        mintab.append((mnpos, mn))
                        mx = this
                        mxpos = x[i]
                        lookformax = True
        return array(maxtab), array(mintab)

    def peak3ppgwave_avg3ppgwave(self, series, maxtab, mintab):
        ''' 
            * Pick Fresh 3 PPG wave 
            args:
                series              =   1-D list of float value of PPG waves
                [maxtab, mintab]    =   [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
                                        maxima and minima ("peaks") in the vector V.
            return:
                fresh_peak_3_ppg    =   values of peak 3 PPG waves. <LIST>
                avg
        '''
        pos_maxtab = array(maxtab)[:,0]
        val_maxtab = array(maxtab)[:,1]
        pos_mintab = array(mintab)[:,0]
        val_mintab = array(mintab)[:,1]
        values, posns = max_first_three(val_maxtab)
        # print(values, posns)

        values_peak_3_ppg = []
        for idx in posns:
            left_idx = idx - 1
            right_idx = idx #+ 1
            if left_idx < 0:
                left_idx = left_idx + 1
            elif right_idx >= len(pos_mintab):
                right_idx = right_idx - 1 
                
            # print(pos_mintab[left_idx], pos_mintab[right_idx])
            l_idx = int(pos_mintab[left_idx])
            r_idx = int(pos_mintab[right_idx]) + 1
            # print(series[l_idx:r_idx])
            # print("Len: ", len(series[l_idx:r_idx]))
            values_peak_3_ppg.append(list(series[l_idx:r_idx]))

        avg_peak_3_ppg = avgNestedLists(values_peak_3_ppg)
        fresh_peak_3_ppg = sum(values_peak_3_ppg, [])  
        return fresh_peak_3_ppg, avg_peak_3_ppg, l_idx, r_idx

    ## -----------------------------------------------------------------
    def _peakFinePPG(self, series, maxtab, mintab):
        ''' 
            * Pick Fresh 3 or 1 PPG wave 
            args:
                series              =   1-D list of float value of PPG waves
                [maxtab, mintab]    =   [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
                                        maxima and minima ("peaks") in the vector V.
            return:
                fresh_peak_ppg    =   values of peak PPG wave. <LIST>
                avg
        '''
        pos_maxtab = array(maxtab)[:,0]
        val_maxtab = array(maxtab)[:,1]
        pos_mintab = array(mintab)[:,0]
        val_mintab = array(mintab)[:,1]
        values, posns = sort_descending_order(val_maxtab)
        # print(values, posns)

        cnt_ppg = 0
        values_peak_ppg = []
        for idx in posns:
            left_idx = idx - 1
            right_idx = idx #+ 1
            if left_idx < 0:
                left_idx = left_idx + 1
            elif right_idx >= len(pos_mintab):
                right_idx = right_idx - 1 
                
            # print(pos_mintab[left_idx], pos_mintab[right_idx])
            l_idx = int(pos_mintab[left_idx])
            r_idx = int(pos_mintab[right_idx]) + 1
            # print(l_idx, r_idx)

            ## check Dicrotic Notch
            sing_ppg = list(series[l_idx:r_idx])
            # print(sing_ppg)
            maxima_index = argrelmax(np.array(sing_ppg))[0]
            minima_index = argrelmin(np.array(sing_ppg))[0]
            # print(maxima_index, minima_index)

            if len(maxima_index) >= 2 and len(minima_index) >= 1:
                values_peak_ppg.append(sing_ppg) # take single ppg
                cnt_ppg += 1
                # LOG_INFO(cnt_ppg) 
                # LOG_INFO(values_peak_ppg) 
            # else:
            #     '''
            #         *when we can't find best right signal
            #     '''
            #     fresh_peak_3_ppg, avg_peak_3_ppg = self.peak3ppgwave_avg3ppgwave(series, maxtab, mintab)
            #     return fresh_peak_3_ppg, avg_peak_3_ppg

            if cnt_ppg == 1: # 1 for one PPG wave & 3 for three PPG waves 
                break

        avg_peak_ppg = avgNestedLists(values_peak_ppg)
        fresh_peak_ppg = sum(values_peak_ppg, [])  
        return fresh_peak_ppg, avg_peak_ppg, l_idx, r_idx

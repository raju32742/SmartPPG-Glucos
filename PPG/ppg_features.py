
import numpy as np
from scipy.signal import argrelmax, argrelmin
from .config import config

class PPGFeatures:
 
    def extract_ppg45(single_waveform, sample_rate=config.PPG_SAMPLE_RATE):
        def __next_pow2(x):
            return 1<<(x-1).bit_length()

        features = []

        maxima_index = argrelmax(np.array(single_waveform))[0]
        minima_index = argrelmin(np.array(single_waveform))[0]

        if len(maxima_index) >= 2 and len(minima_index) >= 1:
            pass
        else:
            f_ind = np.argmax(np.array(single_waveform))
            # print(len(single_waveform[f_ind:]))
            s_ind = f_ind + 1
            if (len(single_waveform) > f_ind + 2) and (s_ind - f_ind >= 1):
                s_val = single_waveform[s_ind]
                t_val = single_waveform[s_ind+1]
                single_waveform[s_ind] = t_val
                single_waveform[s_ind+1] = s_val

            maxima_index = argrelmax(np.array(single_waveform))[0]
            minima_index = argrelmin(np.array(single_waveform))[0]

        # print(maxima_index, minima_index)

        derivative_1 = np.diff(single_waveform, n=1) * float(sample_rate)
        # print(derivative_1)
        derivative_1_maxima_index = argrelmax(np.array(derivative_1))[0]
        derivative_1_minima_index = argrelmin(np.array(derivative_1))[0]
        derivative_2 = np.diff(single_waveform, n=2) * float(sample_rate)
        derivative_2_maxima_index = argrelmax(np.array(derivative_2))[0]
        derivative_2_minima_index = argrelmin(np.array(derivative_2))[0]
        sp_mag = np.abs(np.fft.fft(single_waveform, n=__next_pow2(len(single_waveform))*16))
        freqs = np.fft.fftfreq(len(sp_mag))
        sp_mag_maxima_index = argrelmax(sp_mag)[0]
        # x
        x = single_waveform[maxima_index[0]]
        features.append(x)

        ##========================
        # c slope
        c = 0.55*x 
        features.append(c)
        
        # t_ms
        t_ms = 0.55*(float(maxima_index[0] + 1) / float(sample_rate))
        features.append(t_ms)

        # d = immediate prev. point of diastolic point
        d = single_waveform[maxima_index[1]-1]
        features.append(d)

        # t_ip
        t_ip = float(maxima_index[1]) / float(sample_rate)
        features.append(t_ip)
        ##========================
        
        # y
        y = single_waveform[maxima_index[1]]
        features.append(y)
        # z
        z = single_waveform[minima_index[0]]
        features.append(z)
        # t_pi
        t_pi = float(len(single_waveform)) / float(sample_rate)
        features.append(t_pi)
        # t_1
        t_1 = float(maxima_index[0] + 1) / float(sample_rate)
        features.append(t_1)
        # t_2
        t_2 = float(minima_index[0] + 1) / float(sample_rate)
        features.append(t_2)
        # t_3
        t_3 = float(maxima_index[1] + 1) / float(sample_rate)
        features.append(t_3)
        # width
        single_waveform_halfmax = max(single_waveform) / 2
        width = 0
        for value in single_waveform[maxima_index[0]::-1]:
            if value >= single_waveform_halfmax:
                width += 1
            else:
                break
        for value in single_waveform[maxima_index[0]+1:]:
            if value >= single_waveform_halfmax:
                width += 1
            else:
                break
        features.append(width)
        # A_2/A_1
        features.append(sum(single_waveform[:maxima_index[0]]) / sum(single_waveform[maxima_index[0]:]))
       
        #a1
        a_1 = derivative_1[derivative_1_maxima_index[0]]
        features.append(a_1)
        #b1
        b_1 = derivative_1[derivative_1_minima_index[0]]
        features.append(b_1)
        #e1
        e_1 = derivative_1[derivative_1_maxima_index[1]]
        features.append(e_1)
        #l1
        l_1 = derivative_1[derivative_1_minima_index[1]]
        features.append(l_1)
         
        # a_2
        a_2 = derivative_2[derivative_2_maxima_index[0]]
        features.append(a_2)
        # b_2
        b_2 = derivative_2[derivative_2_minima_index[0]]
        features.append(b_2)
        # e_2
        if len(derivative_2_maxima_index)>1:
            e_2 = derivative_2[derivative_2_maxima_index[1]]
            t_e2 = float(derivative_2_maxima_index[1]) / float(sample_rate)
        elif len(derivative_2_minima_index)>1:
            e_2 = derivative_2[derivative_2_minima_index[1]]
            t_e2 = float(derivative_2_minima_index[1]) / float(sample_rate)
        else:
            e_2 = derivative_2[derivative_2_maxima_index[0]]
            t_e2 = float(derivative_2_maxima_index[0]) / float(sample_rate)
        # print(derivative_2_maxima_index, derivative_2_minima_index)
        # e_2 = derivative_2[derivative_2_maxima_index[1]]
        features.append(e_2)
        
        # t_a1
        t_a1 = float(derivative_1_maxima_index[0]) / float(sample_rate)
        features.append(t_a1)
        # t_b1
        t_b1 = float(derivative_1_minima_index[0]) / float(sample_rate)
        features.append(t_b1)
        # t_e1
        t_e1 = float(derivative_1_maxima_index[1]) / float(sample_rate)
        features.append(t_e1)
        # t_f1
        t_f1 = float(derivative_1_minima_index[1]) / float(sample_rate)
        features.append(t_f1)
        
       
        
        # t_a2
        t_a2 = float(derivative_2_maxima_index[0]) / float(sample_rate)
        features.append(t_a2)
        # t_b2
        t_b2 = float(derivative_2_minima_index[0]) / float(sample_rate)
        features.append(t_b2)
        # t_e2
        features.append(t_e2)
        
        # f_base
        f_base = freqs[sp_mag_maxima_index[0]] * sample_rate
        features.append(f_base)
        # sp_mag_base
        sp_mag_base = sp_mag[sp_mag_maxima_index[0]] / len(single_waveform)
        features.append(sp_mag_base)
        # f_2
        f_2 = freqs[sp_mag_maxima_index[1]] * sample_rate
        features.append(f_2)
        # sp_mag_2
        sp_mag_2 = sp_mag[sp_mag_maxima_index[1]] / len(single_waveform)
        features.append(sp_mag_2)
        # f_3
        f_3 = freqs[sp_mag_maxima_index[2]] * sample_rate
        features.append(f_3)
        # sp_mag_3
        sp_mag_3 = sp_mag[sp_mag_maxima_index[2]] / len(single_waveform)
        features.append(sp_mag_3)
        return features, single_waveform

    def extract_svri(single_waveform):
        def __scale(data):
            data_max = max(data)
            data_min = min(data)
            return [(x - data_min) / (data_max - data_min) for x in data]
        max_index = np.argmax(single_waveform)
        single_waveform_scaled = __scale(single_waveform)
        return np.mean(single_waveform_scaled[max_index:]) / np.mean(single_waveform_scaled[:max_index])
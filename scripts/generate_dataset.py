
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import sys
sys.path.append('../')
import os
import argparse
import pandas as pd
import numpy as np
from numpy import array
from tqdm.auto import tqdm
from PPG.extract_frames import ExtractFrames
from PPG.utils import FFT, LOG_INFO, plot_time_series, plot_sample_series, plot_peak_detect_series, \
    refilter, plot_certain_ppg
from PPG.ppg_features import PPGFeatures
from PPG.ppg import PPG
from PPG.config import config

def main(data_dir, save_dir, out_img_dir, ppg_signal_dir, ppg_feats, xlsx_path, ppg_labels, frames_save, fps_num, frame_rate, default_fps, fig_save, verbose):
    ##---------------------------------
    # data_dir        =   args.data_dir
    # save_dir        =   args.save_dir
    # out_img_dir     =   args.out_img_dir
    # ppg_signal_dir  =   args.ppg_signal_dir
    # ppg_feats       =   args.ppg_feats
    # xlsx_path       =   args.xlsx_path
    # ppg_labels      =   args.ppg_feats_labels
    # frames_save     =   args.frames_save
    # fps_num         =   args.fps_num
    # frame_rate      =   args.frame_rate
    # default_fps     =   args.default_fps
    # fig_save        =   args.fig_save
    # verbose         =   args.verbose
    ##----------------------------------
    file_name_lst = []
    ppg_signals_lst = []
    features_set = []

    _gen_ppg = ExtractFrames(data_dir, save_dir)
    for root, dirs, files in tqdm(sorted(os.walk(data_dir))):
        for file in files:
            LOG_INFO(f"File name= {file}", mcolor="green")
            if default_fps == True:
                frames_fps = _gen_ppg.video_to_frames(file, frames_save, fps_num)
            else:
                frames_fps = _gen_ppg.video_to_frames_fixed_frameRate(file, frames_save, None, frame_rate)

            # to store avg. r, g, b value
            _signal=[]
            _cnt = 0
            for idx, (img,fps) in tqdm(enumerate(frames_fps)):
                try:
                    '''
                        fps = total_frames / video_duration
                    '''
                    # motive to take 600 frames 
                    # if fps <= 30:
                    #     fps = 2*fps
                        
                    # take almost 300/600 frames (10 sec video: 30 fps / 60 fps --> (10x30)=300, (10x60)=600)   
                    if idx >= 2*fps and _cnt <= 10*fps:
                        '''
                            * avoid first 2 sec. video
                            * take 10 sec. video
                        '''

                        ## to get a square right to left (ROI: 500x500)
                        h= img.shape[0]//2 - 250 # height//2 - 250. where height = 1080 px
                        w= img.shape[1] # width. where width = 1920 px
                        img = img[h:h+500, w-500:w]
                        # print("Image size: " + str(img.shape))

                        ## find max and min intensity of image's red channel
                        intensity_min = img[..., 2].min()
                        intensity_max = img[..., 2].max()
                        thresh = (0.5*intensity_min)+(0.5*intensity_max)
                        # print(intensity_min, intensity_max, thresh)

                        mean_pixel = img[:, :, 2].mean()
                        if mean_pixel > thresh:
                            _signal.append(mean_pixel)
    
                        _cnt += 1

                    elif _cnt > 10*fps:
                        LOG_INFO(f"Frame number= {_cnt}", mcolor="green")
                        break
                    
                except Exception as e:
                    LOG_INFO(f"File Number Error= {idx}",mcolor="red")
                    LOG_INFO(f"{e}",mcolor="red") 

            ## check the ppg whether found or not
            try:

                ## plot red signal
                plot_time_series(_signal, out_img_dir, 'r', str(file.split(".")[0])+'- RED Channel Signal', fig_save=fig_save, verbose=verbose)

                ## apply bandpass filter
                _ppg = PPG(_signal)
                rev_PPG_signal = _ppg.bandPass()
                _PPG_signal = rev_PPG_signal[::-1] # reverse bandpass signal (like. PPG)
                plot_time_series(_PPG_signal, out_img_dir, 'r', str(file.split(".")[0])+'- BandPass Signal', fig_save=fig_save, verbose=verbose)
                
                ## Save PPG singals w.r.t video
                file_name_lst.append(file)
                ppg_signals_lst.append(np.array(_PPG_signal))


                ## Peak detection of PPG signal
                series = _PPG_signal # rename or copy
                maxtab, mintab = _ppg.peakdet(series,config.DELTA)
                plot_peak_detect_series(series, maxtab, mintab, out_img_dir, str(file.split(".")[0])+'- Peak detect BandPass Signal', fig_save=fig_save, verbose=verbose)

                ## pick fresh 3 PPG wave and avg PPG wave
                best_systolic = False
                if best_systolic == True:
                    peak_3_ppg, avg_3_ppg, l_idx, r_idx = _ppg.peak3ppgwave_avg3ppgwave(series, maxtab, mintab)
                else:
                    '''
                        sorting decending order w.r.t. sysolic pick
                    '''
                    peak_ppg, avg_ppg, l_idx, r_idx = _ppg._peakFinePPG(series, maxtab, mintab)
                    '''
                        * if best ppg is not found then take most 3 ppg and do avg.
                    '''
                    if len(peak_ppg) == 0 or len(avg_ppg) == 0:
                        LOG_INFO("best ppg is not found!!!")
                        peak_ppg, avg_ppg, l_idx, r_idx = _ppg.peak3ppgwave_avg3ppgwave(series, maxtab, mintab)

                plot_certain_ppg(peak_ppg, l_idx, l_idx+len(peak_ppg), out_img_dir, 'r', str(file.split(".")[0])+"- peak PPG Wave", fig_save=fig_save, verbose=verbose)
                plot_certain_ppg(avg_ppg, l_idx, l_idx+len(avg_ppg), out_img_dir, 'r', str(file.split(".")[0])+"- avg PPG wave", fig_save=fig_save, verbose=verbose)

                ## Total 46 features: Extract PPG 45 features + svri feature _FFT
                '''
                    * with extracted features generate csv file
                '''
                try:
                    if best_systolic:
                        _feat_ppg49, _single_waveform = PPGFeatures.extract_ppg45(avg_3_ppg)
                        _feat_svri = PPGFeatures.extract_svri(_single_waveform)
                    else:
                        _feat_ppg49, _single_waveform = PPGFeatures.extract_ppg45(avg_ppg)
                        _feat_svri = PPGFeatures.extract_svri(_single_waveform)

                    LOG_INFO(f"49 fetures: {_feat_ppg49}||\t\n SVRI: {_feat_svri}",mcolor="green") 
                    plot_certain_ppg(_single_waveform, l_idx, l_idx+len(_single_waveform), out_img_dir, 'r', str(file.split(".")[0])+"- Final Single PPG", fig_save=fig_save, verbose=verbose)
                    
                    """ PPG-49 + extract_svri Feature Extraction """
                    file_name = file.split(".")[0]
                    ID = file_name.split("Hb")[-1]                    
                    _feat_ppg49.insert(0, int(ID))
                    _feat_ppg49.append(_feat_svri)
                    features_set.append(_feat_ppg49)

                except Exception as e:
                    LOG_INFO(f"{e}",mcolor="red") 

            except Exception as e:
                LOG_INFO(f"PPG Signal Error: {e}",mcolor="red") 

    # ## save ppg signal of each video to .csv file
    # list_dict = {'FileName': np.array(file_name_lst, dtype=object), 'PPG_Signal': np.array(ppg_signals_lst, dtype=object)} 
    # ppg_df = pd.DataFrame(list_dict) 
    # ppg_df.to_csv(ppg_signal_dir, index=False) 

    """ 
        * Preprocess the dataset for all videos
    """
    ## make ppg features .csv file
    headers = ['ID', 'Systolic_peak(x)', 'Max. Slope(c)', 'Time of Max. Slope(t_ms)', 'Prev. point a_ip(d)', 'Time of a_ip(t_ip)', 'Diastolic_peak(y)', 'Dicrotic_notch(z)', 'Pulse_interval(tpi)', 'Systolic_peak_time(t1)', 'Diastolic_peak_time(t2)', 'Dicrotic_notch_time(t3)', 'w', 'Inflection_point_area_ratio(A2/A1)', 'a1','b1', 'e1', 'l1', 'a2','b2','e2', 'ta1', 'tb1', 'te1', 'tl1', 'ta2', 'tb2', 'te2', 'Fundamental_component_frequency(fbase)', 'Fundamental_component_magnitude(|sbase|)', '2nd_harmonic_frequency(f2nd)', '2nd_harmonic_magnitude(|s2nd|)', '3rd_harmonic_frequency(f3rd)', '3rd_harmonic_magnitude(|s3rd|)', 'Stress-induced_vascular_response_index(sVRI)']
    print(len(headers))
    df_in = pd.DataFrame(features_set, columns=headers)
    df_in.to_csv(ppg_feats)


if __name__=="__main__":
    '''
        parsing and executions
    '''
    # parser = argparse.ArgumentParser("Raw Video Data Analysis, PPG Generation, and PPG Features Set Generation Script")
    # parser.add_argument("data_dir", help="Path to source data") 
    # parser.add_argument("save_dir", help="Path to save the processed data")
    # parser.add_argument("out_img_dir", help="Path to save the output images") 
    # parser.add_argument("ppg_signal_dir", help="Path to save the ppg signal as csv file")
    # parser.add_argument("ppg_feats", help="Path to save PPG features .csv file")
    # parser.add_argument("xlsx_path", help="Path of gold standard datset in .xlsx")
    # parser.add_argument("ppg_feats_labels", help="Path to save PPG features and labels .csv file")
    # parser.add_argument("--frames_save",type=bool,required=False,default=False,help ="Whether you will save images in save dir : default=False")
    # parser.add_argument("--fps_num",type=int,default=60,help ="Fixed fps: default=30")
    # parser.add_argument("--frame_rate",type=float,default=0.0167,help ="Fixed frame rate: default=0.035")
    # parser.add_argument("--default_fps",type=bool,required=False,default=True,help ="Whether you will use default fps: default=True")
    # parser.add_argument("--fig_save",type=bool,required=False,default=True,help ="Whether you will see save plotted figure: default=True")
    # parser.add_argument("--verbose",type=int,required=False,default=0,help ="Whether you will see message/figure in terminal: default=1")

    # args = parser.parse_args()
    # main(args)
    '''
    root_path="../dataset_folder/" 
    src_path="${root_path}raw_videos/"
    save_img_path="${root_path}raw_images/"
    output_imgs_path="../output_images/"
    ppg_signal_dir="${root_path}ppg_signals.csv"
    ppg_feats="${root_path}ppg_feats.csv"
    xlsx_path="${root_path}Data.xlsx"
    ppg_feats_labels="${root_path}ppg_feats_labels.csv"
    
    python generate_dataset.py $src_path $save_img_path $output_imgs_path $ppg_signal_dir $ppg_feats $xlsx_path $ppg_feats_labels
    #--------------------------------------------------------------------------------------------------------------
    echo succeded
    '''
    
    
    data_dir =  "../dataset_folder/raw_videos/"
    save_dir = "../dataset_folder/raw_images/"
    out_img_dir = "../output_images/"
    ppg_signal_dir = "../dataset_folder/ppg_signals.csv"
    ppg_feats = "../dataset_folder/ppg_feats.csv"
    xlsx_path = "../dataset_folder/Data.xlsx"
    ppg_feats_labels = "../dataset_folder/ppg_feats_labels.csv"
    frames_save = False
    fps_num = 60
    frame_rate = 0.0167
    default_fps = True
    fig_save = True
    verbose = 0
    main(data_dir, save_dir, out_img_dir, ppg_signal_dir, ppg_feats, xlsx_path, ppg_feats_labels, frames_save, fps_num, frame_rate, default_fps, fig_save, verbose)
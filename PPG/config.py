
# -*-coding: utf-8 -
'''
    @author: Md. Rezwanul Haque
'''
from random import randint
class config:
    #---------------------------------
    # fixed params: Band-pass Fileter
    #---------------------------------
    FPS             =  30  # camera captured video using 30 FPS
    '''
    A normal resting heart rate for adults ranges from 60 to 100 beats per minute. 
    Generally, a lower heart rate at rest implies more efficient heart function 
    and better cardiovascular fitness. 
    For example, a well-trained athlete might have a 
    normal resting heart rate closer to 40 beats per minute.
    '''
    BPM_L           =  70 #randint(60, 101) #70  # min Blood Pulse per min (60-100)
    '''
    The American Heart Association (AHA) states that the maximum heart rate 
    during exercise should be roughly equal to 220 bpm minus the age of the person.
    '''
    BPM_H           =  195 #randint(170, 201) #200 #(220-31) # Max Blood Pulse per min (220) 
    ORDER           =  [2,3,4,6,9] # 0 index --> band-pass; 1 index --> transform
    _ORDER          =  randint(1,9)

    #----------------------------------
    # peak detection of PPG signal
    #----------------------------------
    DELTA           =    0.1

    #----------------------------------
    # Rate of PPG signal
    #----------------------------------    
    PPG_SAMPLE_RATE =  200

    #----------------------------------
    # Frame Rate for Extracting Frame
    #---------------------------------- 
    FRAME_RATE     =    0.0167 

    #----------------------------------
    # Frame Rate for Extracting Frame
    #---------------------------------- 
    FRAME_NUM      =    60 
    
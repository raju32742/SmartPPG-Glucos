#-*- coding: utf-8 -*-
"""
@author: Md. Rezwanul Haque,
         Mahmuda Rumi
"""
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import os
import cv2
from .utils import create_dir, LOG_INFO

#----------------------------------------------------------------
# class: for generating PPG signal
#----------------------------------------------------------------
class ExtractFrames(object):
    def __init__(self, video_dir, img_dir):
        '''
            video_dir: directory of finger-trip videos
            img_dir  : directory of finger-trip images
        '''
        self.video_dir = video_dir
        self.img_dir   = img_dir

    def video_to_frames(self, video_filename, frames_save=False, fps_num=None):
        """
            Extract frames from video
        """
        video_file_path = os.path.join(self.video_dir, video_filename)
        if frames_save:
            frames_dir = create_dir(self.img_dir, 'frames_'+video_filename)

        cap = cv2.VideoCapture(video_file_path)
        if fps_num is not None:
            cap.set(cv2.CAP_PROP_FPS, int(fps_num))

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1 # frame numbers
        duration = video_length/fps  

        if cap.isOpened() and video_length > 0:
            count = 0
            success, image = cap.read()
            while success:
                try:
                    fps = round(cap.get(cv2.CAP_PROP_FPS))
                    success, image = cap.read()
                    if frames_save:
                        cv2.imwrite(os.path.join(frames_dir, str(count) + '.jpg'), image)
                    yield image, fps
                    count += 1
                except Exception as e:
                    LOG_INFO(f"Error in frame Check:{count}",mcolor="yellow")
                    LOG_INFO(f"{e}",mcolor="red") 

    def video_to_frames_fixed_frameRate(self, video_filename, frames_save=False, fps_num=None, frame_rate = 0.035):
        """
            Extract frames from video
        """
        video_file_path = os.path.join(self.video_dir, video_filename)
        if frames_save:
            frames_dir = create_dir(self.img_dir, 'frames_'+video_filename)

        cap = cv2.VideoCapture(video_file_path)
        if fps_num is not None:
            cap.set(cv2.CAP_PROP_FPS, int(fps_num))

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1 # frame numbers
        duration = video_length/fps  

        if cap.isOpened() and video_length > 0:
            count = 0
            # success, image = cap.read()
            '''
            * CAP_PROP_POS_MSEC 
                Python: cv.CAP_PROP_POS_MSEC
                Current position of the video file in milliseconds.
            '''
            sec = 0
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            success, image = cap.read()

            while success:
                try:
                    sec = sec + frame_rate
                    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                    fps = round(cap.get(cv2.CAP_PROP_FPS))
                    success, image = cap.read()
                    if frames_save:
                        cv2.imwrite(os.path.join(frames_dir, str(count) + '.jpg'), image)
                    yield image, fps
                    count += 1
                except Exception as e:
                    LOG_INFO(f"Error in frame Check:{count}",mcolor="yellow")
                    LOG_INFO(f"{e}",mcolor="red")

from __future__ import absolute_import, division, print_function
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import segmentation, color
from skimage.future import graph
from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os 

def SGBM_SLIC(left,right):
    block_size = 11
    min_disp = 16
    max_disp = 64
    num_disp = max_disp - min_disp
    # disparity settings
    uniquenessRatio = 5
    speckleWindowSize = 200
    speckleRange = 2
    disp12MaxDiff = 0
    
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,numDisparities=num_disp,blockSize=block_size,uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize, speckleRange=speckleRange, disp12MaxDiff=disp12MaxDiff, P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size)
    disparity_SGBM = stereo.compute(left,right)
    disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255, beta=5, norm_type=cv2.NORM_MINMAX)


    p=1
    q=3
    f,b = (0.12,528.675)
    zf=4
    zn=1
    disp_f=f*b/zf
    disp_n=f*b/zn
    disp_max=255
    disp_min=5
    M=p+(q-p)*(disparity_SGBM-disp_min)/(disp_max-disp_min) #scale the disparity into [p,q]
    # now I know that when disp ref=disp_f I have Mf= 1.129
    Mf=p+(q-p)*(disp_f-disp_min)/(disp_max-disp_min)
    # now I know that when disp ref=disp_n I have Mf= 1.8 , it should be Mf=3 for disp_ref<=disp_n
    Mn=p+(q-p)*(disp_n-disp_min)/(disp_max-disp_min) 
    M[M>=Mn]=q
    M[M<=Mf]=p
 
    numSegments=300
    segments = slic(left, n_segments = numSegments,compactness=15, start_label=1)
    g = graph.rag_mean_color(left, segments)
    labels2 = graph.cut_threshold(segments, g, 18)                                                       
    out = color.label2rgb(labels2, M, kind='avg')
    finalM=out
    #print(out.shape)
    
    return finalM

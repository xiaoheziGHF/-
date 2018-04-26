#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:24:28 2018

@author: huafenguo
"""
################################
 #######ORB feature extractor #####
import numpy as np
import cv2
from matplotlib import pyplot as plt
 
img = cv2.imread('///Users/huafenguo/Desktop/13.jpg',0)  # 需注意格式
 
 # Initiate STAR detector
orb = cv2.ORB_create()
 
 # find the keypoints with ORB
kp = orb.detect(img,None)
 
 # compute the descriptors with ORB
kp, des = orb.compute(img, kp)
 
 #draw only keypoints location,not size and orientation
img = cv2.drawKeypoints(img,kp,None,color=(0,255,0), flags=0)
plt.imshow(img),plt.show()
# 
#=============================================================================
 ###### GLCMs##########
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from PIL import Image
from numpy import asarray
    
    
PATCH_SIZE = 21
    
# open the image, array 
temp=asarray(Image.open('///Users/huafenguo/Desktop/13.jpg'))
x=temp.shape[0]
y=temp.shape[1]*temp.shape[2]
    
temp.resize((x,y))
    
    
    
# select some patches from grassy areas of the image
face_locations = [(300, 550), (100, 430), (190, 930), (150, 600)]    
face_patches = []
for loc in face_locations:
    face_patches.append(temp[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])
    
# select some patches from sky areas of the image
    body_locations = [(10, 10), (650, 10), (10, 1457), (650, 1457)]
    body_patches = []
    for loc in body_locations:
        body_patches.append(temp[loc[0]:loc[0] + PATCH_SIZE,
                                 loc[1]:loc[1] + PATCH_SIZE])
    
# compute some GLCM properties each patch
    xs = []
    ys = []
    for patch in (face_patches + body_patches):
        glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
        xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(greycoprops(glcm, 'correlation')[0, 0])
    
# create the figure
fig = plt.figure(figsize=(8, 8))
    
# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(temp, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
for (y, x) in face_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in body_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')
    
# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(face_patches)], ys[:len(face_patches)], 'go',
           label='face')
ax.plot(xs[len(face_patches):], ys[len(face_patches):], 'bo',
            label='body')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()
    
# display the image patches
for i, patch in enumerate(face_patches):
        ax = fig.add_subplot(3, len(face_patches), len(face_patches)*1 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
                  vmin=0, vmax=255)
        ax.set_xlabel('face %d' % (i + 1))

for i, patch in enumerate(body_patches):
        ax = fig.add_subplot(3, len(body_patches), len(body_patches)*2 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
                  vmin=0, vmax=255)
        ax.set_xlabel('body %d' % (i + 1))
    
    
# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
plt.show()
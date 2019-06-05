import json
import cv2
import os
import numpy as np


def orb(gray):
    # Set the parameters of the ORB algorithm by specifying the maximum number of keypoints to locate and
    # the pyramid decimation ratio
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.5, edgeThreshold = 35, WTA_K = 4, patchSize = 50)    # Find the keypoints in the gray scale training and query images and compute their ORB descriptor.
    # The None parameter is needed to indicate that we are not using a mask in either case.  
    kpt,desp = orb.detectAndCompute(gray, None)

    return kpt, desp


def feature_matcher(descriptors_train, descriptors_query):
    matches=0
    # Create a Brute Force Matcher object. "crossCheck"  is set to True so that the BFMatcher will only return consistent
    # pairs. Such technique usually produces best results with minimal number of outliers when there are enough matches.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Perform the matching between the ORB descriptors of the training image and the query image
    matches = bf.match(descriptors_train, descriptors_query)

    # The matches with shorter distance are required. So, the matches are sort according to distance
    matches = sorted(matches, key = lambda x : x.distance)

    return matches

def remove_outiners(matches, keypoints_train, keypoints_query, size):

    dst = np.empty( shape=(0, 0) )
    #minimum matches for sustainability of a object recognition
    MIN_MATCH_COUNT = 15
    
    if len(matches)>MIN_MATCH_COUNT:
        
        #convert keypoints to co-ordinates
        src_pts = np.float32([ keypoints_train[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints_query[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = size
        
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        area = cv2.contourArea(dst)
        original_area = h*w

        if abs(area-original_area)>500:
            return dst, -1

        # box = [np.int32(dst)][0].reshape(-1,2)
        # for i,j in box:
        #     if i<0 or j<0:
        #         return dst, -1
        # else:
        #     return dst,0
        return dst, 0

    else:
        return dst, -1

def get_coordinates(filename, dst):
    box = [np.int32(dst)][0].reshape(-1,2)
    x1,y1=box[0]
    x2,y2=box[2]
    data = []
    name = os.path.splitext(filename)[0]
    data = [name,[str(x1), str(y1), str(x2), str(y2)]]

    return data

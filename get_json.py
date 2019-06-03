import os
import copy
import cv2
from feature_match import *
import numpy as np
import json, codecs

wd = os.getcwd()
full_img_path = wd + '/MSD/sample_testset/images'
crop_img_path = wd + '/MSD/sample_testset/crops'

def get_features_all_crops():
    training = []
    for filename in os.listdir(crop_img_path):
        img = cv2.imread(crop_img_path+'/'+filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape
        keypoints, descriptors = orb(gray)
        data = [filename, keypoints, descriptors, size, 0]
        training.append(data)
    return training


def get_features_all_fulls():
    query = []
    for filename in os.listdir(full_img_path):
        img = cv2.imread(full_img_path+'/'+filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb(gray)
        data = [filename, keypoints, descriptors]
        query.append(data)
    return query

# training = get_features_all_crops()
# x,y,z = training[0]
# print(x)
# def get_coordinates():

all_data = {}
if __name__ == "__main__":
    training = get_features_all_crops()
    query = get_features_all_fulls()
    for q in query:
        filename, kpts, desp = q
        print(filename)
        if len(kpts)<10:
            name = os.path.splitext(filename)[0]
            data = [name, []]
            all_data[name] = data
            continue
        for t in training:
            tf, tkpts, tdesp, size, c = t
            if len(tkpts)<10:
                c=-1
                continue
            matches = feature_matcher(tdesp, desp)
            dst = remove_outiners(matches, tkpts, kpts, size)
            if dst.size !=0:
                data = get_coordinates(tf, dst)
                name = os.path.splitext(filename)[0]
                if name in all_data:
                    all_data[name].append(data)
                else:
                    all_data[name] = data
        

    for t in training:
        tf = t[0]
        c = t[4]
        if c==-1:
            name = os.path.splitext(tf)[0]
            data = [name, []]
            if 'na' in all_data:
                    all_data['na'].append(data)
            else:
                all_data['na'] = data

    with open('final_data.json', 'wb') as f:
        json.dump(all_data, codecs.getwriter('utf-8')(f), ensure_ascii=False, indent=4)




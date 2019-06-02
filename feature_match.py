import os
import copy

wd = os.getcwd()
path = wd + '/MSD/images/full'

def main():
    for filename in os.listdir(path):
        print(filename)

def orb_features(training_gray, query_gray):
    # Set the parameters of the ORB algorithm by specifying the maximum number of keypoints to locate and
    # the pyramid decimation ratio
    orb = cv2.ORB_create(5000, 2.0)

    # Find the keypoints in the gray scale training and query images and compute their ORB descriptor.
    # The None parameter is needed to indicate that we are not using a mask in either case.  
    keypoints_train, descriptors_train = orb.detectAndCompute(training_gray, None)
    keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)

    return keypoints_train, descriptors_train, keypoints_query, descriptors_query 

def feature_matcher(keypoints_train, descriptors_train, keypoints_query, descriptors_query):
    # Create a Brute Force Matcher object. "crossCheck"  is set to True so that the BFMatcher will only return consistent
    # pairs. Such technique usually produces best results with minimal number of outliers when there are enough matches.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Perform the matching between the ORB descriptors of the training image and the query image
    matches = bf.match(descriptors_train, descriptors_query)

    # The matches with shorter distance are required. So, the matches are sort according to distance
    matches = sorted(matches, key = lambda x : x.distance)

    return matches

def remove_outiners(matches, keypoints_train, keypoints_query):

    #minimum matches for sustainability of a object recognition
    MIN_MATCH_COUNT = 10
    
    if len(matches)>MIN_MATCH_COUNT:
        
        #convert keypoints to co-ordinates
        src_pts = np.float32([ keypoints_train[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints_query[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = training_gray.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        return dst

    else:
        return False

    return dst

def get_coordinates(dst):
    if not dst:
        box = [np.int32(dst)][0].reshape(-1,2)
        x1,y1=box[0]
        x2,y2=box[2]
        data = [x1, y1, x2, y2]

        return data
    
    return None


if __name__ == "__main__":
    main()
import cv2
import matplotlib.pyplot as plt

image1 = cv2.imread('./MSD/sample_testset/crops/625dd192-96cc-59cd-83d0-ad5fa710b4bb.jpg')
image2 = cv2.imread('./MSD/sample_testset/images/00c58d88-53e0-5314-947d-4763004df6df.jpg')

training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
query_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)
query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(5000, 2.0)
keypoints_train, descriptors_train = orb.detectAndCompute(training_gray, None)
keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)

query_img_keyp = copy.copy(query_image)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(descriptors_train, descriptors_query)
matches = sorted(matches, key = lambda x : x.distance)

print("Number of Keypoints Detected In The Training Image: ", len(keypoints_train))
print("Number of Keypoints Detected In The Query Image: ", len(keypoints_query))
print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))

list_kp1 = [keypoints_query[mat.queryIdx].pt for mat in matches] 
list_kp2 = [keypoints_train[mat.trainIdx].pt for mat in matches]
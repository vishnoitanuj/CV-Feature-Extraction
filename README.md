# Oriented FAST and Rotated BRIEF based Feature Matching

Oriented FAST and Rotated BRIEF (ORB) is basically a fusion of FAST keypoint detector and BRIEF descriptor with many modifications to enhance the performance. First it use FAST to find keypoints, then apply Harris corner measure to find top N points among them. It also use pyramid to produce multiscale-features. I will be using opencv-python for the implementation of the assignment and algorithm.

Major advantages of the ORB:
* Scale Invariance
* Rotational Invariance
* Illumination Invariance
* Noise Invariance

## The final json is contained in file <a href="https://github.com/vishnoitanuj/CV-Feature-Extraction/blob/master/final_data.json">final_data.json</a>

## Installing requirements.
 1) python version 3.5.1
 2) pip version 19.1.1
 3) Preferred OS: Ubuntu 16.04 (tested)

 Now go to the directly and run the following command:
  
  >* pip install -r requirements.txt

## Running the code
 The final code lies in the file <a href="https://github.com/vishnoitanuj/CV-Feature-Extraction/blob/master/get_json.py">get_json.py</a>

 For step wise understanding the ORB code please check: <a href="https://github.com/vishnoitanuj/CV-Feature-Extraction/blob/master/ORB-%20Feature%20Matcher.ipynb">ORB- Feature Matcher</a>

 So just run the file to get the output as final_data.json

## Code details
 The algorithm is majorly implemented in file <a href="https://github.com/vishnoitanuj/CV-Feature-Extraction/blob/master/feature_match.py">feature_match.py</a>, which contain the feature matching orb algorithm and also the outlier removal code.

For more insight into code implementation, please check the assignment report <a href="https://github.com/vishnoitanuj/CV-Feature-Extraction/blob/master/Report.pdf" target=__blank>Report</a>
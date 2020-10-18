"""
Object Tracking Algorithm for PiPlane Goggles

This algorithm works well in landscapes where PiPlane is one of the only objects in the sky, and works even better in low-lighting situations. It is
less robust in environments with multiple agents; however, this is not its intended use. The algorithm is very fast and can be run in real-time. This
works best when connected to a pair of video goggles so that a pilot can take full advantage of object tracking.

Authors: Nick Cich, Ethan Das, and Haris Miller
"""

import numpy as np
import cv2

# Accesses webcam for video
cap1 = cv2.VideoCapture(1)

# Parameters and variables used to modify behavior of object tracking
frame1, kp1, des1, max_x1, min_x1, max_y1, min_y1 = None, None, None, None, None, None, None
last_x_max, last_x_min, last_y_max, last_y_min = None, None, None, None
alpha = 0.9     # value between 0 and 1; closer to 0 means that bounding box changes more slowly between frames, closer to 1 means it changes quicker 
zoom = 10       # how much the image is trimmed; image length and width will decrease by 20 times this value in pixels

while(cap1.isOpened()):
    # updates previous values of variables
    prev_frame1, prev_kp1, prev_des1, prev_max_x1, prev_min_x1, prev_max_y1, prev_min_y1 = frame1, kp1, des1, max_x1, min_x1, max_y1, min_y1

    # reads video frame from webcam
    ret1, frame1 = cap1.read()

    # shrinks image based upon `zoom`
    if zoom > 0:
        frame1 = frame1[zoom * 10 : -zoom * 10, zoom* 10 : -zoom * 10]

    # uses ORB algorithm to compute image keypoints
    orb = cv2.ORB_create()
    kp1 = orb.detect(frame1, None)
    kp1, des1 = orb.compute(frame1, kp1)
    img3 = np.copy(frame1)

    # checks that keypoints have been detected
    if prev_frame1 is not None and len(kp1) > 0 and len(prev_kp1) > 0:
        # computes closest 14 matching keypoints between current image and previous image
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, prev_des1)
        matches = sorted(matches, key = lambda x:x.distance)[:14]

        # computes maximum and minimum coordinates of matching keypoints
        max_x1, min_x1 = np.amax([kp1[x.queryIdx].pt[0] for x in matches]), np.amin([kp1[x.queryIdx].pt[0] for x in matches])
        max_y1, min_y1 = np.amax([kp1[y.queryIdx].pt[1] for y in matches]), np.amin([kp1[y.queryIdx].pt[1] for y in matches])

        # checks that matching keypoints are not landscape features (these tend to have very low distances between frames)
        if prev_max_x1 is not None and matches[0].distance > 110:
            # updates bounding box using data about current frame's keypoints and previous frame's keypoints
            max_x1, min_x1 = prev_max_x1 + alpha * (max_x1 - prev_max_x1), prev_min_x1 + alpha * (min_x1 - prev_min_x1)
            max_y1, min_y1 = prev_max_y1 + alpha * (max_y1 - prev_max_y1), prev_min_y1 + alpha * (min_y1 - prev_min_y1)

            # makes sure that bounding box isn't too large
            if abs(max_x1 - min_x1) < np.shape(frame1)[1] // 4 and abs(max_y1 - min_y1) < np.shape(frame1)[0] // 4:
                # draws bounding box
                cv2.rectangle(img3, (int(min_x1), int(min_y1)), (int(max_x1), int(max_y1)), (0, 255, 0), 3)
                last_x_max, last_x_min, last_y_max, last_y_min = max_x1, min_x1, max_y1, min_y1

        elif prev_max_x1 is not None and last_x_max is not None:
            # draws rectangle in last known location of plane
            cv2.rectangle(img3, (int(last_x_min), int(last_y_min)), (int(last_x_max), int(last_y_max)), (0, 255, 0), 3)

    # shows image on screen
    cv2.imshow('frame', cv2.resize(img3, (np.shape(img3)[1] * 3, np.shape(img3)[0] * 3)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# closes access to webcam
cap1.release()
cv2.destroyAllWindows()
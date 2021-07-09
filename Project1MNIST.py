########################################################################
#                                                                      #
#    OpenCV and Keras-Based Real-Time Handwritten Digit Classifier     #
#                        Author: Jeremy Walker                         #                        
#   Author Affiliation: San Jose State University College of Science   #
#                     Date Last Modified: 3/23/2021                    #     
#                     E-mail: jeremy.walker@sjsu.edu                   #       
#                                                                      #     
########################################################################

import numpy as np
import cv2
import os
from keras.models import load_model

#function to perform necessary preprocessing steps to indentify ROIs
def digit_roi_extractor(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    blur = cv2.GaussianBlur(gray, (7,7), 5) #kernel and sigma parameters visually tuned
    binary = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY_INV)[1]
    canny = cv2.Canny(binary, 110, 190) #threshold and canny bounds also visually tuned
    contours, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [cv2.boundingRect(contour) for contour in contours]
    return rectangles, binary

os.chdir('/Users/jeremywalker/Desktop/CMPE258/Project1') #load the MNIST classification model
model = load_model('model_plain.h5')

cap = cv2.VideoCapture(0)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

while(True):
    ret, frame = cap.read()
    height, width, layers = frame.shape
    upper_left = (int(width/2 - width/4), int(height/2 - height/4))
    bottom_right = (int(width/2 + width/4), int(height/2 + height/4))
    #define the region of the video feed in which to place the digits
    img_frame = cv2.rectangle(frame, upper_left, bottom_right, (0,0,255), 3)
    digits_here = img_frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    rectangles, binary = digit_roi_extractor(digits_here)
    
    for rectangle in rectangles: #draw the bounding box and cut-out corresponding segment of binary image for each rectangle
        cv2.rectangle(digits_here, (rectangle[0], rectangle[1]), (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (0,0,255), 2)
        x_dim1 = rectangle[1]-35; x_dim2 = x_dim1 + rectangle[3]+35
        y_dim1 = rectangle[0]-35; y_dim2 = y_dim1 + rectangle[2]+35
        binary_img_seg = binary[x_dim1:x_dim2, y_dim1:y_dim2]
        try: #determine max. dimension and resize segment twice, then classify and display results
            max_dim = np.max(binary_img_seg.shape)
            digit = cv2.resize(binary_img_seg, (max_dim,max_dim), interpolation = cv2.INTER_AREA)
            digit = cv2.resize(digit, (28,28),  interpolation=cv2.INTER_AREA)
            test_img = np.reshape(digit, [1, 28, 28, 1])
            result = np.argmax(model.predict(test_img))
            cv2.putText(digits_here, str(int(result)), (rectangle[0]-20, rectangle[1]-20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
        except:
            pass
    
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()























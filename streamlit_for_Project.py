import cv2
import numpy as np
import streamlit as st
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import joblib #import load
import re
from imutils.object_detection import non_max_suppression

# Define HOG Parameters
orientations = 8
pixels_per_cell = (6, 6)
cells_per_block = (2, 2)
threshold = 0.3

# define the sliding window:


# image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    # this line and the line below actually defines the sliding part and loops over the x and y coordinates
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])


# Load the SVM model
#model = load("D:/university/UIT/term4/ComputerVision/Project/model.npy")
model = joblib.load("D:/university/UIT/term4/ComputerVision/Project/model.npy")

def rescale_bounding_box(bounding_box, scaling_factor):
    xmin, ymin, xmax, ymax = bounding_box

    # Rescale the bounding box coordinates
    rescaled_xmin = int(xmin / scaling_factor[1])
    rescaled_ymin = int(ymin / scaling_factor[0])
    rescaled_xmax = int(xmax / scaling_factor[1])
    rescaled_ymax = int(ymax / scaling_factor[0])

    rescaled_bounding_box = (rescaled_xmin, rescaled_ymin, rescaled_xmax, rescaled_ymax)
    return rescaled_bounding_box


def rescaled_bounding_boxes(bounding_boxes, scaling_factor):
    # Rescale the bounding boxes
    rescaled_bounding_boxes = []
    for bbox in bounding_boxes:
        rescaled_bbox = rescale_bounding_box(bbox, scaling_factor)
        rescaled_bounding_boxes.append(rescaled_bbox)

    return rescaled_bounding_boxes


def calculate_scaling_factor(original_width, original_height, current_width, current_height):
    scaling_factor_width = current_width / original_width
    scaling_factor_height = current_height / original_height
    scaling_factor = (scaling_factor_width, scaling_factor_height)
    return scaling_factor


# Define the sliding window
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])


# Load the SVM model
#model = load("D:\university\UIT\term4\ComputerVision\Project\model.npy")


# Streamlit app
def main():
    st.title("Pedestrian Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(300,200))

        # Perform sliding window detection
        scale = 0
        detections = []

        # defining the size of the sliding window (has to be, same as the size of the image in the training data)
        (winW, winH)= (64,128)
        windowSize=(winW,winH)
        downscale=1.5
        # Apply sliding window:
        for resized in pyramid_gaussian(gray, downscale=1.5): # loop over each layer of the image that you take!
            # loop over the sliding window for each layer of the pyramid
            for (x,y,window) in sliding_window(resized, stepSize=5, windowSize=(winW,winH)):
                # if the window does not meet our desired window size, ignore it!
                if window.shape[0] != winH or window.shape[1] !=winW: # ensure the sliding window has met the minimum size requirement
                    continue
                #window.shape
                #if window.ndim == 2:
                #  window = window[:, :, np.newaxis]
                #window=color.rgb2gray(window)
                #window.shape
                fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2')  # extract HOG features from the window captured
                fds = fds.reshape(1, -1) # reshape the image to make a silouhette of hog
                pred = model.predict(fds) # use the SVM model to make a prediction on the HOG features extracted from the window
                
                if pred == 1:
                    if model.decision_function(fds) > 0.6:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
                        #print("Detection:: Location -> ({}, {})".format(x, y))
                        #print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(fds)))
                        detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                                        int(windowSize[0]*(downscale**scale)), # create a list of all the predictions found
                                            int(windowSize[1]*(downscale**scale))))
            scale+=1
            
        clone = resized.copy()

        # apply nnms
        for (x_tl, y_tl, _, w, h) in detections:
            cv2.rectangle(gray, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness = 2)
        rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) # do nms on the detected bounding boxes
        sc = [score[0] for (x, y, score, w, h) in detections]
        #print("detection confidence score: ", sc)
        sc = np.array(sc)
        pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.2)
        
        # scaling bb
        scaling_factor = calculate_scaling_factor(image.shape[0], image.shape[1], gray.shape[0], gray.shape[1])
        rescaled = rescaled_bounding_boxes(pick, scaling_factor)
        #pick = cv2.dnn.NMSBoxes(bboxes = rects, scores = sc, score_threshold=0.4, nms_threshold=0.8)
        #print("detection utmost confidence score: ", pick)

        
        # Rescale bounding boxes và ground truth
        #file_path = "D:/UIT/Semester 4/Nhập môn Thị giác máy tính/FudanPed00003.txt"
        #scaling_factor = calculate_scaling_factor(image.shape[1], image.shape[0], image.shape[1], image.shape[0])
        #bounding_boxes = rescaled_bounding_boxes(read_bounding_boxes_from_file(file_path), scaling_factor)

        # Vẽ bounding boxes và ground truth lên ảnh
        for (xA, yA, xB, yB) in rescaled:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0,255,0), 3)

        # Hiển thị ảnh kết quả
        st.image(image, channels="BGR")


if __name__ == "__main__":
    main()

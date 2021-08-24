# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

import cv2 as cv
import argparse
import sys
import numpy as np
import pandas as pd
import os.path
import easyocr
from PIL import Image, ImageEnhance, ImageFilter, ImageFont, ImageDraw

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
#confThreshold = 0.1  # Confidence threshold
#nmsThreshold = 0.4  # Non-maximum suppression threshold
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
#Plate Scale
s = 15

inpWidth = 416  # 608     # Width of network's input image
inpHeight = 416  # 608     # Height of network's input image

parser = argparse.ArgumentParser(
    description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Load names of classes
classesFile = "D:\Python\yolo-license-plate-detection-master\classes.names"

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.

modelConfiguration = "D:\Python\yolo-license-plate-detection-master\darknet-yolov3.cfg"
modelWeights = "D:\Python\yolo-license-plate-detection-master\weights\model.weights"
pl = cv.imread('D:/Python/yolo-license-plate-detection-master/plate.png', cv.IMREAD_COLOR)
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)



def alignImages(im1, im2, im3, x, y):
  # Convert images to grayscale
  im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
  im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
  mask = np.zeros(im2Gray.shape[:2], dtype="uint8")
  cv.rectangle(mask,x, y, 255, -1)
   #cv.imshow("Rectangular Mask", mask)
  masked = cv.bitwise_and(im2Gray, im2Gray, mask=mask) 
  # Detect ORB features and compute descriptors.
  orb = cv.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(masked, None)
  # Match features.
  matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
  # Draw top matches
  imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv.imwrite("matches.jpg", imMatches)
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
  # Find homography
  h, mask = cv.findHomography(points1, points2, cv.RANSAC)
  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv.warpPerspective(im3, h, (width, height), cv.INTER_LINEAR, borderValue=(4, 4, 4))

  return im1Reg


#scale plates
def image_resize(im1,im2,x,y, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = im1.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return im1
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    im1 = cv.resize(im1, dim, interpolation = inter)
    l1 = np.zeros((im2.shape[0], im2.shape[1], 3), dtype=np.uint8)
    l1[:] = (4, 4, 4)
#print(im.shape)
# load resized image as grayscale
    h = im1.shape[0]
    w = im1.shape[1]
# load background image as grayscale
    hh = l1.shape[0]
    ww = l1.shape[1]
# compute xoffset and yoff for placement of upper left corner of resized image   
    yoff = x
    xoff = y
# use numpy indexing to place the resized image in the center of background image
    result = l1.copy()
    #print(yoff,yoff+h)
    #print(xoff,xoff+w)
    result[yoff:yoff+h, xoff:xoff+w] = im1
    #result[x, y] = im1
    return result

def trans(im1):
  #make background transparent
  image_bgr = im1
  # get the image dimensions (height, width and channels)
  h, w, c = im1.shape
  # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
  image_bgra = np.concatenate([image_bgr, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
  # create a mask where white pixels ([255, 255, 255]) are True
  black = np.all(image_bgr == [4, 4, 4], axis=-1)
  # change the values of Alpha to 0 for all the white pixels
  image_bgra[black, -1] = 0
  return image_bgra

def merg(trans_img,background):
  h, w, c = trans_img.shape
  result = np.zeros((h, w, 3), np.uint8)
  alpha = trans_img[:, :, 3] / 255.0
  result[:, :, 0] = (1. - alpha) * background[:, :, 0] + alpha * trans_img[:, :, 0]
  result[:, :, 1] = (1. - alpha) * background[:, :, 1] + alpha * trans_img[:, :, 1]
  result[:, :, 2] = (1. - alpha) * background[:, :, 2] + alpha * trans_img[:, :, 2]
  return result

def center_text(img, font, text, color=(0, 0, 0)):
    draw = ImageDraw.Draw(img)
    text_width, text_height = draw.textsize(text, font)
    text_height += int(text_height*0.24)
    position = ((strip_width-text_width)/2,(strip_height-text_height)/2)
    draw.text(position, text, color, font=font)
    return img

# Get the names of the output layers


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box


def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    #cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s: %s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(
        label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 0, 255), cv.FILLED)
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
    #cv.putText(frame, label, (left, top),cv.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

# Remove the bounding boxes with low confidence using non-maxima suppression


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        #print("out.shape : ", out.shape)
        for detection in out:
            # if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:
            confidence = scores[classId]
            if detection[4] > confThreshold:
                print(detection[4], " - ", scores[classId],
                      " - th : ", confThreshold)
                print(detection)
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left,
                 top, left + width, top + height)
    return left,top, left + width, top + height

# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(
        cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(
        frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    test = postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

## create cords data
    L=[test[0]]
    T=[test[1]]
    R=[test[2]]
    B=[test[3]]
    df = pd.DataFrame({ 'Left':L,
                'Top':T,
                'Right':R,
                'Bottom':B})

    df['TL'] = list(zip(df.Left, df.Top))
    df['RB'] = list(zip(df.Right, df.Bottom))
    #print(df)

    # prep frame for reading cropped out cropping top:botton, Left:Right
    cropped_image = frame[df.iloc[0,1]-50:df.iloc[0,3]+50, df.iloc[0,0]-50:df.iloc[0,2]+50]
    cropped_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
    cropped_image - cv.bilateralFilter(cropped_image,11,17,17)
    Q, cropped_image  = cv.threshold(cropped_image , 0, 255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imwrite("text.jpg", cropped_image)

#read & create number plate
    
    # prep frame for reading cropped out cropping top:botton, Left:Right
    cropped_image = frame[df.iloc[0,1]-50:df.iloc[0,3]+50, df.iloc[0,0]-50:df.iloc[0,2]+50]
    cropped_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
    cropped_image - cv.bilateralFilter(cropped_image,11,17,17)
    Q, cropped_image  = cv.threshold(cropped_image , 0, 255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imwrite("text.jpg", cropped_image)
    #read plates
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    text = result[0][-2]
    print("Plate reads =" +text.upper())

    #create plates
    background = Image.open('D:\Python\yolo-license-plate-detection-master\Blank white.png')
    strip_width, strip_height = background.size
    font = ImageFont.truetype("D:\Python\yolo-license-plate-detection-master\CharlesWright-Bold.ttf", 410)
    im = center_text(background, font, text.upper())
    im = np.array(im) 
    im = im[:, :, ::-1].copy() 
    im =  cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    cv.imwrite("Gplate.jpg", im)
    im = cv.imread("Gplate.jpg", cv.IMREAD_COLOR)
    #print(im.shape)
    
    #scale and position plates
    im = image_resize(im,frame,df.iloc[0,1],df.iloc[0,0], width = (df.iloc[0,2]-df.iloc[0,0]+s))
    cv.imwrite("Gplate_scale.jpg", im)

    pl = image_resize(pl,frame,df.iloc[0,1],df.iloc[0,0], width = (df.iloc[0,2]-df.iloc[0,0]+s))
    cv.imwrite("Cplate_scale.jpg", pl)

    #align
    imReg = alignImages(im, frame,pl,df.iloc[0,4],df.iloc[0,5])

    #transparent
    image_bgra = trans(imReg)
    cv.imwrite("transparent plate.jpg", image_bgra)

    #overlay transparent img to frame.
    Output = merg(image_bgra,frame)


    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, Output.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

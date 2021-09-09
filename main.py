import cv2 as cv
import numpy as np
import math
import os

def findDistance(p1, p2, x, thresh):
    num = (p1[1] - p2[1])*x[0] + (p2[0] - p1[0])*x[1] + (p1[0] * p2[1] - p2[0] * p1[1])
    dist = (abs(num) / math.sqrt((p1[1] - p2[1])*(p1[1] - p2[1]) + (p2[0] - p1[0])*(p2[0] - p1[0]))) // thresh
    return dist

dir = os.path.join(os.getcwd(), 'input_images')
# images = os.listdir(dir)
images = ['sample2.png']

for filename in images:
    img = cv.imread('input_images/' + filename)
    blank = np.zeros(img.shape, dtype = "uint8")

    edges = cv.Canny(img, 200, 250)
    contours, heirarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE )
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
    boxes = []
    centers = []
    indices = []
    points = []

    index = 0
    threshH = 0
    threshW = 0
    for i in contours:
        rect = cv.minAreaRect(i)
        box = cv.boxPoints(rect)
        points.extend(box)
        box = np.int0(box)
        M = cv.moments(i)
        if M['m00'] == 0:
            continue
        center = [int(M['m10']/M['m00']), int(M['m01']/M['m00'])]

        if index != 0:
            if center[0] == centers[-1][0] and center[1] == centers[-1][1]:
                continue
        
        if index >= 9:
            continue

        (x, y), (width, height), angle = rect
        threshH += height
        threshW += width
        boxes.append(box)
        centers.append(center)
        blank = cv.fillPoly(blank, [box], (255,255,255))
        index += 1

    threshH = (threshH) // (len(boxes) * 2)
    threshW = (threshW) // (len(boxes) * 2)

    outerBox = cv.boxPoints(cv.minAreaRect(np.int32(points)))
    outerBox = np.int0(outerBox)

    img = cv.bitwise_and(img, blank)
    sorterBoxes = sorted(zip(boxes, centers), key=lambda x: findDistance(outerBox[0], outerBox[3], x[0][0], threshW))
    sorterBoxes = sorted(sorterBoxes, key=lambda x: findDistance(outerBox[0], outerBox[1], x[0][0], threshH))
    boxes = [x for x, _ in sorterBoxes]
    centers = [x for _, x in sorterBoxes]

    # figuring out the colors

    cubeColors = []
    colors = []

    hsvImage = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    for i in boxes:
        height, width = hsvImage.shape[:2]
        mask = np.zeros((height, width, 1), np.uint8)
        cv.fillPoly(mask, [i], (255,255,255))
        hue = cv.mean(hsvImage, mask)[0]

        index = -1
        for j in range(len(colors)):
            if abs(colors[j] - hue) < 1:
                index = j+1
                break

        if index == -1:
            colors.append(hue)
            cubeColors.append(len(colors))
        else:
            cubeColors.append(index)


    output = ""
    for i in range(0, len(cubeColors), 3):
        for j in range(i, i+3):
            output += str(cubeColors[j]) + ' '
        output += '\n'
    
    outfilename = "".join(filename.split('.')[:-1])
    file = open('output/output_' + outfilename + '.txt', 'w+')
    file.write(output)
    file.close()
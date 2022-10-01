import cv2 as cv
import numpy as np
import math

vid = cv.VideoCapture(0)

while True:
    ret, frame = vid.read()

    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #cv.imshow('gray', grayscale)

    bgModel = cv.createBackgroundSubtractorMOG2(0, 50)
    fgmask = bgModel.apply(grayscale)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv.erode(fgmask, kernel, iterations=1)
    img = cv.bitwise_and(frame, frame, mask=fgmask)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    lower = np.array([0,133,77],np.uint8)
    upper = np.array([235,163,127],np.uint8)
    skinMask = cv.inRange(hsv, lower, upper)
    #skinMask = cv.fastNlMeansDenoising(skinMask, None, 20, 7, 21)

    thresh = cv.threshold(skinMask, 128, 255, cv.THRESH_BINARY)[1]
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        rect = cv.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100: continue
        x, y, w, h = rect
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = frame[y:y+h, x:x+w]
        cv.imshow('cropped', cropped)



    #imCrop = grayscale[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv.imshow('frame', frame)
    cv.imshow('Threshold Hands', skinMask)



    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.relase()

cv.destroyAllWindows()

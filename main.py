import cv2 as cv
import numpy as np
import tensorflow as tf
import os

#name = 'Z'
#directory = 'C:\\Users\\lucas\\Downloads\\Data\\' + name
model = tf.keras.models.load_model("ASL_CNN.model")
letterOptions = ['a', 'b', 'c', 'd', 'del', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'nothing', 'o', 'p', 'q',
                 'r', 's', 'space', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ImageSize = 100

def callDatabase(img, data):
    new_img = img.reshape(-1, ImageSize, ImageSize, 3)
    predict = data.predict([new_img])[0].tolist()
    rip = max(predict)
    index = predict.index(rip)
    print(letterOptions[index])
    return letterOptions[index]



vid = cv.VideoCapture(0)
#os.chdir(directory)
#counter = 0

while True:
    ret, frame = vid.read()

    # Creates Grayscale image
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Creates SkinMask parameters
    bgModel = cv.createBackgroundSubtractorMOG2(0, 50)
    fgmask = bgModel.apply(grayscale)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv.erode(fgmask, kernel, iterations=1)
    img = cv.bitwise_and(frame, frame, mask=fgmask)

    # Creates SkinMask with Parameters and limits
    hsv = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    lower = np.array([100, 133, 77], np.uint8)
    upper = np.array([235, 163, 127], np.uint8)
    skinMask = cv.inRange(hsv, lower, upper)

    # Creates threshold limit from the SkinMask image
    thresh = cv.threshold(skinMask, 128, 255, cv.THRESH_BINARY)[1]
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Draw Rectangle from SkinMask contours onto original frame
    frameCopy = frame.copy()
    #if counter > 400:
    #    break
    for c in contours:
        rect = cv.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100: continue
        x, y, w, h = rect
        maxSize = max(x+w, y+h)
        cv.rectangle(frame, (x-10, y-10), (max(x+w, maxSize), max(y+h, maxSize)), (0, 255, 0), 2)
        # Crops image and displays
        cropped = frameCopy[y-10:max(y+h, maxSize), x-10:max(x+w, maxSize)]
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            cropped = cv.resize(cropped, (ImageSize, ImageSize), interpolation=cv.INTER_AREA)
            cv.putText(frame, callDatabase(cropped, model), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv.imshow('cropped', cropped)
            #cv.imwrite(name + 'DataModel' + str(counter) + '00.jpg', cropped)
            #counter += 1


    # Shows Original frame and skinMask
    cv.imshow('frame', frame)
    cv.imshow('Threshold Hands', skinMask)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()

cv.destroyAllWindows()

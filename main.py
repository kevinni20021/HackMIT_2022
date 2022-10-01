import cv2 as cv

vid = cv.VideoCapture(0)

while True:
    ret, frame = vid.read()

    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', frame)
    cv.imshow('gray', grayscale)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.relase()

cv.destroyAllWindows()

import tensorflow as tf
import cv2 as cv
from PIL import Image

model = tf.keras.models.load_model("ASL_CNN.model")
letterOptions = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'del', 'space']
ImageSize = 64


# os.chdir(directory)

def callDatabase(img, data):
    new_img = img.reshape(-1, ImageSize, ImageSize, 3)
    predict = data.predict([new_img])[0].tolist()
    rip = max(predict)
    index = predict.index(rip)
    print(letterOptions[index])
    print(predict)
    return letterOptions[index]


img = cv.imread(
    "C:\\Users\\lucas\\Documents\\Code\\Hackathon\\HackMIT_2022\\ASL_CNN.model\\Personal_Data\\A\\ADataModel10600.jpg",
    cv.IMREAD_COLOR)
callDatabase(img, model)

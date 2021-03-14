import cv2
import os
import numpy as np


# suppress info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

HEIGHT   = 1072
WIDTH    = 1920
CHANNELS = 3
CLASSES  = 7

color_class = {(2, 2, 255): 1, (27, 138, 28): 2, (55, 255, 55): 3, (81, 255, 241): 4, (188, 22, 183): 5, (54, 61, 110): 6, (244, 14, 14): 7}
class_color = {1: (2, 2, 255), 2: (27, 138, 28), 3: (55, 255, 55), 4: (81, 255, 241), 5: (188, 22, 183), 6: (54, 61, 110), 7: (244, 14, 14)}


print('Getting data')

path = f'dataset/masks'

filenames = [filename for filename in os.listdir(path)]

X_train = np.zeros((len(filenames), HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)
y_train = np.zeros((len(filenames), HEIGHT, WIDTH, CLASSES), dtype=np.float64)

for i, filename in enumerate(filenames):
    print(i)

    image_path = f'dataset/images/{filename}'
    mask_path  = f'dataset/masks/{filename}'

    X_train[i] = cv2.resize(cv2.imread(image_path), (WIDTH, HEIGHT))

    mask = cv2.resize(cv2.imread(mask_path), (WIDTH, HEIGHT))
    
    for x in range(HEIGHT):
        for y in range(WIDTH):
            for class_ in class_color:
                if class_color[class_] == tuple(mask[x][y]):
                    y_train[i][x][y][class_-1] = 1.
                    break


print('Training model')
import tensorflow as tf
from model import UNet

model = UNet(HEIGHT, WIDTH, CHANNELS)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience = 5)

model.fit(X_train, y_train, batch_size=1, epochs=100, callbacks=[callback])

model.save(f'models/slanic.h5')


from tensorflow import keras

print('Loading model')
model = keras.models.load_model(f'models/slanic.h5')

print('Computing images')
video_path  = 'video.mp4'

video = cv2.VideoCapture(video_path)

success = True
count = 0

while success:
    count += 1

    print(f'Processing frame {count}/9021')
            
    result_path = f'tmp2/{count}.png'

    success, image = video.read()

    if image is None:
        break

    img = cv2.resize(image, (WIDTH, HEIGHT))

    image = np.zeros((HEIGHT, WIDTH, CHANNELS))
    pred = np.reshape(model.predict(np.reshape(img, (1, HEIGHT, WIDTH, 3))), (HEIGHT, WIDTH, CLASSES))

    for i in range(HEIGHT):
        for j in range(WIDTH):
            c = np.argmax(pred[i][j])

            if pred[i][j][c] > 0.3:
                image[i][j] = np.array(class_color[c+1])

    res = image * 0.3 + cv2.resize(img, (WIDTH, HEIGHT)) * 0.7

    cv2.imwrite(result_path, res.astype(np.uint8))

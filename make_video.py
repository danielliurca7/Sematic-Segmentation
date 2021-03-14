import cv2
import os
import glob

writer = cv2.VideoWriter('slanic_model.mp4', -1, 30, (1920, 1072))

file_list = glob.glob('tmp2/*.png')

for img in sorted(file_list, key=lambda x: int(x[4:-4])):
    print(img)
    
    image = cv2.imread(img)

    writer.write(image)

writer.release()


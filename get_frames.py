import cv2

def write_frames(video, n):
    success = True
    count = 0


    while success:
        success, image = video.read()

        if image is not None and count % n == 0:
            path = f'dataset/images/frame_{count}.png'

            cv2.imwrite(path, image)

        count += 1


video = cv2.VideoCapture('video.MP4')
frames = write_frames(video, 90)
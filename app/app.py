import os
import cv2
from model import Model


def read_media_file(media_file):
    model = Model()

    print(f'Opening {media_file}...')
    capture = cv2.VideoCapture(media_file)
    if not capture.isOpened():
        raise Exception(f'Could not open {media_file}.')
    else:
        print(f'Opened {media_file}.')

    while True:
        video_playing, frame = capture.read()
        if not video_playing:
            break

        fire, prob = model.predict(frame)
        print(fire, str(round(prob * 100, 2)) + '%')

        cv2.putText(
            img=frame,
            text=f'{fire} ({prob*100:.2f}%)',
            org=(10, 60),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=4,
            color=(255, 255, 255),
            thickness=3,
            lineType=cv2.LINE_AA
        )

        cv2.imshow(media_file, frame)
        ESCAPE_KEY = 27
        if cv2.waitKey(1) == ESCAPE_KEY:
            break
    capture.release()
    cv2.destroyAllWindows()


MEDIA_DIR = 'media'
while True:
    for media_file in os.listdir(MEDIA_DIR):
        media_path = f'{MEDIA_DIR}/{media_file}'
        read_media_file(media_path)
        os.rename(media_path, media_file)

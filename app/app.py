import cv2
from model import Model


def play_video_file(video_file):
    model = Model()
    model_queue = model.create_infer_queue()

    print(f'Opening {video_file}...')
    capture = cv2.VideoCapture(video_file)
    if not capture.isOpened():
        raise Exception(f'Could not open {video_file}.')
    else:
        print(f'Opened {video_file}.')

    while True:
        video_playing, frame = capture.read()
        if not video_playing:
            break

        model_input = model.preprocess(frame)
        model_queue.start_async({model.input_layer: model_input})
        model_queue.wait_all()
        fire, fire_prob = model.get_result()

        cv2.putText(
            img=frame,
            text=f'{fire} ({fire_prob:.2f})',
            org=(10, 60),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=4,
            color=(255, 255, 255),
            thickness=3,
            lineType=cv2.LINE_AA
        )

        cv2.imshow(video_file, frame)
        ESCAPE_KEY = 27
        if cv2.waitKey(1) == ESCAPE_KEY:
            break
    capture.release()
    cv2.destroyAllWindows()


play_video_file('videos/forest.mp4')
play_video_file('videos/fire.mp4')
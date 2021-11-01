import cv2
import imutils
import argparse


def detection(frame):
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride=(3, 3), padding=(6, 6), scale=1.03)
    people = 0

    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, '{}'.format(people), (x, y), cv2.FONT_ITALIC,0.5, (255, 255, 0), 1)
        people += 1

    cv2.putText(frame, 'No of People: {}'.format(people - 1), (50, 80), cv2.FONT_ITALIC, 0.7, (255, 255, 0), 2)
    cv2.imshow('output', frame)

    return frame

def detectNCountByImage(path, output_path):
    image = cv2.imread(path)

    image = imutils.resize(image, width = min(800, image.shape[1]))

    result_image = detection(image)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectNCountByVideo(path, writer):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if not check:
        print('Enter valid path or Invalid path')
        return

    print('Detection of People ... ')
    while video.isOpened():

        check, frame = video.read()

        if check:
            frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            frame = detection(frame)

            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()


def detectNCountByCamera(writer):
    video = cv2.VideoCapture(0)
    print('Detecting people...')

    while True:
        check, frame = video.read()

        frame = detection(frame)
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def humanDetector(args):
    i_path = args["image"]
    v_path = args['video']
    if str(args["camera"]) == 'true' : camera = True
    else : camera = False

    writer = None
    if args['output'] is not None and i_path is None:
        writer = cv2.VideoWriter(args['output'],cv2.VideoWriter_fourcc(*'MJPG'), 10, (700,700))

    if camera:
        print(' Opening Web Cam.')
        detectNCountByCamera(writer)
    elif v_path is not None:
        print(' Opening Video from path.')
        detectNCountByVideo(v_path, writer)
    elif i_path is not None:
        print(' Opening Image from path.')
        detectNCountByImage(i_path, args['output'])

def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('-v', '--video', default=None)
    arg_parse.add_argument('-i', '--image', default=None)
    arg_parse.add_argument('-c', '--camera', default=True)
    arg_parse.add_argument('-o', '--output', type=str)
    args = vars(arg_parse.parse_args())

    return args


if __name__ == '__main__':
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    args = argsParser()
    humanDetector(args)
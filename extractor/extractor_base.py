import cv2


def extract_video2frames(video_path, interval):
    capture_tool = cv2.VideoCapture(video_path)
    extracted_frame_list = []
    extracted_frame_idx = []
    frame_count = 0
    while True:
        success, frame = capture_tool.read()
        if not success:
            break
        frame_count += 1
        if (frame_count - 1) % interval != 0:
            continue
        extracted_frame_list.append(frame)
        # count from 1 to n
        extracted_frame_idx.append(frame_count)
    return extracted_frame_idx, extracted_frame_list



class BaseExtractor(object):
    def __init__(self):
        super().__init__()

    def extract_all(self):
        raise NotImplementedError

    def extract_item(self, *inputs):
        raise NotImplementedError

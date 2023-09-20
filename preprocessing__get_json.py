import cv2
import pandas as pd
import os
import json
from json import JSONEncoder
import numpy as np

PATH = r'C:\Users\Daria\Documents\online_courses\2023_09_ODS_systemdesign\gesture_recognition\data'
folder_with_data = 'slovo_full360'
STEP = 10

annotations = pd.read_csv(os.path.join(PATH, 'annotations_full360.tsv'), sep='\t')

def get_frames(video_path: str, data: dict, step=1, dsize=None, fps=30):
    # step - sampling step (in frames)
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    n_frames = int(data['length'] * fps / step)
    for i in range(n_frames):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, (i * step // fps) - 1)
        success, frame = vidcap.read()
        if success:
            frames.append(
                {
                    'attachment_id': data['attachment_id'],
                    'user_id': data['user_id'],
                    #'frame_init': frame[:, :, ::-1],
                    'frame_resized': cv2.resize(frame[:, :, ::-1], dsize) if dsize is not None else frame[:, :, ::-1],
                    'text': data['text'],
                    'length': data['length'],
                    'begin': data['begin'], # frame index
                    'end': data['end'], # frame index
                    'height': data['height'],
                    'width': data['width'],
                    'index': i,
                    'boarders_of_gesture': i < data['begin'] or i >= data['end'] # те кадры, которые вне заданных рамок, считаем "обрамляющими" жест
                })
    return frames


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


for di, data in enumerate(annotations.to_dict('records')):
    if data['train']: #  and f'{data["attachment_id"]}.mp4' in os.listdir(os.path.join(PATH, folder_with_data, 'train'))
        folder = 'train'
    else:
        folder = 'test'
    path_from = os.path.join(PATH, folder_with_data, folder, '{}.mp4'.format(data["attachment_id"]))
    path_to = os.path.join(PATH, folder_with_data, f'jsons_step{STEP}', folder, '{}.json'.format(data["attachment_id"]))
    result = get_frames(video_path=path_from, data=data, step=STEP, dsize=(224,224)) # dsize для efficient net
    with open(path_to, 'w', encoding='utf8') as f:
        json.dump(result, f, cls=NumpyArrayEncoder)
    if (di+1) % 10 == 0:
        print('processed {} rows out of {}'.format(di, len(annotations)))
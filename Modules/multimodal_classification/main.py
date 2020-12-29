import os
import torch
import numpy as np

from AnalysisModule import settings
from WebAnalyzer.utils.media import frames_to_timecode
from Modules.multimodal_classification.multi_modal_classification_model import Transformer

class multimodal_classification:
#    model = None
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
	# TODO
	#   - initialize and load model here
	#        model_path = os.path.join(self.path, "model.txt")
	#        self.model = open(model_path, "r")
        src_vocab = 55 * 10
        trg_vocab = 10
        output_words = 6
        d_model = 256
        N = 1
        heads = 4
        model = Transformer(src_vocab, trg_vocab, d_model,N,heads, output_words)
        model_path = os.path.join(self.path, "model_2.pt")
        model.load_state_dict(torch.load(model_path),strict=False)
        self.model = model

    def inference_by_word(self, word_path):
        result = []
        result = {'frame_result': []}

        X = np.load(word_path, allow_pickle=True)
        for i in X:
            model_input = torch.tensor(i,dtype=torch.float32).unsqueeze(0)
            preds = model(model_input,None)
            result['frame_result'].append(list(preds.max(-1)[1].squeeze(0).numpy()))
        self.result = result
        return self.result

    def inference_by_image(self, image_path):
        result = []
        # TODO
        #   - Inference using image path

        # result sample
        result = {"frame_result": [
            {
                # 1 bbox & multiple object
                'label': [
                    {'description': 'person', 'score': 1.0},
                    {'description': 'chair', 'score': 1.0}
                ],
                'position': {
                    'x': 0.0,
                    'y': 0.0,
                    'w': 0.0,
                    'h': 0.0
                }
            },
            {
                # 1 bbox & 1 object
                'label': [
                    {'description': 'car', 'score': 1.0},
                ],
                'position': {
                    'x': 100.0,
                    'y': 100.0,
                    'w': 100.0,
                    'h': 100.0
                }
            }
        ]}
        self.result = result

        return self.result

    def inference_by_video(self, frame_path_list, infos):
        results = []
        video_info = infos['video_info']
        frame_urls = infos['frame_urls']
        fps = video_info['extract_fps']
        for idx, (frame_path, frame_url) in enumerate(zip(frame_path_list, frame_urls)):
            result = self.inference_by_image(frame_path)
            result["frame_url"] = settings.MEDIA_URL + frame_url[1:]
            result["frame_number"] = int((idx + 1) * fps)
            result["timestamp"] = frames_to_timecode((idx + 1) * fps, fps)
            results.append(result)

        self.result = results

        return self.result

    def inference_by_audio(self, audio_path, infos):
        video_info = infos['video_info']
        result = []
        # TODO
        #   - Inference using image path
        #   -
        result = {"audio_result": [
            {
                # 1 timestamp & multiple class
                'label': [
                    {'score': 1.0, 'description': 'class_name'},
                    {'score': 1.0, 'description': 'class_name'}
                ],
                'timestamp': "00:00:01:00"
            },
            {
                # 1 timestamp & 1 class
                'label': [
                    {'score': 1.0, 'description': 'class_name'}
                ],
                'timestamp': "00:00:01:00"
            }
        ]}
        self.result = result

        return self.result

    def inference_by_text(self, data, video_info):
        result = []
        # TODO
        #   - Inference using image path
        #   -
        result = {"text_result": [
            {
                # 1 timestamp & multiple class
                'label': [
                    {'score': 1.0, 'description': 'word_name'},
                    {'score': 1.0, 'description': 'word_name'}
                ],
            },
            {
                # 1 timestamp & 1 class
                'label': [
                    {'score': 1.0, 'description': 'word_name'}
                ],
            }
        ]}
        self.result = result

        return self.result

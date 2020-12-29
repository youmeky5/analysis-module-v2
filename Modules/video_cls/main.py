import json
import os
import torch
from Modules.dummy.example import test
from Modules.video_cls.transformer_six_words import Transformer
from Modules.video_cls.json_to_vector import j2v
class Video_Classification:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        # TODO
        #   - initialize and load model here
        src_vocab = 55 * 3 # x_input size
        trg_vocab = 10 #output_size
        output_words = 6 #number_of_output_words
        d_model = 256 #hidden unit size
        N = 2 # quantity of blocks
        heads = 4 # head
        model = Transformer(src_vocab, trg_vocab, d_model, N, heads, output_words)
        self.path = self.path
        model_path = os.path.join(self.path, "model_1.pt")
        model.float()
        model.load_state_dict(torch.load(model_path),strict = False)
        self.model = model

    def inference_by_data(self, aggregation_result):
#        aggregation_result = '{}'.format(aggregation_result)
#        aggregation_result=aggregation_result.replace('\'','\"')
        json_data = json.loads(aggregation_result)
        feature_vector = j2v(self.path,json_data,3)
        feature_vector = torch.tensor(feature_vector,dtype=torch.float32).unsqueeze(0)
        pred = self.model(feature_vector,None)
        pred = list(pred.max(-1)[1].squeeze(0).numpy())
        recommend_place_={'0':'경복궁','1':'한강','2':'망원시장','3':'코엑스','4':'서울역','5':'지하철','6':'서울랜드','7':'롯데월드','8':'남산타워','9':'인사동'}
        where_={'0':'홍대','1':'연남동','2':'성수동','3':'잠실','4':'삼청동','5':'부산','6':'명동','7':'종로','8':'압구정','9':'익선동'}
        who_={'0':'황인여자','1':'황인남자','2':'백인여자','3':'백인남자','4':'아이','5':'황인여자','6':'황인남자','7':'백인여자','8':'백인남자','9':'아이'}
        recommend_doing_={'0':'구경','1':'맛집투어','2':'쇼핑','3':'관광','4':'놀이기구','5':'카페투어','6':'걷기','7':'예술관방문','8':'사진찍기','9':'숙박'}
        recommend_eat_={'0':'라면','1':'빵','2':'떡볶이','3':'중식','4':'커피','5':'쌀국수','6':'볶음밥','7':'김치찌개','8':'파스타','9':'일식'}
        etc_={'0':'탈북 청소년 드림학교','1':'웃음','2':'학생','3':'맛집','4':'야경','5':'한강','6':'카페','7':'맛있다','8':'화장품','9':'조형물'}
        # TODO
        #   - Inference using aggregation result
        #result = {"aggregation_result": [
        #    {
        #        # 1 timestamp & multiple class
        #        'label': [
        #            {'description': 'word_name', 'score': 1.0},
        #            {'description': 'word_name', 'score': 1.0}
        #        ],
        #    },
        #    {
        #        # 1 timestamp & 1 class
        #        'label': [
        #            {'description': 'word_name', 'score': 1.0}
        #        ],
        #    }
        #]}
        result = {"aggregation_result":[
            {'where':where_[str(pred[0])],
             'who':who_[str(pred[1])],
             'recommend_place':recommend_place_[str(pred[2])],
             'recommend_doing':recommend_doing_[str(pred[3])],
             'recommend_eat':recommend_eat_[str(pred[4])],
             'etc':etc_[str(pred[5])]}]
                            }
        self.result = result

        return self.result
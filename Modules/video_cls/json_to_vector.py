import bcolz
import pickle
import os
import numpy as np


def j2v(path,json_data,n):
    glove_path=os.path.join(path, 'glove.6B')
    vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    result=[]
    for i in json_data['result']:
        sub_result=[]
        for j in i['frame_result']:
            x = j['position']['x']
            y = j['position']['y']
            w = j['position']['w']
            h = j['position']['h']
            score = j['label'][0]['score']
            word = j['label'][0]['description']
            word = ''.join(''.join(word.strip().split('_')).split(' ')).lower()
            if word in glove.keys():
                feat_vec = list(glove[word])
                feat_vec += [x,y,w,h,score]
            else:
                feat_vec = list(glove['padding'])
                feat_vec += [0,0,0,0,0]
            feat_vec = np.array(feat_vec)
            sub_result.append(feat_vec)
        sub_result.sort(key=lambda y:y[-1],reverse=True)
        if len(sub_result)>=n:
            sub_result=sub_result[:n]
        else:
            for t in range(n-len(sub_result)):
                feat_vec=list(glove['padding'])
                feat_vec+=[0,0,0,0,0]
                feat_vec=np.array(feat_vec)
                sub_result.append(feat_vec)
        sub_result=np.array(sub_result).reshape(-1)
        result.append(sub_result)
    return np.array(result)
    

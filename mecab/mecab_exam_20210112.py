
# 은전한닢 VER
''' 
import MeCab

m = MeCab.Tagger()
out = m.parse("안녕하세요")
print(out)
'''

# konply 버전
from konlpy.tag import Mecab
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
out = mecab.morphs("오늘저녁먹었어??")
print(out)

# Word2Vec 모듈
from gensim.models import Word2Vec, KeyedVectors

import numpy as np

#테스트
training_data_path = "C:\github\python\mecab\TrainingData\\training_data_dialogflow.txt"
raw_data_path = 'C:\github\python\mecab\TrainingData\\raw_data_dialogflow.txt'
model_path = 'C:\github\python\mecab\TrainingData\\training_model'
dialog_file = open(training_data_path, 'rt', encoding='utf-8')
vector_size=300
training_word_data = []

##### 형태소 분석 ######
while True:
    # 사용자 발화 값 Line read
    line = dialog_file.readline()

    if line:
        temp = []

        #형태소 분석
        tokenlist = mecab.pos(line)

        for word in tokenlist:
            try:
                value = word[0]
                tag = word[1]

                if tag in ["NNG", "NNP", "VA", "SN", "SL"]:
                    try:			
                        #EOS는 문장의 끝
                        if value != 'EOS' and len(value) > 1:
                            temp.append(value)
                    except:
                        pass
            except:
                pass

        if temp:
            training_word_data.append(temp)
            #print(training_word_data)
    else:
        break

##### 형태소 분석 ######    


##### Word2Vec Model 생성 ###### 
#model = Word2Vec.load(model_path + '.model')

#Model 생성
model = Word2Vec(training_data_path, size=300, window=3, min_count=1, workers=1)

model.build_vocab(training_word_data, update=True)
model.train(training_word_data, epochs=model.epochs, total_examples=model.corpus_count)

model.save(model_path + '.model')
##### Word2Vec Model 생성 ###### 


###### Pickle Data 생성 ########
f = open(raw_data_path, 'rt', encoding='utf-8')
raw_data = f.readlines()
f.close()

for item in raw_data:
    featureVec = np.zeros((vector_size,), dtype="float32")
    nwords = 0

    tokenlist = mecab.pos(item)

    for word in tokenlist:
        try:
            #token = word.split(',')[0].split('\t')
            value = word[0]
            tag = word[1].split('+')[0]

            #if tag not in ['SS', 'SF']:
            if tag in ["NNG", "NNP", "VA", "SN", "SL"]:#, "VV"]: 
                try:			
                    if value != 'EOS' and len(value) > 1:
                        featureVec = np.add(featureVec, model.wv[value])
                        print(featureVec)
                        nwords = nwords + 1
                except:
                    #print('no voca: ', value)
                    pass
        except:
            pass

    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)

################################
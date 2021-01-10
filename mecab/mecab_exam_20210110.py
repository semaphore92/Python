
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

#테스트
training_data_path = "C:\github\python\mecab\TrainingData\\training_data_dialogflow.txt"
dialog_file = open(training_data_path, 'rt', encoding='utf-8')
training_word_data = []

##### 형태소 분석 ######
while True:
    line = dialog_file.readline()

    if line:
        temp = []

        tokenlist = mecab.pos(line)
        print(tokenlist)

        for word in tokenlist:
            try:
                #token = word.split(',')[0].split('\t')
                value = word[0]
                tag = word[1]

                #if tag not in ['SS', 'SF']:
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
            print(training_word_data)
    else:
        break

##### 형태소 분석 ######    

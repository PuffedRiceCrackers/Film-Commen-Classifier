# -*- coding: utf-8 -*-

from konlpy.tag import Kkma
from konlpy.tag import Twitter
from konlpy.tag import Okt
from math import log
from random import randint
import json

okt = Okt()

# 에러를 대비해서 단계마다 json 파일을 만듦
def makeJson(filename, data):
    with open(filename, 'w', encoding="utf-8") as make_file:
        json.dump(data, make_file, ensure_ascii=False, indent="\t")

def loadJson(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

# 형태소 분석 + 사용단어 추출
def tokenize(filename, order):
    print(f"{order} : tokenize 중")
    data = []
    wordpool = []
    countPos = 0
    countNeg = 0
    count = 0
    verbPreFix = None

    with open(filename, 'r', encoding='utf8') as oldF:
        for line in oldF:
            if count == 0:
                count += 1
                continue
            else:
                # ① id를 제외하고 문장만추출
                temp = line.split()
                del temp[0]

                if order == 'train':
                    label = temp[-1]
                    if label == '1':
                        countPos += 1
                    if label == '0':
                        countNeg += 1
                    del temp[-1]    
                targetSentence = ' '.join(temp)
                
                # ② 형태소분석기로 쪼갬
                refined = okt.pos(targetSentence, norm=True, stem=True)
                buffer = []   


                # ③ 사용된 단어는 중복되지 않도록 wordpool이라는 자료구조에 넣음
                for idx in range(len(refined)):

                    # refined[idx] = [('재밌다','verb'),('영화','noun'),('재밌다', 'verb')]
                    if refined[idx][0] != ".":
                        if refined[idx][1] == 'VerbPrefix':     # [1] '못', '안' 등은 아무것도 하지 말고 저장만 하고 있음
                            if idx+1 <= len(refined)-1:
                                if refined[idx+1][1] == 'Verb': # 혹시 그 다음 'Verb'가 안나오는 경우엔 무효
                                    verbPreFix = refined[idx][0]
                        else:
                            if refined [idx][1] == 'Verb':      # [2] 그 다음 동사가 나오면 저장해둔 verbPreFix를 붙인 형태를 저장
                                if verbPreFix != None:
                                    tempTuple = (verbPreFix + ' ' + refined[idx][0], refined[idx][1])
                                    del refined[idx]
                                    refined.insert(idx, tempTuple)
                                    verbPreFix = None
                            if order == 'train':
                                if [refined[idx][0],refined[idx][1],0,0,0,0,0] not in wordpool:
                                    wordpool.append([ refined[idx][0] , refined[idx][1] ,0,0,0,0, 0])
                    buffer.append( [ refined[idx][0],refined[idx][1] ] )

                # ④ 쪼개진 형태소는 다시 모아서 data라는 자료구조에 넣어줌
                if order == 'train':
                    buffer.append(label)
                data.append(buffer)
                count += 1

    if order == 'train':
        data.append([countPos, countNeg])

    print(f"{order} : tokenize완료")
    dict = {'refinedData' : data, 'wordpool' : wordpool}
    return dict
 
# 어떤 data set에서 특정 (word,type)이 긍부정에서 각각 몇번 쓰였는지를 리턴함
def countPosNeg(word, type, data):
    countPos = 0
    countNeg = 0

    # sentence : [[영화,명사],[별로다,형용사],'0']
    for sentence in data[:len(data)-1]: #마지막엔 전체 pos,neg의 개수가 들어있음
        for idx in range(len(sentence)-1):
            if sentence[idx][0] == word and sentence[idx][1] == type:
                if sentence[-1] == '1':
                    countPos += 1
                    break               # 한번 나온게 확인 되었으면 그 문장에서 세는건 그만 둔다
                if sentence[-1] == '0':
                    countNeg += 1
                    break
    return {'countPos' : countPos, 'countNeg' : countNeg}

# training set에서 사용된 word들의 조건부확률을 계산
def labelWordPool(data, wordpool):
    print("wordpool 통계분석 시작")

    # training data에서 긍정/부정 셋의 개수
    numPos = data[-1][0]
    numNeg = data[-1][1]
    numAll = numPos+numNeg

    for idx in range(0, len(wordpool)):
        
        countDict = countPosNeg(wordpool[idx][0], wordpool[idx][1], data)
        countPos = countDict['countPos']
        countNeg = countDict['countNeg']

        # wordpool[idx] = ["단어", "품사" log(S|P), log(-S|P), log(S|N), log(-S|N), Bool]
        # 마지막 bool은 나중에 Test Data Set에서, 해당 단어가 사용되었는지를 표시할 때 쓰려고 놔둠
        if countPos == 0:
            wordpool[idx][2] = log(1/(1*numAll),2)
            wordpool[idx][3] = log(1-(1/(1*numAll)),2)
        else:
            wordpool[idx][2] = log(countPos/numPos, 2)
            wordpool[idx][3] = log(1-(countPos/numPos), 2)
        if countNeg == 0:
            wordpool[idx][4] = log(1/(1*numAll), 2)
            wordpool[idx][5] = log(1-(1/(1*numAll)), 2)
        else:
            wordpool[idx][4] = log(countNeg/numNeg, 2)
            wordpool[idx][5] = log(1-(countNeg/numNeg), 2)

    # 전체 데이터에서 pos와 neg의 확률도 넘겨줌           
    pos = log(numPos / (numPos + numNeg), 2)
    neg = log(numNeg / (numPos + numNeg), 2)
    wordpool.append([pos,neg])
    return {'wordpool' : wordpool}


# wordpool 자료구조에서 sentence안에 사용된 단어를 사용되었다고 표시해줌
def markUsedWord(wordpool, sentence, exclude):
    for targetWord in sentence[:len(sentence)-exclude]:
        for idx in range(len(wordpool)-1): #마지막에는 logPos, logNeg정보가 들어있음
            if wordpool[idx][0] == targetWord[0] and wordpool[idx][1] == targetWord[1]:
                wordpool[idx][6] = 1
    return wordpool
            
# 나이브 베이지안 모델에 근거해 labeling
def naiveBayes(data, order):
    print(f"{order} : 나이브베이지안")

    # training data는 test data와 달리 끝에 붙어있는 라벨이 있으므로 계산시 제외해야 함
    if order == 'train':  # training, data = [ [ [좋다,형용사],[영화, 명사],0 ], ....[500, 550] ]
        excludeLabel = 1  # test, data = [ [[좋다,형용사],[영화, 명사]], .... [[나쁘다,형용사],[영화, 명사]]]
        excludeCount = 1
    if order == 'test':  
        excludeLabel = 0
        excludeCount = 0
    
    # 
    for index in range(0,len(data)-excludeCount): 
        pos = 0
        neg = 0
        wordpool = loadJson('wordpool.json') # 모든 쿼리는 처음에는 Bool 부분이 모두 0인 wordpool이 필요함
        logPos = wordpool[-1][0]
        logNeg = wordpool[-1][1]
        wordpool = markUsedWord(wordpool, data[index], excludeLabel) # 해당쿼리에서 사용된 단어가
                                                                     # wordpool의 Bool부분에 표시됨
        
        # item = ["단어", "품사" log(S|P), log(-S|P), log(S|N), log(-S|N), Bool]
        for item in wordpool[:len(wordpool)-1]: # 마지막에는 [logPos, logNeg] 정보가 들어있음
            if item[6]==1:
                pos += item[2]
                neg += item[4]
            if item[6]==0:
                pos += item[3]
                neg += item[5]

        # 실제 그 라벨일 확률도 더해줌
        pos += logPos
        neg += logNeg

        if pos < neg:
            data[index].append('0')
        elif pos > neg:
            data[index].append('1')
        else:
            data[index].append(f"{randint(0,1)}")
    return data

# ratings_valid 의 성공률을 반납
def checkSuccess(data):
    count = 0
    correct = 0
    for item in data:
        count += 1
        if item[-1] == item[-2]:
            correct += 1
    return round(correct/count, 5)

# ratings_result_txt에 쓰는 부분
def writeResult(filepath, data):
    with open(filepath) as fp:
        lines = fp.read().splitlines()
    with open('ratings_result.txt', "w") as fp:
        count = 0
        for line in lines: # 첫번째줄 제외
            if count == 0:
                count += 1
                continue
            else:
                if type(data[count-1][-1])==list:
                    print(f"data[count-1][-1] = {data[count-1][-1]}")
                print(line + data[count-1][-1], file=fp)
                count += 1

def main():
    
    ##Training / json 만들기
    dict = tokenize('ratings_train.txt', 'train')
    wordpoolDict = labelWordPool(dict['refinedData'],dict['wordpool'])
    wordpool = wordpoolDict['wordpool']
    makeJson('ratings_train.json', dict['refinedData'])
    makeJson('wordpool.json', wordpool)

    #Train 데이터를 json에서 불러옴
    refinedData = loadJson('ratings_train.json')
    wordpool = loadJson('wordpool.json')
    logPos = wordpool[-1][0]
    logNeg = wordpool[-1][1]

    # valid 데이터 분석 / json 만들기
    validDict = tokenize('ratings_valid.txt', 'train')
    makeJson('ratings_valid.json', validDict['refinedData'])
    validData = loadJson('ratings_valid.json')
    labeledValidData = naiveBayes(validData, 'train')
    makeJson('valid_naive.json', labeledValidData)
    labeledValidData = loadJson('valid_naive.json')
    print(f"success rate = {checkSuccess(labeledValidData)}")

    # test 데이터에 라벨링하기
    dict = tokenize('ratings_test.txt', 'test')
    makeJson('ratings_test.json', dict['refinedData'])
    testData = loadJson('ratings_test.json')
    labeledTestData = naiveBayes(testData, 'test')
    makeJson('ratings_result.json', labeledTestData)
    writeResult('ratings_test.txt',labeledTestData)

main()

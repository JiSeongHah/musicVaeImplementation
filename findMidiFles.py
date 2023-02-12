import os


# baseDir 내의 모든 mid 파일 탐색
def getAllMidiFles(baseDir):

    lst = []
    WALKS = os.walk(baseDir)

    for Root,Dir,Fles in WALKS:
        for eachFle in Fles:
            if eachFle.endswith('.mid'):
                lst.append(os.path.join(Root,eachFle))


    return lst



def splitTrainVal(pathLst):

    trainLst = []
    valLst = []

    for eachPath in pathLst:
        if 'eval_session' not in eachPath:
            trainLst.append(eachPath)
        else:
            valLst.append(eachPath)


    return trainLst,valLst

import numpy as np
np.seterr(divide='ignore')
from hmmlearn import hmm
import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.externals import joblib

def normalize(sequence, axis=None):
    min = sequence.min(axis=axis, keepdims=True)
    max = sequence.max(axis=axis, keepdims=True)
    result = (sequence-min)/(max-min)
    print(max, min)
    return result

if __name__ == '__main__':

    print("generating HMM with sequences in each emotion ...")

    # 対象にするクラスをまとめておく
    emotions = ["disgust", "anger", "sadness", "happy" , "fear", "surprise"]

    subject=""
    num=""

    # 各クラスのデータリストとその要素数
    list_disgust = []
    list_anger = []
    list_sadness = []
    list_happy = []
    list_fear = []
    list_surprise = []
    count_disgust = 0
    count_anger = 0
    count_sadness = 0
    count_happy = 0
    count_fear = 0
    count_surprise = 0

    # confusion matrix を定義
    confusion_mat = [[0 for i in range(6)] for j in range(6)]

    '''
    print("Collecting CK+ samples ...")

    # CK+データセットにOpenFaceを適用して得たデータをリスト構造で取得
    for i in range(999):
        subject = "S" + "%03.f"%(i + 1)
        if os.path.exists("../data/CK+/" + subject):
            for j in range(999):
                num = "%03.f"%(j + 1)
                if os.path.exists("../data/CK+/" + subject + "/" + num + "/" + num + "_AU.csv"):
                    f = open("../data/CK+/" + subject +"/" + num + "/" + num + "_Emotion.txt", "r")
                    emotion = f.read()
                    if "disgust" in emotion:
                        sequence_disgust = np.genfromtxt("../data/CK+/" + subject +"/" + num + "/" + num + "_AU.csv", delimiter=",", skip_header=1)
                        list_disgust.append(sequence_disgust)
                        count_disgust+=1
                    elif "anger" in emotion:
                        sequence_anger = np.genfromtxt("../data/CK+/" + subject +"/" + num + "/" + num + "_AU.csv", delimiter=",", skip_header=1)
                        list_anger.append(sequence_anger)
                        count_anger+=1
                    elif "sadness" in emotion:
                        sequence_sadness = np.genfromtxt("../data/CK+/" + subject +"/" + num + "/" + num + "_AU.csv", delimiter=",", skip_header=1)
                        list_sadness.append(sequence_sadness)
                        count_sadness+=1
                    elif "happy" in emotion:
                        sequence_happy = np.genfromtxt("../data/CK+/" + subject +"/" + num + "/" + num + "_AU.csv", delimiter=",", skip_header=1)
                        list_happy.append(sequence_happy)
                        count_happy+=1
                    elif "fear" in emotion:
                        sequence_fear = np.genfromtxt("../data/CK+/" + subject +"/" + num + "/" + num + "_AU.csv", delimiter=",", skip_header=1)
                        list_fear.append(sequence_fear)
                        count_fear+=1
                    elif "surprise" in emotion:
                        sequence_surprise = np.genfromtxt("../data/CK+/" + subject +"/" + num + "/" + num + "_AU.csv", delimiter=",", skip_header=1)
                        list_surprise.append(sequence_surprise)
                        count_surprise+=1
    '''

    print("Collecting MMI samples ...")

    # MMI データセットにOpenFaceを適用して得たデータセットをリストに追加
    for i in range(100, 2054):
        for subject in range(54):
            subject = "%03.f"%(subject+1)
            for sequence in range(150):
                sequence = "%03.f"%(sequence+1)
                if os.path.exists("../data/MMI_custum/Sessions/" + str(i) + "/S" + subject + "-" + sequence + "-new.mp4"):
                    tree = ET.parse("../data/MMI_custum/Sessions/" + str(i) + "/S" + subject + "-" + sequence + ".xml")
                    root = tree.getroot()
                    for metatag in root.iter('Metatag'):
                        meta = metatag.get('Name')
                        if meta == "Emotion":
                            emotion_MMI = metatag.get('Value')
                            if "1" == emotion_MMI:
                                sequence_anger = np.genfromtxt("../data/MMI_custum/Sessions/" +str(i) + "/S" + subject + "-" + sequence + "-AU/S" + subject+ "-" + sequence + "-AU.csv", delimiter=",", skip_header=1)
                                """
                                sequence_anger = sequence_anger.tolist()
                                length = len(sequence_anger)
                                # 各フレームの二乗平均誤差を計算して無表情の時間を削減する
                                rms = []
                                for j in range(length):
                                    row_data = pd.Series(sequence_anger[j])
                                    rms.append(np.sqrt((row_data ** 2).sum() / row_data.size))
                                for j in range(length)[::-1]:
                                    if rms[j] < 0.5:
                                        sequence_anger.pop(j)
                                sequence_anger = np.asarray(sequence_anger)
                                if len(sequence_anger) != 0:
                                    list_anger.append(sequence_anger)
                                    count_anger+=1
                                """
                                """
                                # シークエンスの後ろ半分を切り取る
                                sequence_anger = sequence_anger.tolist()
                                length = len(sequence_anger)
                                for j in range(length//2):
                                        sequence_anger.pop()
                                sequence_anger = np.asarray(sequence_anger)
                                """
                                print("../data/MMI_custum/Sessions/" +str(i) + "/S" + subject + "-" + sequence + "-AU/S" + subject+ "-" + sequence + "-AU.csv")
                                sequence_anger = normalize(sequence_anger)
                                list_anger.append(sequence_anger)
                                count_anger+=1
                            if "2" == emotion_MMI:
                                sequence_disgust = np.genfromtxt("../data/MMI_custum/Sessions/" +str(i) + "/S" + subject + "-" + sequence + "-AU/S" + subject+ "-" + sequence + "-AU.csv", delimiter=",", skip_header=1)
                                """
                                sequence_disgust = sequence_disgust.tolist()
                                length = len(sequence_disgust)
                                # 各フレームの二乗平均誤差を計算して無表情の時間を削減する
                                rms = []
                                for j in range(length):
                                    row_data = pd.Series(sequence_disgust[j])
                                    rms.append(np.sqrt((row_data ** 2).sum() / row_data.size))
                                for j in range(length)[::-1]:
                                    if rms[j] < 0.5:
                                        sequence_disgust.pop(j)
                                sequence_disgust = np.asarray(sequence_disgust)
                                if len(sequence_disgust) != 0:
                                    list_disgust.append(sequence_disgust)
                                    count_disgust+=1
                                """
                                """
                                sequence_disgust = sequence_disgust.tolist()
                                length = len(sequence_disgust)
                                for j in range(length//2):
                                        sequence_disgust.pop()
                                sequence_disgust = np.asarray(sequence_disgust)
                                """
                                print("../data/MMI_custum/Sessions/" +str(i) + "/S" + subject + "-" + sequence + "-AU/S" + subject+ "-" + sequence + "-AU.csv")
                                sequence_disgust = normalize(sequence_disgust)
                                list_disgust.append(sequence_disgust)
                                count_disgust+=1
                            if "3" == emotion_MMI:
                                sequence_fear = np.genfromtxt("../data/MMI_custum/Sessions/" +str(i) + "/S" + subject + "-" + sequence + "-AU/S" + subject+ "-" + sequence + "-AU.csv", delimiter=",", skip_header=1)
                                """
                                sequence_fear = sequence_fear.tolist()
                                length = len(sequence_fear)
                                # 各フレームの二乗平均誤差を計算して無表情の時間を削減する
                                rms = []
                                for j in range(length):
                                    row_data = pd.Series(sequence_fear[j])
                                    rms.append(np.sqrt((row_data ** 2).sum() / row_data.size))
                                for j in range(length)[::-1]:
                                    if rms[j] < 0.5:
                                        sequence_fear.pop(j)
                                sequence_fear = np.asarray(sequence_fear)
                                if len(sequence_fear) != 0:
                                    list_fear.append(sequence_fear)
                                    count_fear+=1
                                """
                                """
                                sequence_fear = sequence_fear.tolist()
                                length = len(sequence_fear)
                                for j in range(length//2):
                                        sequence_fear.pop()
                                sequence_fear = np.asarray(sequence_fear)
                                """
                                print("../data/MMI_custum/Sessions/" +str(i) + "/S" + subject + "-" + sequence + "-AU/S" + subject+ "-" + sequence + "-AU.csv")
                                sequence_fear = normalize(sequence_fear)
                                list_fear.append(sequence_fear)
                                count_fear+=1
                            if "4" == emotion_MMI:
                                sequence_happy = np.genfromtxt("../data/MMI_custum/Sessions/" +str(i) + "/S" + subject + "-" + sequence + "-AU/S" + subject+ "-" + sequence + "-AU.csv", delimiter=",", skip_header=1)
                                """
                                sequence_happy = sequence_happy.tolist()
                                length = len(sequence_happy)
                                # 各フレームの二乗平均誤差を計算して無表情の時間を削減する
                                rms = []
                                for j in range(length):
                                    row_data = pd.Series(sequence_happy[j])
                                    rms.append(np.sqrt((row_data ** 2).sum() / row_data.size))
                                for j in range(length)[::-1]:
                                    if rms[j] < 0.5:
                                        sequence_happy.pop(j)
                                sequence_happy = np.asarray(sequence_happy)
                                if len(sequence_happy) != 0:
                                    list_happy.append(sequence_happy)
                                    count_happy+=1
                                """
                                """
                                sequence_happy = sequence_happy.tolist()
                                length = len(sequence_happy)
                                for j in range(length//2):
                                        sequence_happy.pop()
                                sequence_happy = np.asarray(sequence_happy)
                                """
                                print("../data/MMI_custum/Sessions/" +str(i) + "/S" + subject + "-" + sequence + "-AU/S" + subject+ "-" + sequence + "-AU.csv")
                                sequence_happy = normalize(sequence_happy)
                                list_happy.append(sequence_happy)
                                count_happy+=1
                            if "5" == emotion_MMI:
                                sequence_sadness = np.genfromtxt("../data/MMI_custum/Sessions/" +str(i) + "/S" + subject + "-" + sequence + "-AU/S" + subject+ "-" + sequence + "-AU.csv", delimiter=",", skip_header=1)
                                """
                                sequence_sadness = sequence_sadness.tolist()
                                length = len(sequence_sadness)
                                # 各フレームの二乗平均誤差を計算して無表情の時間を削減する
                                rms = []
                                for j in range(length):
                                    row_data = pd.Series(sequence_sadness[j])
                                    rms.append(np.sqrt((row_data ** 2).sum() / row_data.size))
                                for j in range(length)[::-1]:
                                    if rms[j] < 0.5:
                                        sequence_sadness.pop(j)
                                sequence_sadness = np.asarray(sequence_sadness)
                                if len(sequence_sadness) != 0:
                                    list_sadness.append(sequence_sadness)
                                    count_sadness+=1
                                """
                                """
                                sequence_sadness = sequence_sadness.tolist()
                                length = len(sequence_sadness)
                                for j in range(length//2):
                                        sequence_sadness.pop()
                                sequence_sadness = np.asarray(sequence_sadness)
                                """
                                print("../data/MMI_custum/Sessions/" +str(i) + "/S" + subject + "-" + sequence + "-AU/S" + subject+ "-" + sequence + "-AU.csv")
                                sequence_sadness = normalize(sequence_sadness)
                                list_sadness.append(sequence_sadness)
                                count_sadness+=1
                            if "6" == emotion_MMI:
                                sequence_surprise = np.genfromtxt("../data/MMI_custum/Sessions/" +str(i) + "/S" + subject + "-" + sequence + "-AU/S" + subject+ "-" + sequence + "-AU.csv", delimiter=",", skip_header=1)
                                """
                                sequence_surprise = sequence_surprise.tolist()
                                length = len(sequence_surprise)
                                # 各フレームの二乗平均誤差を計算して無表情の時間を削減する
                                rms = []
                                for j in range(length):
                                    row_data = pd.Series(sequence_surprise[j])
                                    rms.append(np.sqrt((row_data ** 2).sum() / row_data.size))
                                for j in range(length)[::-1]:
                                    if rms[j] < 0.5:
                                        sequence_surprise.pop(j)
                                sequence_surprise = np.asarray(sequence_surprise)
                                if len(sequence_surprise) != 0:
                                    list_surprise.append(sequence_surprise)
                                    count_surprise+=1
                                """
                                """
                                sequence_surprise = sequence_surprise.tolist()
                                length = len(sequence_surprise)
                                for j in range(length//2):
                                        sequence_surprise.pop()
                                sequence_surprise = np.asarray(sequence_surprise)
                                """
                                print("../data/MMI_custum/Sessions/" +str(i) + "/S" + subject + "-" + sequence + "-AU/S" + subject+ "-" + sequence + "-AU.csv")
                                sequence_surprise = normalize(sequence_surprise)
                                list_surprise.append(sequence_surprise)
                                count_surprise+=1


    print()
    print("component:3")
    print("""transmat_ = [[0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                [0.0, 0.0, 1.0]]""")
    print()
    # 10分割検証用に、データのリストを10分割する
    for emo in emotions:
        exec("list_" + emo + "_folded = np.array_split(list_" + emo + ", 10)")

    # 検証に使うデータリストの番号で一回のループを制御
    for test_num in range(10):

        # モデルを定義・更新
        # 各クラスに対する3状態のleft-to-right HMMを定義
        for emo in emotions:
            exec("model_" + emo + " = hmm.GaussianHMM(n_components=5, covariance_type=\"diag\", init_params=\"cm\", params=\"cmt\")")
            exec("model_" + emo + ".startprob_ = np.array([1.0, 0.0, 0.0, 0.0, 0.0])")
            exec("""model_""" + emo + """.transmat_ = ([[0.5, 0.5, 0.0, 0.0, 0.0],
                                                        [0.0, 0.5, 0.5, 0.0, 0.0],
                                                        [0.0, 0.0, 0.5, 0.5, 0.0],
                                                        [0.0, 0.0, 0.0, 0.5, 0.5],
                                                        [0.0, 0.0, 0.0, 0.0, 1.0]])""")                                    

        print()
        print("Validation: " + str(test_num + 1))
        print()

        # 学習用と検証用のリストを定義
        list_disgust_learn = []
        list_anger_learn = []
        list_sadness_learn = []
        list_happy_learn = []
        list_fear_learn = []
        list_surprise_learn = []
        list_disgust_test = []
        list_anger_test = []
        list_sadness_test = []
        list_happy_test = []
        list_fear_test = []
        list_surprise_test = []
        # 学習用のリストと検証用のリストの長さを定義
        count_disgust_learn = 0
        count_anger_learn = 0
        count_sadness_learn = 0
        count_happy_learn = 0
        count_fear_learn = 0
        count_surprise_learn = 0
        count_disgust_test = 0
        count_anger_test = 0
        count_sadness_test = 0
        count_happy_test = 0
        count_fear_test = 0
        count_surprise_test = 0

        for i in range(10):
            # 各クラスに対して学習用と検証用のリストを取得
            for emo in emotions:
                if i != test_num:
                    exec("list_" + emo + "_learn.append(list_" + emo + "_folded[i])")
                    exec("count_" + emo + "_learn += len(list_" + emo + "_folded[i])")
                else:
                    exec("list_" + emo + "_test.append(list_" + emo + "_folded[i])")
                    exec("count_" + emo + "_test += len(list_" + emo + "_folded[i])")
        
        # 学習用のデータリストと検証用のデータリストを一つのリストに連結
        for emo in emotions:
            exec("X_" + emo + " = np.concatenate(list_" + emo + "_learn)")
            exec("X_" + emo + "_test = np.concatenate(list_" + emo + "_test)")


        # 各クラスの各シークエンスの長さを全てリストとして保持
        lengths_disgust = []
        lengths_anger = []
        lengths_sadness = []
        lengths_happy = []
        lengths_fear = []
        lengths_surprise = []

        for emo in emotions:
            exec("""for i in range(count_""" + emo + """_learn):
                lengths_""" + emo + """.append(len(X_""" + emo + """[i]))
            """)
            exec("X_" + emo + " = np.concatenate(X_" + emo + ")")

        # 学習
        for emo in emotions:
            exec("model_" + emo + ".fit(np.asarray(X_" + emo + "), np.asarray(lengths_" + emo + "))")

        # 正解の感情をインクリメントして管理 ( disgust = 0, anger = 1, sadness= 2, happy = 3, fear = 4, surprise = 5 )
        Correct_Emotion = 0 

        # 評価
        # emo1: 正解のクラス, emo2: 識別に使うHMMのクラス
        for emo1 in emotions:
            print("------------------------test data: " + emo1 + " ------------------------")
            print()
            exec("""for j in range(count_""" + emo1 + """_test):
                emotion_eval_list = []
                for emo2 in emotions:
                    exec("score_" + emo2 + " = model_" + emo2 + ".score(X_""" + emo1 + """_test[j])")
                    exec("print(\\"" + emo2 + ":\\" + str(score_" + emo2 + "))")
                    exec("emotion_eval_list.append([score_" + emo2 + ", \\"\" + emo2 + \"\\"])")
                emotion_eval_list.sort()
                emotion_eval_list.reverse()
                print("=======================>")
                if emotion_eval_list[0][0] == 0:
                    print("couldn't recognize")
                else:
                    print("recognized as " + emotion_eval_list[0][1])
                print()
                if emotion_eval_list[0][1] == "disgust":
                    Recognized_Emotion = 0
                elif emotion_eval_list[0][1] == "anger":
                    Recognized_Emotion = 1
                elif emotion_eval_list[0][1] == "sadness":
                    Recognized_Emotion = 2
                elif emotion_eval_list[0][1] == "happy":
                    Recognized_Emotion = 3
                elif emotion_eval_list[0][1] == "fear":
                    Recognized_Emotion = 4
                elif emotion_eval_list[0][1] == "surprise":
                    Recognized_Emotion = 5
                confusion_mat[Correct_Emotion][Recognized_Emotion] += 1
            """)
            Correct_Emotion += 1
        for emo in emotions:
            exec('print(model_' + emo + '.means_)')
            # exec('joblib.dump(model_' + emo + ', "../models/model_' + emo + '_'+ str(test_num) + '.pkl")')

    print("Calculating the confusion matrix")
    print()
    for i in range(6):
        if i == 0:
            print('{:^10}'.format("disgust"), end="|")
        elif i == 1:
            print('{:^10}'.format("anger"), end="|")
        elif i == 2:
            print('{:^10}'.format("sadness"), end="|")
        elif i == 3:
            print('{:^10}'.format("happy"), end="|")
        elif i == 4:
            print('{:^10}'.format("fear"), end="|")
        elif i == 5:
            print('{:^10}'.format("surprise"), end="|")
        for j in range(6):
            print('{:^4d}'.format(confusion_mat[i][j]),end="")
        print()

    # 評価のための計算
    i = 0
    data_count = 0
    correct_count = 0
    precision = [0 for i in range(6)]
    recall = [0 for i in range(6)]
    f_measure = [0 for i in range(6)]
    for emo in emotions:
        TP = confusion_mat[i][i]
        FP = 0
        FN = 0
        for j in range(6):
            if i != j:
                FP += confusion_mat[i][j]
                FN += confusion_mat[j][i]
            else:
                correct_count += confusion_mat[i][j]
            data_count += confusion_mat[i][j]
        precision[i] = (float(TP) / float(TP + FP))
        recall[i] = (float(TP) / float(TP + FN))
        f_measure[i] = (2 * recall[i] * precision[i]) / (recall[i] + precision[i])
        i += 1
    recognition_accuracy = float(correct_count) / float(data_count)

    print()


    print('{:^10}'.format("precision"), end="|")
    for i in range(6):
        print('{:^10f}'.format(precision[i]),end="")
    print()
    print('{:^10}'.format("recall"), end="|")
    for i in range(6):
        print('{:^10f}'.format(recall[i]),end="")
    print()
    print('{:^10}'.format("F-measure"), end="|")
    for i in range(6):
        print('{:^10f}'.format(f_measure[i]),end="")
    print()
    print()
    print("Recognition Accuracy : " + str(recognition_accuracy))

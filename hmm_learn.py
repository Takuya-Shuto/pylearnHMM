import numpy as np
from hmmlearn import hmm
import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.externals import joblib

def normalize(sequence, axis=None):
    min = sequence.min(axis=axis, keepdims=True)
    max = sequence.max(axis=axis, keepdims=True)
    result = (sequence-min)/(max-min)
    return result

if __name__ == '__main__':

    print("generating HMM with sequences in each emotion ...")

    # 対象にするクラスをまとめておく
    emotions = ["disgust", "anger", "sadness", "happy" , "fear", "surprise"]

    # 各クラスに対する3状態のleft-to-right HMMを定義
    for emo in emotions:
        exec("model_" + emo + " = hmm.GaussianHMM(n_components=5, n_iter=100, covariance_type=\"diag\", init_params=\"cm\", params=\"cmt\")")
        exec("model_" + emo + ".startprob_ = np.array([1.0, 0.0, 0.0, 0.0, 0.0])")
        exec("""model_""" + emo + """.transmat_ = np.array([[0.5, 0.5, 0.0, 0.0, 0.0],
                                                    [0.0, 0.5, 0.5, 0.0, 0.0],
                                                    [0.0, 0.0, 0.5, 0.5, 0.0],
                                                    [0.0, 0.0, 0.0, 0.5, 0.5],
                                                    [0.0, 0.0, 0.0, 0.0, 1.0]])""")


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

    '''
    # CK+データセットにOpenFaceを適用して得たデータをリスト構造で取得
    for i in range(999):
        subject = "S" + "%03.f"%(i + 1)
        if os.path.exists("../data/CK+/" + subject):
            for j in range(999):
                num = "%03.f"%(j + 1)
                if os.path.exists("../data/CK+/" + subject +"/" + num + "/" + num + "_AU.csv"):
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
                                sequence_surprise = normalize(sequence_surprise)
                                list_surprise.append(sequence_surprise)
                                count_surprise+=1

    # 各クラスのデータリストを連結する
    for emo in emotions:
        exec("X_" + emo + " = np.concatenate(list_" + emo + ")")
    
    # 各クラスの各シークエンスの長さを全てリストとして保持
    lengths_disgust = []
    lengths_anger = []
    lengths_sadness = []
    lengths_happy = []
    lengths_fear = []
    lengths_surprise = []

    # 連結したリストの各シークエンスの長さを保存しておく
    for emo in emotions:
        exec("""for i in range(count_""" + emo + """):
            lengths_""" + emo + """.append(len(list_""" + emo + """[i]))
        """)

    # 学習
    for emo in emotions:
        print(emo)
        if emo == "disgust":
            exec("model_" + emo + ".fit(np.asarray(X_" + emo + "[:, [0, 1, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]), np.asarray(lengths_" + emo + "))")
        elif emo == "anger":
            exec("model_" + emo + ".fit(np.asarray(X_" + emo + "[:, [0, 1, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]), np.asarray(lengths_" + emo + "))")
        elif emo == "sadness":
            exec("model_" + emo + ".fit(np.asarray(X_" + emo + "[:, [0, 1, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]), np.asarray(lengths_" + emo + "))")
        elif emo == "happy":
            exec("model_" + emo + ".fit(np.asarray(X_" + emo + "[:, [0, 1, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]), np.asarray(lengths_" + emo + "))")
        elif emo == "fear":
            exec("model_" + emo + ".fit(np.asarray(X_" + emo + "[:, [0, 1, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]), np.asarray(lengths_" + emo + "))")
        elif emo == "surprise":
            exec("model_" + emo + ".fit(np.asarray(X_" + emo + "[:, [0, 1, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]), np.asarray(lengths_" + emo + "))")
        print("startprob")
        exec("print(model_" + emo + ".startprob_)")
        print("means")
        exec("print(model_" + emo + ".means_)")
        print("covars")
        exec("print(model_" + emo + ".covars_)")
        print("transmat")
        exec("print(model_" + emo + ".transmat_)")
        exec("np.savetxt('../results/MMI/" + emo + "_means_5_components.csv',model_" + emo + ".means_ ,delimiter=',', header='_AU01_r, _AU02_r, _AU04_r, _AU05_r, _AU06_r, _AU07_r, _AU09_r, _AU10_r, _AU12_r, _AU14_r, _AU15_r, _AU17_r, _AU20_r, _AU23_r, _AU25_r, _AU26_r')")


    #print(model.startprob_) 初期状態決定確率 
    #print(model.means_) 各状態からの出力の平均
    #print(model.covars_) 共分散
    #print(model.transmat_) 状態間遷移確率　遷移行列

    '''
    sample_surprise = np.genfromtxt("../data/MMI_custum/Sessions/117/S001-117-AU/S001-117-AU.csv", delimiter=",", skip_header=1)
    print(model_surprise.predict(sample_surprise))
    print(model_surprise.score(sample_surprise))
    '''
    '''
    print("Evaluating score using sample with each emotionHMM ...")

    emotion_eval_list = []

    for emo in emotions:
        exec("score_" + emo + " = np.exp(model_" + emo + ".score(sample_surprise))")
        exec("print(\"" + emo + ":\" + str(score_" + emo + "))")
        exec("emotion_eval_list.append(score_" + emo + ")")

    emotion_eval_list.sort()
    emotion_eval_list.reverse()
    best = emotion_eval_list[0]


    for emo in emotions:
        exec("""if best == score_""" + emo + """:
            print(\"""" + emo + """ is the most likely emotion\")
        """)

    for emo in emotions:
        exec("print(\"the number of " + emo + " sequences :\" + str(count_" + emo + "))")

    '''


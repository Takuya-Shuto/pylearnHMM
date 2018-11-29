import numpy as np
from hmmlearn import hmm
import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.externals import joblib

print("generating HMM with sequences in each emotion ...")

# 対象にするクラスをまとめておく
emotions = ["disgust", "anger", "sadness", "happy" , "fear", "surprise"]

# 各クラスに対する3状態のleft-to-right HMMを定義
# 各クラスの平均値の初期値は，適当なサンプルから少数第一位までを用いた
for emo in emotions:
    exec("model_" + emo + " = hmm.GaussianHMM(n_components=5, n_iter=100, covariance_type=\"diag\", init_params=\"cm\", params=\"cmt\")")
    exec("model_" + emo + ".startprob_ = np.array([1.0, 0.0, 0.0, 0.0, 0.0])")
    exec("""model_""" + emo + """.transmat_ = np.array([[0.5, 0.5, 0.0, 0.0, 0.0],
                                                [0.0, 0.5, 0.5, 0.0, 0.0],
                                                [0.0, 0.0, 0.5, 0.5, 0.0],
                                                [0.0, 0.0, 0.0, 0.5, 0.5],
                                                [0.0, 0.0, 0.0, 0.0, 1.0]])""")
    exec("model_" + emo + ".n_features = 17")
    exec("""model_""" + emo + """.means_ = ([[0.00, 0.00, 0.00, 0.00, 0,00, 0,00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                     [0.25, 0.25, 0.25, 0.25, 0,25, 0,25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                                                     [0.50, 0.50, 0.50, 0.50, 0,50, 0,50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
                                                     [0.25, 0.25, 0.25, 0.25, 0,25, 0,25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                                                     [0.00, 0.00, 0.00, 0.00, 0,00, 0,00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])""")
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
                        if "1" in emotion_MMI:
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
                            list_anger.append(sequence_anger)
                            count_anger+=1
                        if "2" in emotion_MMI:
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
                            list_disgust.append(sequence_disgust)
                            count_disgust+=1
                        if "3" in emotion_MMI:
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
                            list_fear.append(sequence_fear)
                            count_fear+=1
                        if "4" in emotion_MMI:
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
                            list_happy.append(sequence_happy)
                            count_happy+=1
                        if "5" in emotion_MMI:
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
                            list_sadness.append(sequence_sadness)
                            count_sadness+=1
                        if "6" in emotion_MMI:
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
                            list_surprise.append(sequence_surprise)
                            count_surprise+=1

# 各クラスのデータリストを連結する
for emo in emotions:
    exec("list_" + emo + " = list(map(lambda x: x / 5.0, list_" + emo + "))")
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
    exec("model_" + emo + ".fit(np.asarray(X_" + emo + "), np.asarray(lengths_" + emo + "))")
    print("startprob")
    exec("print(model_" + emo + ".startprob_)")
    print("means")
    exec("print(model_" + emo + ".means_)")
    print("covars")
    exec("print(model_" + emo + ".covars_)")
    print("transmat")
    exec("print(model_" + emo + ".transmat_)")
    exec("np.savetxt('../results/MMI/" + emo + "_means_5_components.csv',model_" + emo + ".means_ ,delimiter=',', header='_AU01_r, _AU02_r, _AU04_r, _AU05_r, _AU06_r, _AU07_r, _AU09_r, _AU10_r, _AU12_r, _AU14_r, _AU15_r, _AU17_r, _AU20_r, _AU23_r, _AU25_r, _AU26_r, _AU45_r')")


#print(model.startprob_) 初期状態決定確率 
#print(model.means_) 各状態からの出力の平均
#print(model.covars_) 共分散
#print(model.transmat_) 状態間遷移確率　遷移行列

sample_anger = [[0.00, 0.41, 0.00, 0.00, 0.56, 0.57, 0.11, 0.00, 0.89, 0.00, 0.00, 0.00, 0.70, 0.23, 0.00, 0.03, 0.47],
                 [0.00, 0.38, 0.00, 0.00, 0.56, 0.47, 0.07, 0.00, 0.87, 0.02, 0.00, 0.14, 0.77, 0.27, 0.00, 0.17, 0.62],
                 [0.00, 0.35, 0.00, 0.00, 0.56, 0.48, 0.05, 0.00, 0.86, 0.02, 0.00, 0.15, 0.78, 0.29, 0.00, 0.20, 0.76],
                 [0.00, 0.27, 0.00, 0.00, 0.55, 0.57, 0.09, 0.00, 0.83, 0.10, 0.00, 0.20, 0.89, 0.35, 0.00, 0.26, 0.85],
                 [0.00, 0.26, 0.00, 0.00, 0.55, 0.69, 0.13, 0.00, 0.87, 0.19, 0.00, 0.28, 0.90, 0.37, 0.00, 0.25, 0.89],
                 [0.00, 0.15, 0.00, 0.00, 0.59, 0.73, 0.10, 0.00, 0.92, 0.26, 0.00, 0.48, 0.93, 0.43, 0.00, 0.34, 0.89],
                 [0.00, 0.11, 0.01, 0.00, 0.67, 0.88, 0.05, 0.00, 0.98, 0.30, 0.00, 0.71, 0.92, 0.41, 0.00, 0.22, 0.94],
                 [0.00, 0.03, 0.09, 0.00, 0.78, 1.00, 0.01, 0.02, 1.07, 0.36, 0.03, 0.99, 0.95, 0.42, 0.00, 0.17, 1.00],
                 [0.00, 0.03, 0.26, 0.00, 0.85, 1.07, 0.09, 0.08, 1.09, 0.49, 0.10, 1.31, 1.04, 0.44, 0.00, 0.06, 1.08],
                 [0.00, 0.00, 0.47, 0.00, 0.86, 0.91, 0.21, 0.16, 1.07, 0.51, 0.20, 1.63, 1.08, 0.57, 0.00, 0.03, 1.09],
                 [0.00, 0.00, 0.67, 0.00, 0.85, 0.78, 0.37, 0.25, 0.99, 0.49, 0.27, 2.01, 1.13, 0.78, 0.00, 0.00, 1.08]]

sample_disgust = [[0.40, 0.42, 0.22, 0.00, 0.20, 0.47, 1.12, 0.96, 0.14, 0.00, 1.05, 1.09, 0.43, 0.37, 0.52, 0.88, 0.98],
                  [0.19, 0.21, 0.19, 0.00, 0.21, 0.60, 1.24, 1.02, 0.13, 0.00, 1.22, 1.15, 0.49, 0.46, 0.47, 0.67, 0.88],
                  [0.06, 0.07, 0.33, 0.00, 0.35, 0.78, 1.54, 1.09, 0.08, 0.00, 1.29, 1.17, 0.45, 0.44, 0.46, 0.57, 0.82],
                  [0.05, 0.06, 0.52, 0.00, 0.62, 1.21, 2.00, 1.26, 0.04, 0.00, 1.18, 1.12, 0.54, 0.45, 0.47, 0.45, 0.84],
                  [0.00, 0.00, 0.88, 0.00, 0.94, 1.59, 2.55, 1.40, 0.00, 0.00, 1.14, 1.18, 0.82, 0.44, 0.51, 0.45, 0.87],
                  [0.00, 0.00, 1.04, 0.00, 1.24, 1.98, 2.98, 1.59, 0.00, 0.00, 1.05, 1.23, 1.18, 0.42, 0.55, 0.42, 0.95],
                  [0.00, 0.00, 1.21, 0.00, 1.44, 2.26, 3.36, 1.70, 0.00, 0.00, 1.07, 1.37, 1.37, 0.35, 0.61, 0.35, 0.99],
                  [0.00, 0.00, 1.19, 0.00, 1.58, 2.47, 3.50, 1.88, 0.02, 0.00, 1.04, 1.38, 1.43, 0.29, 0.65, 0.30, 1.00],
                  [0.00, 0.00, 1.28, 0.00, 1.60, 2.68, 3.63, 1.88, 0.03, 0.00, 1.09, 1.43, 1.29, 0.36, 0.68, 0.36, 1.01],
                  [0.00, 0.00, 1.35, 0.00, 1.64, 2.81, 3.75, 1.78, 0.03, 0.00, 1.06, 1.36, 1.23, 0.39, 0.70, 0.38, 1.07],
                  [0.00, 0.00, 1.42, 0.00, 1.65, 2.95, 3.96, 1.60, 0.00, 0.00, 0.99, 1.33, 1.25, 0.44, 0.71, 0.10, 1.18]]

sample_surprise = np.genfromtxt("../data/MMI_custum/Sessions/117/S001-117-AU/S001-117-AU.csv", delimiter=",", skip_header=1)
print(model_surprise.predict(sample_surprise))
print(model_surprise.score(sample_surprise))

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


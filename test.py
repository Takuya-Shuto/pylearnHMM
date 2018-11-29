# -*- encoding: utf-8 -*-
import numpy as numpy
from hmmlearn import hmm
import os
import xml.etree.ElementTree as ET

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
                        print(emotion_MMI)
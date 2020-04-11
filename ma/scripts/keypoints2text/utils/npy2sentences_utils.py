"""npy2sentences_utils.py: create the link from the npy files to the sentences files"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


class NpyToSentence:

    def __init__(self, path_to_numpy_file, path_to_csv):
        self.path_to_numpy_file = Path(path_to_numpy_file)
        self.path_to_csv = Path(path_to_csv)
        old = np.load
        np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

    def main(self):
        self.keypoints2sentence()

    def keypoints2sentence(self):
        """ load from .npy file """
        kp_files = np.load(self.path_to_numpy_file).item()
        df_kp = pd.DataFrame(kp_files.keys(), columns=["keypoints"])
        kp2sentence = []

        d = {'keypoints': [], 'text': []}
        with open(self.path_to_csv) as f:
            for line in f:
                d['keypoints'].append(line.split(" ")[0])
                d['text'].append(" ".join(line.split()[1:]))
        df_text = pd.DataFrame(d)

        speaker = []
        for kp in df_kp["keypoints"]:
            vid_speaker = kp[:11] + kp[11:].split('-')[0]
            speaker.append(vid_speaker)

            for idx in range(len(df_text['keypoints'])):
                if vid_speaker in df_text['keypoints'][idx]:
                    kp2sentence.append([kp, df_text['text'][idx]])
                    break
        df_kp_text_train = pd.DataFrame(kp2sentence, columns=["keypoints", "text"])
        df_kp_text_train.to_csv(self.path_to_csv.name + "_2npy.txt", index=False)


if __name__ == '__main__':
    # file with sentences
    if len(sys.argv) > 1:
        path_to_numpy_file = sys.argv[1]
    else:
        print("Set path to npy file")
        sys.exit()

    # sentences file
    if len(sys.argv) > 2:
        path_to_csv = sys.argv[2]
    else:
        print("Set path to transformed file containing sentences")
        sys.exit()

    npy = NpyToSentence(path_to_numpy_file, path_to_csv)
    npy.main()

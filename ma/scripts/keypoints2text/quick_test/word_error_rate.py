from jiwer import wer
import numpy as np

ground_truth = "WIND SCHWACH WEHEN ZEIGEN-BILDSCHIRM"
hypothesis = "__ON__ __ON__ WEHEN"

error = wer(ground_truth, hypothesis)
print(error)

res = []

with open('log_slr_2.txt', 'r') as f:
    for line in f:
        # print(line)
        if len(line) > 2:
            print(line)
            print(line[0])
            print(line[2])
            print("\n\n")
            res.append(wer(line[0], line[2]))
        else:
            res.append(0)
print(res)
print(np.mean(res))
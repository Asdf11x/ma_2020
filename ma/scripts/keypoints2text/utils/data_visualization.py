"""data_visualization.py: Used to check the data and visualize it

"""
from collections import Counter
from pathlib import Path
import ast


def delete_identifier(file_path_source, file_path_target):
    """
    delete identifier from a text file containing the sentences, the file needs to contain transformed (cleaned) text
    because here its not dealt with punctiuation etc.

    -fZc293MpJk_2 the aileron is the control surface in the wing that is controlled
    by lateral movement right and left of the stick
    ->
    the aileron is the control surface in the wing that is controlled
    by lateral movement right and left of the stick

    modification: -fZc293MpJk_2 is missing in the second sentence

    :param file_path_source:
    :param file_path_target:
    :return:
    """
    file_content = []
    with open(file_path_source, 'r') as f:
        for line in f:
            file_content.append(line.split()[1:])

    with open(file_path_target, "w+") as myfile:
        for line in file_content:
            myfile.write(" ".join(line))
            myfile.write("\n")


def save_frquency(target):
    """
    save frquency of cleaned text without identifiers
    e.g.
    the aileron is the control surface in the wing that is controlled by lateral movement right and left of the stick
    ->
    {'the': 4, 'is': 2, 'aileron': 1, 'control': 1, 'surface': 1, 'in': 1, 'wing': 1, 'that': 1, 'controlled': 1, ...}
    :param target:
    :return:
    """
    target = Path(target)
    words = []
    with open(target, 'r') as f:
        for line in f:
            words.extend(line.split())
    counts = dict(Counter(words))
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}

    with open(target.parents[0] / str(target.stem + "_freq.txt"), 'w+') as w:
        w.write(str(counts))

    print(counts)


def compare_frequency(train_frq_path, val_frq_path, test_frq_path):
    """
    Compare the frquencies and show various things of the sentences
    :param train_frq_path:
    :param val_frq_path:
    :param test_frq_path:
    :return:
    """
    train_frq_path = Path(train_frq_path)
    val_frq_path = Path(val_frq_path)
    test_frq_path = Path(test_frq_path)

    with open(train_frq_path, "r") as file:
        contents = file.read()
        train_frq = ast.literal_eval(contents)

    with open(val_frq_path, "r") as file:
        contents = file.read()
        val_frq = ast.literal_eval(contents)

    with open(test_frq_path, "r") as file:
        contents = file.read()
        test_frq = ast.literal_eval(contents)

    print(f"unique words: train {len(train_frq)}, val {len(val_frq)}, test {len(test_frq)}")

    in_val_not_in_train = {k: v for k, v in val_frq.items() if k not in train_frq}
    print(f"in_val_not_in_train amount {len(in_val_not_in_train)}:")
    print(in_val_not_in_train)

    in_test_not_in_train = {k: v for k, v in test_frq.items() if k not in train_frq}
    print(f"in_test_not_in_train amount {len(in_test_not_in_train)}:")
    print(in_test_not_in_train)

    # print(train_frq)
    # print(val_frq)
    # print(test_frq)


# source: file containing identifiers
# target: file without identifiers
source = r"C:\Users\Asdf\Downloads\How2Sign_samples\text\2_transformed\how2sign.val.id_transformed.txt"
target = r"C:\Users\Asdf\Downloads\How2Sign_samples\text\5_visualization\how2sign.val.id_transformed.txt"

# delete_identifier(source, target)
# save_frquency(target)

# train, val, test Counter dictionary file, containing word frequencies
train_frq_path = r"C:\Users\Asdf\Downloads\How2Sign_samples\text\5_visualization\how2sign.train.id_transformed_freq.txt"
val_frq_path = r"C:\Users\Asdf\Downloads\How2Sign_samples\text\5_visualization\how2sign.val.id_transformed_freq.txt"
test_frq_path = r"C:\Users\Asdf\Downloads\How2Sign_samples\text\5_visualization\how2sign.test.id_transformed_freq.txt"

compare_frequency(train_frq_path, val_frq_path, test_frq_path)

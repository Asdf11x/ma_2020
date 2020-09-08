from pathlib import Path


def testy():
    a = [0.84, 5.21, 5.50, 12.05, 12.49, 15.16, 15.58, 18.13, 18.30, 21.0]
    multiplied_list = [element * 1.33 for element in a]
    multiplied_list_prec = [element * (4 / 3) for element in a]
    multiplied_list_prec_rnd = [round(element * (4 / 3), 3) for element in a]
    print(a)
    print(multiplied_list)
    print(multiplied_list_prec)
    print(multiplied_list_prec_rnd)


def extend():
    """
    recalculate timestamps and extend them by (4/3) (33%) or from 0.75 to 1 length
    :return:
    """

    path = Path(
        r"C:\Eigene_Programme\Git-Data\Own_Repositories\ma_2020\ma\scripts\keypoints2text\quick_test\recut\segments_train")

    save = []

    with open(path, 'r', encoding='utf-8') as infile:
        for line in infile:
            temp_line = line.split()
            temp_line[2] = round(float(temp_line[2]) * (4 / 3), 2)
            temp_line[3] = round(float(temp_line[3]) * (4 / 3), 2)
            save.append(temp_line)

    print(save)

    with open(str(path) + "_recut", 'w+', encoding='utf-8') as infile:

        for l, el in enumerate(save):
            string = ' '.join(map(str, el))
            for item in string:
                infile.write(item)
            infile.write('\n')


def test_dictionary():
    test_dick = {"hi": 00, "hola": 78263, "asdasd": 897987, "ajsdoa": 87896}
    print(test_dick.keys())




    # with open('contents.txt', 'a') as f:
    #     for element in test_dick.keys():
    #         f.write(element)
    #         f.write("\n")

test_dictionary()

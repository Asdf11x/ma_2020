# 03.02.2020 - File changer to rename files

output = ""

f1 = open("test_video_list.txt", "r")
f2 = open("video_segments.txt", "r")
for line1 in f1:
    for line2 in f2:
        if line1[:11] == line2[:11]:
            print("SAME-----------------------------------------")
        else:
            print(line1 + line2)
        break
f1.close()
f2.close()
"""
with open('test_video_list.txt', 'r') as file1:
    with open('video_segments.txt', 'r') as file2:
        for line in file1:
            print(line[:11])
            for check_line in file2:
                print(check_line[:11])
                if line[:11] == check_line[:11]:
                    line = line.split()
                    check_line = check_line.split()
                    print(line)

        same = set(file1).intersection(file2)

same.discard('\n')

with open('some_output_file.txt', 'w') as file_out:
    for line in same:
        file_out.write(line)
"""

print("finish")
# 03.02.2020 - File changer to rename files

counter = 0

file1 = open("test_video_list.txt", "r")
file2 = open("test_segments.txt", "r")
file3 = open("file3.txt", "w")
file1.seek(0, 0)
file2.seek(0, 0)
list1 = file1.readlines()
list2 = file2.readlines()
for i in list1:
    for j in list2:
        if i[:11] == j[:11]:

            counter += 1

            # take name from i and remaing information from j
            j_split = j.split()

            file3.write(str(j_split[0]) + ".mp4" + " " + str(i) + ".mp4" + " " + str(j_split[2]) + " " + str(j_split[3]) + "\n")

print("Found %s entries. Finish " %counter)
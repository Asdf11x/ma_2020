# 03.02.2020 - File changer to rename files

output = ""

with open('test_video_list.txt', 'r') as file1:
    with open('test_segments', 'r') as file2:
        for line in file1:
            for check_line in file2:
                if line[:11] == check_line[:11]:
                    line = line.split()
                    check_line = check_line.split()
                    print(line[0])

        same = set(file1).intersection(file2)

same.discard('\n')

with open('some_output_file.txt', 'w') as file_out:
    for line in same:
        file_out.write(line)

print("finish")
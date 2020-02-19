# so you will need a script that will give this command as an output:
# ffmpeg -i NAME_OF_THE_FILE(second columm) -ss START(3rd columm) -to END(4th columm) -async 1 NAME_OF_THE_SEGMENT(1st columm)

# an example of this command would be
# ffmpeg -i fzcsY2gm7t0-8-rgb_front.mp4 -ss 00.9 -to 3.09 -async 1 cut/test_sec3.mp4

# 03.02.2020 - File changer to rename files

counter = 0
file_type = "train"
file1 = open("rename_%s_video_segments.txt" %file_type, "r")
file2 = open("ffmpeg_%s.txt" %file_type, "w")

file1.seek(0, 0)

list1 = file1.readlines()

for i in list1:
    i_split = i.split()

    # ffmpeg -i NAME_OF_THE_FILE(second columm) -ss START(3rd columm) -to END(4th columm) -async 1 NAME_OF_THE_SEGMENT(1st columm)
    file2.write("ffmpeg -i %s -ss %s -to %s -async %s \n" %(str(i_split[1]), str(i_split[2]), str(i_split[3]), str(i_split[0])))

file1.close()
file2.close()
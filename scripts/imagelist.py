import os
import shutil


f = open("insta360/videos/insta3601693/insta3601693.txt", "r")
list = f.readlines()
new_insta = []
for i in list:
    #delete '\n' from the end of the line
    i = i.replace("\n", "")
    new_insta.append(i)
print(new_insta)

img_list = os.listdir("insta360/videos/insta3601693/mav0/cam0/data/")
print(img_list)
for i in range(len(img_list)):
    img = img_list[i]
    file_oldname = os.path.join("insta360/videos/insta3601693/mav0/cam0/data", img)
    file_newname_newfile = os.path.join("insta360/videos/insta3601693/mav0/cam0/data", "{}.png".format(new_insta[i]))

    newFileName=shutil.move(file_oldname, file_newname_newfile)

    print ("The renamed file has the name:",file_newname_newfile)

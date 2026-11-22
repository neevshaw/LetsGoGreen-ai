import os
import shutil

# One-off script used to merge the aluminum_soda_cans and food_cans classes into metal_cans
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
move_from = "aluminum_soda_cans"
move_to = "food_cans"
rename = "metal_cans"

for folder in ["train", "test", "val"]:
    os.chdir(path + "\\" + folder + "\\" + move_from)
    i = os.listdir()
    for image in i:
        shutil.move(image, path + "\\" + folder + "\\"+ move_to)
    os.chdir(path)
    shutil.rmtree(path + "\\" + folder + "\\" + move_from)
    os.rename(path + "\\" + folder + "\\" + move_to, path + "\\" + folder + "\\" + rename)

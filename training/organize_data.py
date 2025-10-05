import os
import shutil

path = r'C:\Users\neevs\OneDrive - Davidson Academy\Desktop\Summer 2024 AI\Deep Learning Specialization\Recycling CAC\images'
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


    
        
            


        

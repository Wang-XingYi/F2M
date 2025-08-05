import itertools
import os
import random
import cv2

import numpy as np


def generate_unique_combinations(lists, path_list):
    # Set to store unique combinations
    unique_combinations = set()

    # Read all images from each body part and generate combinations
    for i, img_list in enumerate(lists):
        for img in img_list:
            # Create a combination: fix the image from the i-th body part and randomly select from the others
            repeat_num=0
            while True:
                combination = []
                for j, lst in enumerate(lists):
                    if j == i:
                        # Fix the current body part image
                        combination.append(img)
                    else:
                        # Randomly select an image from other parts, or use 'None' if the list is empty
                        combination.append(random.choice(lst) if lst else 'None')

                combination_tuple = tuple(combination)
                print(combination_tuple)
                print(path_list[i])

                # Check if the combination already exists, if not, add it
                if combination_tuple not in unique_combinations:
                    unique_combinations.add(combination_tuple)
                    break # Successfully generated a unique combination
                else:
                    repeat_num+=1
                if repeat_num<2:
                    unique_combinations.add(combination_tuple)
                    break



    unique_combinations_list = list(unique_combinations)
    return unique_combinations_list

def datasetToTxt(dir,file,wildcard,recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    filename=[]
    # i=1
    for first_file in files:
        print(first_file)
        seconde_file_path = os.path.join(dir, first_file)
        seconde_files = os.listdir(seconde_file_path)
        for seconde_file in seconde_files:
            third_file_path = os.path.join(seconde_file_path, seconde_file)
            third_files = os.listdir(third_file_path)
            files_list=[]
            path_list=[]
            path = first_file + "/" + seconde_file + "/"
            for third_file in third_files:
                files_list.append(os.listdir(os.path.join(third_file_path,third_file)))
                path_list.append(path+third_file+"/")

            combinations_list=generate_unique_combinations(files_list,path_list)
            item = None
            for combination in combinations_list:
                for i in range(len(combination)):
                    if i == 0:
                        item = path_list[i] + combination[i] + " "
                    else:
                        item += path_list[i] + combination[i] + " "
                microscope=0
                if combination[-2]!="None":
                    microscope+=1
                if combination[-1]!="None":
                    microscope+=1
                item += str(microscope) + "\n"
                filename.append(item)
            # print("OK")


            print(seconde_file)


    np.random.shuffle(filename)
    for i in filename:
        file.write(i)
    print("Completed")
def Train():
  dir=r'./Train'
  outfile="Train_List.txt"
  wildcard = ".jpg .txt .exe .dll .lib .bmp"

  file = open(outfile,"w")
  if not file:
    print ("cannot open the file %s for writing" % outfile)

  datasetToTxt(dir,file,wildcard, 1)

  file.close()

def Val():
    dir = r'./Val'
    outfile = "Val_List.txt"
    wildcard = ".jpg .txt .exe .dll .lib .bmp"

    file = open(outfile, "w")
    if not file:
        print("cannot open the file %s for writing" % outfile)


    datasetToTxt(dir, file, wildcard, 1)

    file.close()
def Test():
    dir = r'./Test'
    outfile = "Test_List.txt"
    wildcard = ".jpg .txt .exe .dll .lib .bmp"

    file = open(outfile, "w")
    if not file:
        print("cannot open the file %s for writing" % outfile)

    datasetToTxt(dir, file, wildcard, 1)

    file.close()

if __name__ == '__main__':
    random.seed(42)
    Test()
    Train()
    Val()


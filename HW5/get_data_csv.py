import numpy as np

import random
import csv
import os


DATA_PATH="./hw5_data/train/"
TEST_PATH="./hw5_data/test/"


label_dict={
    "Bedroom":0,
    "Coast":1,
    "Forest":2,
    "Highway":3,
    "Industrial":4,
    "InsideCity":5,
    "Kitchen":6,
    "LivingRoom":7,
    "Mountain":8,
    "Office":9,
    "OpenCountry":10,
    "Store":11,
    "Street":12,
    "Suburb":13,
    "TallBuilding":14
}


def main():
    # parse train
    class_folders=os.listdir(DATA_PATH)
    data=[]
    for class_folder in class_folders:
        class_path=os.path.join(DATA_PATH,class_folder)
        imgs=os.listdir(class_path)
        for img in imgs:
            img_path=os.path.join(class_path,img)
            label=label_dict[class_folder]
            pair=[]
            pair.append(img_path)
            pair.append(label)
            data.append(pair)

    random.shuffle(data)
    num_of_data=len(data)
    num_of_eval=int(num_of_data*0.3)
    num_of_train=num_of_data-num_of_eval
    train_data=data[:num_of_train]
    eval_data=data[num_of_train:]

    header=["image_path","label"]
    with open("./hw5_data/train.csv", 'w', encoding='UTF8', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_data)

    with open("./hw5_data/eval.csv", 'w', encoding='UTF8', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(header)
        writer.writerows(eval_data)
    
    # parse test
    class_folders=os.listdir(TEST_PATH)
    data=[]
    for class_folder in class_folders:
        class_path=os.path.join(TEST_PATH,class_folder)
        imgs=os.listdir(class_path)
        for img in imgs:
            img_path=os.path.join(class_path,img)
            label=label_dict[class_folder]
            pair=[]
            pair.append(img_path)
            pair.append(label)
            data.append(pair)
    
    with open("hw5_data/test.csv", 'w', encoding='UTF8', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
        
    



    



if __name__=="__main__":
    main()
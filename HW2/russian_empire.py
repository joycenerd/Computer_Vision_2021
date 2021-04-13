import glob


DATA_PATH="./hw2_data/task3_colorizing"

for image_path in glob.glob(DATA_PATH+"/*"):
    print(image_path)

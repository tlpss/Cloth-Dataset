import os
import json

if __name__ == "__main__":
    root_folder = "/home/tlips/Onedrive_UGent/A-PHD/SyntheticClothJournal/Cloth-Dataset/Dataset/train/location_1/boxershorts"

    # get all the json files
    json_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            # if file.endswith(".json"):
            #     json_dict = json.load(open(os.path.join(root, file)))
            #     json_dict["cloth_id"] += 1
            #     json.dump(json_dict, open(os.path.join(root, file), "w"))
            print(file)
            idx = file.find("2023")
            print(file[idx:])
            os.rename(os.path.join(root, file), os.path.join(root, file[idx:]))
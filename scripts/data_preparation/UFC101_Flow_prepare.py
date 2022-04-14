import numpy as np
from PIL import Image
from tqdm import tqdm

import os, os.path

import logging


from utils import get_classes, prepare_base_folder, prepare_classes_folders, prepare_video_folders
from utils import merge_u_v_into_image

path = "../../datasets/ucf101_tvl1_flow/tvl1_flow/"


def get_number_frames_in_clip(route, name):
    #print(f"La longitud es {len(os.listdir(route+'u/'+name))}")
    #print(f"La ruta es {route+'u/'+name}")
    #input("Pauss")
    return len(os.listdir(route+"u/"+name))

def move_images(route,destination, class_name, split):
    """
        destination = "../../datasets/HMDB51_Flow/"
        split : (split \in {train, test, val} , names of the files to move)
    """

    split_name, split_files = split

    for name in (sbar := tqdm(split_files, leave=False)):
        sbar.set_description("Processing %s" % name[:20])
        prepare_video_folders(destination , class_name,name, split_name)

        for frame in range(1, 1 + get_number_frames_in_clip(route, name)):
            #print(f"Estamos en el frame {frame}")
            #input("Pausa")
            merged_img = merge_u_v_into_image(route, name + f"/frame{frame:06d}.jpg")
            frame_out = frame-1
            merged_img.save(destination+ split_name+"/"+class_name+"/"+name +  f"/{frame_out:06d}.jpg")


def get_class_split(split_dir, class_name, split ="1"):
    """
        Gets class name and split and returns three lists for train test and val.

        Split_dir: Path to the folder of split classes (../datasets/UFC101/splits/)
    """

    train = []
    test = []

    train_split_path = split_dir + "trainlist0" + split + ".txt"
    
    with open(train_split_path, "r") as file:

        for line in file:
            if class_name in line:
                video_name = line.split()[0].split("/")[1].split(".")[0]
                train += [video_name]
            
    test_split_path = split_dir + "testlist0" + split + ".txt"
    
    with open(test_split_path, "r") as file:
        class_name_f =  "_" + line + "_"
        for class_name_f in file:
            if class_name in line:
                video_name = line.split("/")[1].split(".")[0]
                test += [video_name]

    return ('val', []), ('train', train), ('test', test)



def prepare(split_id = 1, route = "", destination = ""):
    """
        split_id: Identifier of the split selected (1, 2 or  3)
        route: Where the flow
        destination: Where the flow will go
    """
    
    assert route != ""

    prepare_base_folder(destination)
    

    classes = get_classes("../../dataloaders/labels/ucf101_labels.txt")
    
    for class_i in (pbar := tqdm(classes)):
        pbar.set_description("Processing %s" % class_i)
        logging.info('Preparing images from class ' + str(class_i))
        prepare_classes_folders(destination, class_i)

        splits = get_class_split( "../../datasets/UFC101/ucfTrainTestlist/",class_i, str(split_id))

        for split_list in splits:
            move_images(route, destination, class_i, split_list)
        

if __name__ == '__main__':
    prepare(
        split_id="1",
        route="../../datasets/ucf101_tvl1_flow/tvl1_flow/",
        destination="../../datasets/UCF101_Flow/")
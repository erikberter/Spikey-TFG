import numpy as np
from PIL import Image
from tqdm import tqdm

import os, os.path

import logging

path = "../datasets/hmdb51_tvl1_flow/tvl1_flow/"

def get_classes(labels_file = "../dataloaders/labels/hmdb_labels.txt", skip_first_n = 0):
    
    labels = []

    with open(labels_file, "r") as file:
        for i, line in enumerate(file):
            if i < skip_first_n:
                continue
            labels += [line.split()[1]]

    return labels


def get_class_split(split_dir, class_name, split ="1"):
    """
        Gets class name and split and returns three lists for train test and val.

        Split_dir: Path to the folder of split classes (../datasets/HMBD51/splits/)
    """
    assert split_dir is not None and class_name is not None
    assert split_dir != "" and class_name != ""
    assert split_dir[-1] == "/"
    assert split in ["1", "2", "3"]

    names = [[], [], []] # Order of list val, train, test

    split_path = split_dir + class_name + "_test_split" + split + ".txt"
    with open(split_path, "r") as file:
        

        for line in file:
            line_s = line.split() # Get only the name
            sector = int(line_s[1])
            name = line_s[0].split(".")[0] # Delete the avi part

            names[sector] += [name]

    return ('val', names[0]), ('train', names[1]), ('test', names[2])


def merge_u_v_into_image(path, file):
    """
        Merges the u and v component from grayscales to a image with 3 channels.

        path = ../datasets/hmdb51_tvl1_flow/tvl1_flow/
        file = #20_Rhythm_clap_u_nm_np1_fr_goo_0/frame000055.jpg
    """

    u_img = Image.open(path + "u/" + file).convert('RGB') 
    v_img = Image.open(path + "v/" + file).convert('RGB') 

    r_u, _, _ = u_img.split()
    _, g_v, b = v_img.split()

    b = b.point(lambda i: 127) # Average color is 127

    merged =  Image.merge('RGB', (r_u, g_v, b))

    resized = merged.resize((171,128), Image.ANTIALIAS)

    return resized

def get_number_frames_in_clip(route, name):
    #print(f"La longitud es {len(os.listdir(route+'u/'+name))}")
    #print(f"La ruta es {route+'u/'+name}")
    #input("Pauss")
    return len(os.listdir(route+"u/"+name))


def move_images(route,destination, class_name, split):
    """
        destination = "../datasets/HMDB51_Flow/"
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


def prepare_base_folder(destination ):
    for i in ["train", "test", "val"]:
        if not os.path.exists(destination + i):
            os.mkdir(destination + i)

    pass

def prepare_classes_folders(destination, class_name):
    for i in ["train", "test", "val"]:
        if not os.path.exists(destination + i+ "/" + class_name):
            os.mkdir(destination + i + "/" + class_name)

    pass

def prepare_video_folders(destination, class_name, video_name, split):
    if not os.path.exists(destination + split+ "/" + class_name + "/" + video_name):
        os.mkdir(destination + split + "/" + class_name + "/" + video_name)

def prepare(split_id = 1, route = "", destination = ""):
    """
        split_id: Identifier of the split selected (1, 2 or  3)
        route: Where the flow
        destination: Where the flow will go
    """
    
    assert route != ""

    prepare_base_folder(destination)
    

    classes = get_classes()
    
    for class_i in (pbar := tqdm(classes)):
        pbar.set_description("Processing %s" % class_i)
        logging.info('Preparing images from class ' + str(class_i))
        prepare_classes_folders(destination, class_i)

        splits = get_class_split( "../datasets/HMBD51/splits/",class_i, str(split_id))

        for split_list in splits:
            move_images(route, destination, class_i, split_list)
        

if __name__ == '__main__':
    prepare(
        split_id="1",
        route="../datasets/hmdb51_tvl1_flow/tvl1_flow/",
        destination="../datasets/HMDB51_Flow/")
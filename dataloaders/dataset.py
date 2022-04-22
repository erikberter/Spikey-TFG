import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path



def get_ufc101(dataset, file, video_files):
    train_file, val_file, test_file = Path.train_test_val_split_files(dataset)
    train = set(line.strip().split("/")[1].split(" ")[0] for line in open(train_file) if file == line.strip().split("/")[0])
    if val_file is not None:
        val = set(line.strip().split("/")[1].split(" ")[0] for line in open(val_file) if file == line.strip().split("/")[0])
    else:
        val = None
    test = set(line.strip().split("/")[1] for line in open(test_file) if file == line.strip().split("/")[0])
    return train, val, test

def get_kth(dataset,file, video_files):
    train_file, val_file, test_file = Path.train_test_val_split_files(dataset)
    train = set(video_file for line in open(train_file)  for video_file in video_files if line.strip() in video_file)
                    
    if val_file is not None:
        val = set(video_file for line in open(val_file) for video_file in video_files if line.strip() in video_file)
    else:
        val = None
    test = set(video_file for line in open(test_file) for video_file in video_files if line.strip() in video_file)
    return train, val, test


def get_hmdb51(dataset, file, video_files):

    split_n = 1

    
    split_file_base, _, _ = Path.train_test_val_split_files(dataset)
    file_r = os.path.join(split_file_base, file+'_test_split' + str(split_n) + '.txt')

    train_files = [train_name.split()[0] for train_name in open(file_r) if train_name.split()[1] == '1' ]
    val_files = [train_name.split()[0] for train_name in open(file_r) if train_name.split()[1] == '0' ]
    test_files = [train_name.split()[0] for train_name in open(file_r) if train_name.split()[1] == '2' ]

    
    return train_files, val_files, test_files


class SplitExtractor:
    def __init__(self):
        pass

    def get_split_train_test_files_names(self, dataset, label, video_files):
        
        if dataset == 'ufc101':
            return get_ufc101(dataset, label, video_files)
        elif dataset == 'kth':
            return get_kth(dataset, label, video_files)


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='ucf101', split='train', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        self.dataset_name = dataset

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if (not self.check_preprocess()) or preprocess:
            input("Dataset will be preprocess. Are you okey with it?")
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            if dataset == 'ucf101' or dataset == 'kth' or dataset =='hmdb51':
                self.preprocess(custom_ttv = True)
            else:
                self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if not os.path.exists('dataloaders/labels/' + dataset+ '_labels.txt'):
            with open('dataloaders/labels/' + dataset + '_labels.txt', 'w') as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id+1) + ' ' + label + '\n')


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        #if self.split == 'test':
            # Perform data augmentation
        #    buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or (np.shape(image)[1] != 171 and np.shape(image)[1] != 172):
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self, custom_ttv = False):
        """
            Parameters:
                custom_ttv (bool): True if the dataset has a custom train-test-val ordering
                has_val (bool): False if the dataset doesn't have validation
        """
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            if not custom_ttv:
                train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
                train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)
            else:
                train_file, val_file, test_file = Path.train_test_val_split_files(self.dataset_name)
                
                if 'ufc' in self.dataset_name: 
                    train = set(line.strip().split("/")[1].split(" ")[0] for line in open(train_file) if file == line.strip().split("/")[0])
                    if val_file is not None:
                        val = set(line.strip().split("/")[1].split(" ")[0] for line in open(val_file) if file == line.strip().split("/")[0])
                    else:
                        val = None
                    test = set(line.strip().split("/")[1] for line in open(test_file) if file == line.strip().split("/")[0])
                elif 'kth' == self.dataset_name:
                    train = set(video_file for line in open(train_file)  for video_file in video_files if line.strip() in video_file)
                    
                    if val_file is not None:
                        val = set(video_file for line in open(val_file) for video_file in video_files if line.strip() in video_file)
                    else:
                        val = None
                    test = set(video_file for line in open(test_file) for video_file in video_files if line.strip() in video_file)
                else:
                    train, val, test = get_hmdb51(self.dataset_name, file, video_files)


            train_dir = os.path.join(self.output_dir, 'train', file)
            test_dir = os.path.join(self.output_dir, 'test', file)
            if val is not None:
                val_dir = os.path.join(self.output_dir, 'val', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if val is not None and not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)
            
            if val is not None:
                for video in val:
                    self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)[:, :171, :] # TODO cambiar, lo he puesto porque en vez de 171, en una he puesto 172..
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering


        while buffer.shape[0] < clip_len + 1:
            buffer = np.repeat(buffer, 2, axis=0)
            
        time_index = np.random.randint(buffer.shape[0] - clip_len)


        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer





if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='ucf101', split='test', clip_len=8, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break
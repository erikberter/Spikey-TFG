class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = 'datasets\\UFC101\\UCF101\\UCF-101'

            # Save preprocess data into output_dir
            output_dir = 'datasets\\UFC101\\UCF101\\Processed'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = 'datasets\\HMBD51\\HMBD51'

            output_dir = 'datasets\\HMBD51\\Processed'

            return root_dir, output_dir
        elif database == 'hmdb51_2':
            # folder that contains class labels
            root_dir = 'datasets\\HMBD51\\HMBD51'

            output_dir = 'datasets\\HMBD51_2\\Processed'

            return root_dir, output_dir
        elif database == 'hmdb51_3':
            # folder that contains class labels
            root_dir = 'datasets\\HMBD51\\HMBD51'

            output_dir = 'datasets\\HMBD51_3\\Processed'

            return root_dir, output_dir
        elif database == 'kth':
            # folder that contains class labels
            root_dir = 'datasets\\KTH\\KTH'

            output_dir = 'datasets\\KTH\\Processed'

            return root_dir, output_dir
        elif database == 'hmdb51_flow':
            # folder that contains class labels
            root_dir = 'datasets\\HMBD51\\HMBD51'

            output_dir = 'datasets\\HMDB51_Flow'

            return root_dir, output_dir
        elif database == 'ucf101_flow':
            # folder that contains class labels
            root_dir = 'datasets\\UFC101\\UCF101\\UCF-101'

            output_dir = 'datasets\\UCF101_Flow'

            return root_dir, output_dir
        elif database == 'kth_rbg_diff':
            # folder that contains class labels
            root_dir = 'datasets\\KTH\\Processed'

            output_dir = 'datasets\\KTH_RGB_Diff'

            return root_dir, output_dir
        elif database == 'hmdb51_rbg_diff':
            # folder that contains class labels
            root_dir = 'datasets\\HMBD51\\Processed'

            output_dir = 'datasets\\HMDB51_RGB_Diff'

            return root_dir, output_dir
        elif database == 'kith_small':
            # folder that contains class labels
            root_dir = 'datasets\\KTH\\Processed'

            output_dir = 'datasets\\KTH_Small'

            return root_dir, output_dir
        elif database == 'kith_rgb_small':
            # folder that contains class labels
            root_dir = 'datasets\\KTH\\Processed'

            output_dir = 'datasets\\KTH_Small_RGB_Diff'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def train_test_val_split_files(database):
        if database == 'ucf101':
            train_file = 'datasets\\UFC101\\ucfTrainTestlist\\trainlist01.txt'
            test_file = 'datasets\\UFC101\\ucfTrainTestlist\\testlist01.txt'


            return train_file, None, test_file
        elif database == 'kth':
            
            train_file = 'datasets\\KTH\\train_labels.txt'
            val_file = 'datasets\\KTH\\val_labels.txt'
            test_file = 'datasets\\KTH\\test_labels.txt'

            return train_file, val_file, test_file
        elif 'hmdb51' in database:
            return 'datasets\\HMBD51\\splits', None, None
        else:
            print('Database splits {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return 'C:\\Users\\whiwho\\Documents\\GitHub\\SNN_TFG\\models\\C3D\\c3d-pretrained.pth'
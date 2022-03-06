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
            root_dir = '/Path/to/hmdb-51'

            output_dir = '/path/to/VAR/hmdb51'

            return root_dir, output_dir
        elif database == 'kth':
            # folder that contains class labels
            root_dir = 'datasets\\KTH\\KTH'

            output_dir = 'datasets\\KTH\\Processed'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def train_test_val_split_dir(database):
        if database == 'ucf101':
            ttv_dir = 'datasets\\UFC101\\ucfTrainTestlist'

            return ttv_dir

    @staticmethod
    def model_dir():
        return 'C:\\Users\\whiwho\\Documents\\GitHub\\SNN_TFG\\models\\C3D\\c3d-pretrained.pth'
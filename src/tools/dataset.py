import zipfile
import tempfile
import pathlib

import pandas as pd
import numpy as np


shl_dataset_label_order = [
    'Null',
    'Still',
    'Walking',
    'Run',
    'Bike',
    'Car',
    'Bus',
    'Train',
    'Subway',
]


class SHLDataset:
    def __init__(
        self,
        acc_x, acc_y, acc_z,
        acc_mag,
        mag_x, mag_y, mag_z,
        mag_mag,
        gyr_x, gyr_y, gyr_z,
        gyr_mag,
        labels
    ):
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.acc_z = acc_z
        self.acc_mag = acc_mag

        self.mag_x = mag_x
        self.mag_y = mag_y
        self.mag_z = mag_z
        self.mag_mag = mag_mag

        self.gyr_x = gyr_x
        self.gyr_y = gyr_y
        self.gyr_z = gyr_z
        self.gyr_mag = gyr_mag
        
        self.labels = labels

    def concat_inplace(self, other):
        self.acc_x = np.concatenate((self.acc_x, other.acc_x), axis=0)
        self.acc_y = np.concatenate((self.acc_y, other.acc_y), axis=0)
        self.acc_z = np.concatenate((self.acc_z, other.acc_z), axis=0)
        self.acc_mag = np.concatenate((self.acc_mag, other.acc_mag), axis=0)

        self.mag_x = np.concatenate((self.mag_x, other.mag_x), axis=0)
        self.mag_y = np.concatenate((self.mag_y, other.mag_y), axis=0)
        self.mag_z = np.concatenate((self.mag_z, other.mag_z), axis=0)
        self.mag_mag = np.concatenate((self.mag_mag, other.mag_mag), axis=0)

        self.gyr_x = np.concatenate((self.gyr_x, other.gyr_x), axis=0)
        self.gyr_y = np.concatenate((self.gyr_y, other.gyr_y), axis=0)
        self.gyr_z = np.concatenate((self.gyr_z, other.gyr_z), axis=0)
        self.gyr_mag = np.concatenate((self.gyr_mag, other.gyr_mag), axis=0)

        self.labels = np.concatenate((self.labels, other.labels), axis=0)


def load_shl_dataset(dataset_dir: pathlib.Path, nrows=None):
    acc_x = np.nan_to_num(pd.read_csv(dataset_dir / 'Acc_x.txt', header=None, sep=' ', nrows=nrows).to_numpy())
    print('Acc_x Import Done')
    acc_y = np.nan_to_num(pd.read_csv(dataset_dir / 'Acc_y.txt', header=None, sep=' ', nrows=nrows).to_numpy())
    print('Acc_y Import Done')
    acc_z = np.nan_to_num(pd.read_csv(dataset_dir / 'Acc_z.txt', header=None, sep=' ', nrows=nrows).to_numpy())
    print('Acc_z Import Done')
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    print('Acc_mag Import Done')

    mag_x = np.nan_to_num(pd.read_csv(dataset_dir / 'Mag_x.txt', header=None, sep=' ', nrows=nrows).to_numpy())
    print('Mag_x Import Done')
    mag_y = np.nan_to_num(pd.read_csv(dataset_dir / 'Mag_y.txt', header=None, sep=' ', nrows=nrows).to_numpy())
    print('Mag_y Import Done')
    mag_z = np.nan_to_num(pd.read_csv(dataset_dir / 'Mag_z.txt', header=None, sep=' ', nrows=nrows).to_numpy())
    print('Mag_z Import Done')
    mag_mag = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
    print('Mag_mag Import Done')

    gyr_x = np.nan_to_num(pd.read_csv(dataset_dir / 'Gyr_x.txt', header=None, sep=' ', nrows=nrows).to_numpy())
    print('Gyr_x Import Done')
    gyr_y = np.nan_to_num(pd.read_csv(dataset_dir / 'Gyr_y.txt', header=None, sep=' ', nrows=nrows).to_numpy())
    print('Gyr_y Import Done')
    gyr_z = np.nan_to_num(pd.read_csv(dataset_dir / 'Gyr_z.txt', header=None, sep=' ', nrows=nrows).to_numpy())
    print('Gyr_z Import Done')
    gyr_mag = np.sqrt(gyr_x**2 + gyr_y**2 + gyr_z**2)
    print('Gyr_mag Import Done')

    labels = np.nan_to_num(pd.read_csv(dataset_dir / 'Label.txt', header=None, sep=' ', nrows=nrows).mode(axis=1).to_numpy().flatten())
    print('Labels Import Done')

    return SHLDataset(
        acc_x, acc_y, acc_z,
        acc_mag,
        mag_x, mag_y, mag_z, 
        mag_mag,
        gyr_x, gyr_y, gyr_z, 
        gyr_mag,
        labels
    )


def load_zipped_shl_dataset(zip_dir: pathlib.Path, tqdm=None, nrows=None, subdir_in_zip='train'):
    with tempfile.TemporaryDirectory() as unzip_dir:
        with zipfile.ZipFile(zip_dir, 'r') as zip_ref:
            if tqdm:
                for member in tqdm(zip_ref.infolist(), desc=f'Extracting {zip_dir}'):
                    zip_ref.extract(member, unzip_dir)
            else:
                zip_ref.extractall(unzip_dir)

        train_dir = pathlib.Path(unzip_dir) / subdir_in_zip
        sub_dirs = [x for x in train_dir.iterdir() if train_dir.is_dir()]

        result_dataset = None
        for sub_dir in sub_dirs:
            sub_dataset = load_shl_dataset(train_dir / sub_dir, nrows=nrows)
            if result_dataset is None:
                result_dataset = sub_dataset
            else:
                result_dataset.concat_inplace(sub_dataset)
            del sub_dataset
        return result_dataset
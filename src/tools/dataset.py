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


def load_shl_dataset(dataset_dir: pathlib.Path):
    acc_x = pd.read_csv(dataset_dir / 'Acc_x.txt', header=None, sep=' ').to_numpy()
    print('Acc_x Import Done')
    acc_y = pd.read_csv(dataset_dir / 'Acc_y.txt', header=None, sep=' ').to_numpy()
    print('Acc_y Import Done')
    acc_z = pd.read_csv(dataset_dir / 'Acc_z.txt', header=None, sep=' ').to_numpy()
    print('Acc_z Import Done')
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    print('Acc_mag Import Done')

    mag_x = pd.read_csv(dataset_dir / 'Mag_x.txt', header=None, sep=' ').to_numpy()
    print('Mag_x Import Done')
    mag_y = pd.read_csv(dataset_dir / 'Mag_y.txt', header=None, sep=' ').to_numpy()
    print('Mag_y Import Done')
    mag_z = pd.read_csv(dataset_dir / 'Mag_z.txt', header=None, sep=' ').to_numpy()
    print('Mag_z Import Done')
    mag_mag = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
    print('Mag_mag Import Done')

    gyr_x = pd.read_csv(dataset_dir / 'Gyr_x.txt', header=None, sep=' ').to_numpy()
    print('Gyr_x Import Done')
    gyr_y = pd.read_csv(dataset_dir / 'Gyr_y.txt', header=None, sep=' ').to_numpy()
    print('Gyr_y Import Done')
    gyr_z = pd.read_csv(dataset_dir / 'Gyr_z.txt', header=None, sep=' ').to_numpy()
    print('Gyr_z Import Done')
    gyr_mag = np.sqrt(gyr_x**2 + gyr_y**2 + gyr_z**2)
    print('Gyr_mag Import Done')

    labels = pd.read_csv(dataset_dir / 'Label.txt', header=None, sep=' ').mode(axis=1).to_numpy().flatten()
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


def load_zipped_shl_dataset(zip_dir: pathlib.Path, tqdm=None):
    with tempfile.TemporaryDirectory() as unzip_dir:
        with zipfile.ZipFile(zip_dir, 'r') as zip_ref:
            if tqdm:
                for member in tqdm(zip_ref.infolist(), desc=f'Extracting {zip_dir}'):
                    zip_ref.extract(member, unzip_dir)
            else:
                zip_ref.extractall(unzip_dir)

        train_dir = pathlib.Path(unzip_dir) / 'train'
        sub_dirs = [x for x in train_dir.iterdir() if train_dir.is_dir()]
        assert len(sub_dirs) == 1
        return load_shl_dataset(train_dir / sub_dirs[0])
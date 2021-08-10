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

shl_dataset_X_attributes = [
    'acc_x', 'acc_y', 'acc_z',
    'mag_x', 'mag_y', 'mag_z',
    'gyr_x', 'gyr_y', 'gyr_z',
    'gra_x', 'gra_y', 'gra_z',
    'lacc_x', 'lacc_y', 'lacc_z',
    'ori_x', 'ori_y', 'ori_z', 'ori_w',
]

shl_dataset_y_attributes = ['labels']

shl_dataset_attributes = shl_dataset_X_attributes + shl_dataset_y_attributes

shl_dataset_files = [
    'Acc_x.txt', 'Acc_y.txt', 'Acc_z.txt',
    'Mag_x.txt', 'Mag_y.txt', 'Mag_z.txt',
    'Gyr_x.txt', 'Gyr_y.txt', 'Gyr_z.txt',
    'Gra_x.txt', 'Gra_y.txt', 'Gra_z.txt',
    'LAcc_x.txt', 'LAcc_y.txt', 'LAcc_z.txt',
    'Ori_x.txt', 'Ori_y.txt', 'Ori_z.txt', 'Ori_w.txt',
    'Label.txt'
]


class SHLDataset:
    def __init__(self):
        pass

    def concat_inplace(self, other):
        for attribute in shl_dataset_attributes:
            setattr(self, attribute, np.concatenate((
                getattr(self, attribute),
                getattr(other, attribute)
            ), axis=0))


def load_shl_dataset(dataset_dir: pathlib.Path, tqdm=None, nrows=None, dtype=np.float16):
    dataset = SHLDataset()
    if tqdm is None:
        tqdm = lambda x, desc: x # passthrough
    for attribute, filename in tqdm(
        list(zip(shl_dataset_attributes, shl_dataset_files)),
        desc=f'Loading dataset subfiles'
    ):
        df = pd.read_csv(dataset_dir / filename, header=None, sep=' ', nrows=nrows, dtype=dtype)
        np_arr = np.nan_to_num(df.to_numpy())
        setattr(dataset, attribute, np_arr)
    return dataset


def load_zipped_shl_dataset(zip_dir: pathlib.Path, tqdm=None, nrows=None, subdir_in_zip='train', dtype=np.float16):
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
            sub_dataset = load_shl_dataset(train_dir / sub_dir, tqdm=tqdm, nrows=nrows, dtype=dtype)
            if result_dataset is None:
                result_dataset = sub_dataset
            else:
                result_dataset.concat_inplace(sub_dataset)
                del sub_dataset
        return result_dataset
from os.path import join

import numpy as np
import pandas
import scipy.signal as sps
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset

from autodsp.__config__ import MSFT_AEC_DIR

"""
This file contains the dataloader and prep scripts for the Microsoft AEC dataset.
"""

def read_microsoft_aec_data(synthetic_dataset_filepath):
    """Read the Microsoft AEC Challenge metadata (synthetic dataset) 

    Arguments:
        synthetic_dataset_filepath {str} -- Absolute filepath of MS AEC Challenge filepath

    Returns:
        data {dict} -- Metadata of all data
        data_train {dict} -- Metadata of all training data
        data_test {dict} -- Metadata of all testing data
        is_farend_nonlinear {dict} -- Metadata of all fileids with linear and nonlinear echos
        is_farend_noisy {dict} -- Metadata of all fileids with noisy or quiet farend
        is_nearend_noisy {dict} -- Metadata of all fileids with noisy or quiet nearend
    """

    synthetic_csv = join(synthetic_dataset_filepath, 'meta.csv')

    data = {}
    data_train = {}
    data_test = {}

    is_farend_nonlinear = {'train': {1: [], 0: []}, 'test': {1: [], 0: []}}
    is_farend_noisy = {'train': {1: [], 0: []}, 'test': {1: [], 0: []}}
    is_nearend_noisy = {'train': {1: [], 0: []}, 'test': {1: [], 0: []}}

    df = pandas.read_csv(synthetic_csv)
    for _, row in df.iterrows():

        fileid = row['fileid']
        data[fileid] = {}
        data[fileid]['ser'] = row['ser']
        data[fileid]['is_farend_nonlinear'] = row['is_farend_nonlinear']
        data[fileid]['is_farend_noisy'] = row['is_farend_noisy']
        data[fileid]['is_nearend_noisy'] = row['is_nearend_noisy']
        data[fileid]['split'] = row['split']
        data[fileid]['fileid'] = row['fileid']
        data[fileid]['nearend_scale'] = row['nearend_scale']

        # Add the absolute path of the echo, far-end, near-end, and clean near-end
        data[fileid]['echo_path'] = join(synthetic_dataset_filepath,
                                         'echo_signal/echo_fileid_' + str(fileid) + '.wav')
        data[fileid]['farend_speech_path'] = join(synthetic_dataset_filepath,
                                                  'farend_speech/farend_speech_fileid_' + str(fileid) + '.wav')
        data[fileid]['nearend_mic_path'] = join(synthetic_dataset_filepath,
                                                'nearend_mic_signal/nearend_mic_fileid_' + str(fileid) + '.wav')
        data[fileid]['nearend_speech_path'] = join(synthetic_dataset_filepath,
                                                   'nearend_speech/nearend_speech_fileid_' + str(fileid) + '.wav')

        # Dictionary train/test -> nonlinear -> file id
        is_farend_nonlinear[row['split']
                            ][row['is_farend_nonlinear']].append(fileid)
        is_farend_noisy[row['split']][row['is_farend_noisy']].append(fileid)
        is_nearend_noisy[row['split']][row['is_nearend_noisy']].append(fileid)

        if row['split'] == 'test':
            data_test[fileid] = data[fileid]
        else:
            data_train[fileid] = data[fileid]

    return data, data_train, data_test, is_farend_nonlinear, is_farend_noisy, is_nearend_noisy


class MSFTAECDset(Dataset):
    """ Microsoft dataset module used for all experiments in this paper.
    """
    def __init__(self, mode, data_dir, sr, out_sec, include_noisy, include_nonlinear,
                 just_echo, is_double_len):
        data, _, _, is_farend_nonlinear, is_farend_noisy, is_nearend_noisy = read_microsoft_aec_data(
            data_dir)
        if mode == 'val' or mode == 'train':
            sample_ids = self._get_ids(is_farend_nonlinear,
                                       is_farend_noisy,
                                       is_nearend_noisy,
                                       include_noisy,
                                       include_nonlinear,
                                       'train')
            np.random.seed(0)
            np.random.shuffle(sample_ids)
        else:
            sample_ids = self._get_ids(is_farend_nonlinear,
                                       is_farend_noisy,
                                       is_nearend_noisy,
                                       include_noisy,
                                       include_nonlinear,
                                       'test')

        train_val_split = int(.95 * len(sample_ids))
        if mode == 'train':
            sample_ids = sample_ids[:train_val_split]
        elif mode == 'val':
            sample_ids = sample_ids[train_val_split:]

        self.sample_ids = sample_ids
        self.sr = sr
        self.out_size = out_sec * sr
        self.mode = mode
        self.data = data
        self.sr = sr
        self.just_echo = just_echo
        self.proj_vals = {}
        self.is_double_len = is_double_len

    def _get_ids(self, is_farend_nonlinear, is_farend_noisy, is_nearend_noisy, include_noisy, include_nonlinear, mode):
        nearend_noisy = set(is_nearend_noisy[mode][0])
        farend_noisy = set(is_farend_noisy[mode][0])
        if include_noisy:
            nearend_noisy = nearend_noisy.union(set(is_nearend_noisy[mode][1]))
            farend_noisy = farend_noisy.union(set(is_farend_noisy[mode][1]))

        nonlinear = set(is_farend_nonlinear[mode][0])
        if include_nonlinear:
            nonlinear = nonlinear.union(set(is_farend_nonlinear[mode][1]))

        return np.array(list(nearend_noisy.intersection(farend_noisy).intersection(nonlinear)))

    def __len__(self):
        if self.mode == 'train':
            return 99999999
        else:
            return len(self.sample_ids)

    def _load_resample(self, fname):
        old_sr, y = wavfile.read(fname)
        new_len = int(len(y) * float(self.sr) / old_sr)
        y = sps.resample(y, new_len).reshape(-1)
        return y

    def _pad(self, x):
        if len(x) < self.out_size:
            return np.pad(x, (0, self.out_size - len(x)))
        else:
            return x

    def get_single_item(self, idx):
        idx = self.sample_ids[idx % len(self.sample_ids)]

        # get path
        echo_path = self.data[idx]['echo_path']
        nearend_mic_path = self.data[idx]['nearend_mic_path']
        nearend_speech_path = self.data[idx]['nearend_speech_path']
        farend_path = self.data[idx]['farend_speech_path']

        # load and resample
        echo = self._load_resample(echo_path)
        nearend_mic = self._load_resample(nearend_mic_path)
        nearend_speech = self._load_resample(
            nearend_speech_path) * self.data[idx]['nearend_scale']
        farend = self._load_resample(farend_path)

        # normalize
        echo_max_abs = np.max(np.abs(echo)) + 1e-10
        echo /= echo_max_abs
        nearend_mic /= echo_max_abs
        nearend_speech /= echo_max_abs
        farend /= echo_max_abs

        # pad back up after resampling
        echo = self._pad(echo)
        nearend_mic = self._pad(nearend_mic)
        nearend_speech = self._pad(nearend_speech)
        farend = self._pad(farend)

        meta = [self.data[idx]['ser'],
                self.data[idx]['is_farend_noisy'] or self.data[idx]['is_nearend_noisy'],
                self.data[idx]['is_farend_nonlinear']]

        if idx not in self.proj_vals:
            self.proj_vals[idx] = np.dot(
                nearend_mic, nearend_speech) / np.dot(nearend_speech, nearend_speech)

        proj_mic_speech = nearend_speech * self.proj_vals[idx]
        nearend_no_speech = nearend_mic[:, None] - proj_mic_speech[:, None]
        return farend[:, None], nearend_no_speech, echo, meta

    def __getitem__(self, idx):
        if self.is_double_len:
            f1, n1, e1, meta1 = self.get_single_item(idx)
            f2, n2, e2, meta2 = self.get_single_item(idx + 1)
            meta_or = [0, 0, 0]
            meta_or[0] = (meta1[0] + meta2[0]) / 2
            meta_or[1] = meta1[1] or meta2[1]
            meta_or[2] = meta1[2] or meta2[2]
            return np.vstack((f1, f2)), np.vstack((n1, n2)), np.hstack((e1, e2)), meta_or
        else:
            return self.get_single_item(idx)


def np_collate_fn(batch):
    batch_t = list(zip(*batch))
    farends = np.array(batch_t[0])
    nearends = np.array(batch_t[1])
    clean = np.array(batch_t[2])
    meta = np.array(batch_t[3])
    return farends, nearends, clean, meta


def get_msft_data_gen(mode='train',
                      data_dir=MSFT_AEC_DIR,
                      sr=8000,
                      out_sec=10,
                      batch_sz=1,
                      num_workers=1,
                      include_noisy=True,
                      include_nonlinear=True,
                      is_iter=True,
                      just_echo=True,
                      is_double_len=False):
    """ Function called from the configuration file to create a dataloader function/object.
    """

    dset = MSFTAECDset(mode=mode,
                       data_dir=data_dir,
                       sr=sr,
                       out_sec=out_sec,
                       include_noisy=include_noisy,
                       include_nonlinear=include_nonlinear,
                       just_echo=just_echo,
                       is_double_len=is_double_len)

    shuffle = mode == 'train'
    dloader = DataLoader(dset, batch_size=batch_sz, shuffle=shuffle,
                         num_workers=num_workers, collate_fn=np_collate_fn, drop_last=True)

    if is_iter:
        d_iter = iter(dloader)

        def data_gen():
            return next(d_iter)
        return data_gen
    else:
        return dloader


if __name__ == "__main__":
    func = get_msft_data_gen(batch_sz=1)
    farend, nearend, _ = func()
    print(farend.shape, nearend.shape, type(farend), type(nearend))

    func = get_msft_data_gen(batch_sz=4)
    farend, nearend, _ = func()
    print(farend.shape, nearend.shape, type(farend), type(nearend))

    func = get_msft_data_gen(batch_sz=8)
    farend, nearend, _ = func()
    print(farend.shape, nearend.shape, type(farend), type(nearend))

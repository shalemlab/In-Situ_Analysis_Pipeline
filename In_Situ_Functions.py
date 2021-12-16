import warnings
# from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage
import skimage.feature
import skimage.segmentation
import skimage.registration
import skimage.filters
import pandas as pd
from csbdeep.utils import normalize
from ops.firesnake import Snake
import sys
import os
from glob import glob
# from scipy.optimize import curve_fit
from cellpose import models as cellpose_models
from os import listdir
from os.path import isfile, join, splitext
import nd2reader as nd2
from natsort import natsorted
warnings.filterwarnings('ignore')


def Manage_Save_Directories(list_subfolders, _save_path):
    path_check = _save_path
    if not (os.path.isdir(path_check)):
        os.mkdir(path_check)

    for i in range(len(list_subfolders)):
        path_check = os.path.join(_save_path, list_subfolders[i])
        if not (os.path.isdir(path_check)):
            os.mkdir(path_check)


def Import_ND2_by_Tile_and_Well(_n_tile, _n_well, _path_name):
    _data = np.empty([0])
    onlyfiles = [f for f in listdir(_path_name) if isfile(join(_path_name, f))]

    for i in range(len(onlyfiles)):

        if onlyfiles[i].endswith('.nd2'):

            n_tile = int(onlyfiles[i].split('_')[2])
            n_well = int(onlyfiles[i].split('_')[0].split('Well')[-1])

            if n_tile == _n_tile and n_well == _n_well:
                full_path = _path_name + '/' + onlyfiles[i]
                _data = np.array(nd2.ND2Reader(full_path), dtype=np.float64)
                only_name = onlyfiles[i].split('.')[0]
                # print('Imported tile', n_tile, 'of well', n_well)

    if _data.shape[0] == np.array(0, dtype=tuple):
        print('Image of tile ' + str(_n_tile) + ' of well ' + str(_n_well) + ' not found')

    return _data  # , only_name


def Assemble_Data_From_ND2(_n_tile, _n_well, _path_name, verbose=False):

    """
    def compare_char(a, b):
        assert len(a) == len(b), "nd2 files should have the same char length"

        _char_comp = np.zeros([len(a)], dtype = 1)

        for i in range(len(a)):
            if a[i] != b[i]:
                _char_comp[i] = 1

        return _char_comp
    """

    _data = np.empty([0])
    list_cycle_dirs = glob(_path_name + '/*')
    list_cycle_dirs = natsorted(list_cycle_dirs, key=lambda y: y.lower())

    _complete_data = np.empty([len(list_cycle_dirs), 0, 0, 0])

    for j, cycle_path in enumerate(list_cycle_dirs):
        files = [f for f in listdir(cycle_path) if isfile(join(cycle_path, f)) and join(cycle_path, f).endswith('.nd2')]

        for i in range(len(files)):

            n_tile = int(files[i].split('_')[2])
            n_well = int(files[i].split('_')[0].split('Well')[-1])

            if n_tile == _n_tile and n_well == _n_well:
                full_path = cycle_path + '/' + files[i]
                _data_cycle = np.array(nd2.ND2Reader(full_path), dtype=np.float64)
                if j == 0:
                    _complete_data = np.empty(np.concatenate(([len(list_cycle_dirs)], _data_cycle.shape)))
                    _complete_data[0] = _data_cycle
                else:
                    _complete_data[j] = _data_cycle

                if verbose:
                    print('Imported tile', n_tile, 'of well', n_well, cycle_path.split('/')[-1])




    return _complete_data


def Find_Peaks(data, verbose=False):
    aligned = Snake._align_SBS(data, method='DAPI')
    if verbose:
        print('Aligned')

    loged = Snake._transform_log(aligned, skip_index=0)
    if verbose:
        print('Loged')

    maxed = Snake._max_filter(loged, 3, remove_index=0)
    if verbose:
        print('Maxed')

    std = Snake._compute_std(loged, remove_index=0)
    if verbose:
        print('Std')

    peaks = Snake._find_peaks(std)
    if verbose:
        print('Peaks')

    return maxed, peaks


def plot_reads(cycle, calls_int, df_reads, nuclei, cells):
    marker_size = 5

    x = np.empty([0])
    y = np.empty([0])
    b = np.empty([0])
    for i in range(len(calls_int)):
        b_i = calls_int[i, cycle - 1]
        x_i = int(df_reads['j'].values[i])
        y_i = int(df_reads['i'].values[i])

        b = np.append(b, b_i)
        x = np.append(x, x_i)
        y = np.append(y, y_i)

    ind_A = np.where(b == 0)[0]
    ind_C = np.where(b == 1)[0]
    ind_G = np.where(b == 2)[0]
    ind_T = np.where(b == 3)[0]

    title_name = 'Cycle: ' + str(cycle)
    plt.title(title_name)

    plt.scatter(x[ind_A], y[ind_A], color='blue', s=marker_size, marker='.')
    plt.scatter(x[ind_C], y[ind_C], color='yellow', s=marker_size, marker='.')
    plt.scatter(x[ind_G], y[ind_G], color='red', s=marker_size, marker='.')
    plt.scatter(x[ind_T], y[ind_T], color='green', s=marker_size, marker='.')

    plt.imshow(2 * nuclei + cells, cmap='Greys')


def Call_Bases(cells, maxed, peaks, THRESHOLD_STD=500, verbose=False):

    cycles = list(range(1, maxed.shape[0] + 1))
    bases = ['G', 'T', 'A', 'C']
    CYCLE = 'cycle';
    CHANNEL = 'channel';
    POSITION_I = 'i';
    POSITION_J = 'j';
    INTENSITY = 'intensity';
    READ = 'read';
    CELL = 'cell';
    BARCODE = 'barcode';  # WELL='well'; TILE='tile'

    read_mask = (peaks > THRESHOLD_STD)
    values = maxed[:, :, read_mask].transpose([2, 0, 1])
    labels = cells[read_mask]
    positions = np.array(np.where(read_mask)).T

    index = (CYCLE, cycles), (CHANNEL, bases)
    names, levels = zip(*index)
    columns = pd.MultiIndex.from_product(levels, names=names)
    df = pd.DataFrame(values.reshape(values.shape[0], -1), columns=columns)

    df_positions = pd.DataFrame(positions, columns=[POSITION_I, POSITION_J])
    df_bases = (df.stack([CYCLE, CHANNEL])
                .reset_index()
                .rename(columns={0: INTENSITY, 'level_0': READ})
                .join(pd.Series(labels, name=CELL), on=READ)
                .join(df_positions, on=READ)
                .sort_values([CELL, READ, CYCLE])
                )

    cycles = len(set(df_bases['cycle']))
    channels = len(set(df_bases['channel']))
    # df_bases.sort_values([WELL, TILE, CELL, READ, CYCLE, CHANNEL])
    df_bases.sort_values([CELL, READ, CYCLE, CHANNEL])

    def dataframe_to_values(df, value='intensity'):
        cycles = df[CYCLE].value_counts()
        n_cycles = len(cycles)
        n_channels = len(df[CHANNEL].value_counts())
        x = np.array(df[value]).reshape(-1, n_cycles, n_channels)
        return x

    def transform_medians(X):
        arr = []
        for i in range(X.shape[1]):
            arr += [np.median(X[X.argmax(axis=1) == i], axis=0)]
        M = (np.array(arr)).T
        # M = M.T
        M = M / M.sum(axis=0)
        W = np.linalg.inv(M)
        Y = W.dot(X.T).T.astype(int)
        return Y, W

    X_ = dataframe_to_values(df_bases.query('cell > 0'))
    _, W = transform_medians(X_.reshape(-1, channels))
    X = dataframe_to_values(df_bases)
    Y = W.dot(X.reshape(-1, channels).T).T.astype(int)

    bases = sorted(set(df_bases[CHANNEL]))
    # df_reads = df_bases.drop_duplicates([WELL, TILE, READ]).copy()
    df_reads = df_bases.drop_duplicates([READ]).copy()
    Y_rearranged = Y.reshape(-1, cycles, channels)
    calls_int = Y_rearranged.argmax(axis=2)
    calls = np.array(list(bases))[calls_int]
    df_reads[BARCODE] = [''.join(x) for x in calls]

    Y_ = Y.reshape(-1, cycles, channels)
    Y_ = np.abs(np.sort(Y_, axis=-1).astype(float))
    Q = 1 - np.log(2 + Y_[..., -2]) / np.log(2 + Y_[..., -1])
    Q = (Q * 2).clip(0, 1)

    for i in range(len(Q[0])):
        df_reads['Q_%d' % i] = Q[:, i]

    df_reads = df_reads.assign(Qumin=lambda x: x.filter(regex='Q_\d+').min(axis=1))
    df_reads = df_reads.assign(Qumean=lambda x: x.filter(regex='Q_\d+').mean(axis=1))
    df_reads = df_reads.assign(Qumax=lambda x: x.filter(regex='Q_\d+').max(axis=1))

    df_reads = df_reads.drop([CYCLE, CHANNEL, INTENSITY], axis=1)

    if verbose:
        print('Barcodes Read')

    return df_reads


def Segment_Nuclei(_img, nuc_diameter=None, GPU=False):

    nuclei_model = cellpose_models.Cellpose(gpu=GPU, model_type='nuclei')
    # cyto_model = cellpose_models.Cellpose(gpu=False, model_type='cyto', torch=True)
    # cyto2_model = cellpose_models.Cellpose(gpu=False, model_type='cyto2', torch=True)

    _masks_nuc, _, _, _ = nuclei_model.eval(_img, diameter=nuc_diameter, channels=[[0, 0]])

    return _masks_nuc


def Segment_Cells(_data, _nuc, method='Cellpose', cells_diameter=None, GPU=False):

    if method == 'Cellpose':

        assert _data.ndim == 2, 'Input one channel for cell segmentation. Array should have 2 dimensions.'

        # nuclei_model = cellpose_models.Cellpose(model_type='nuclei')
        # cyto_model = cellpose_models.Cellpose(gpu=False, model_type='cyto', torch=True)
        cyto2_model = cellpose_models.Cellpose(gpu=GPU, model_type='cyto2', torch=True)

        h, w = _data.shape
        seg_data = np.zeros([h,w,3])
        seg_data[:, :, 0] = _data
        seg_data[:, :, 2] = _nuc

        _masks_cells, _, _, _ = cyto2_model.eval(_data, diameter=cells_diameter, channels=[[1, 3]])

    if method == 'Watershed':

        assert _data.ndim == 2 or _data.ndim == 3, 'Array should have 2 or 3 dimensions.'
        if _data.ndim == 2:
            _data = np.expand_dims(_data, axis=0)

        mask = np.mean(_data, axis=0)
        mask = normalize(mask, 0, 100, axis=(0, 1))
        mask = skimage.filters.rank.mean(mask, selem=skimage.morphology.disk(15))

        thresh = skimage.filters.threshold_otsu(mask)
        mask = mask > thresh

        distance = scipy.ndimage.distance_transform_cdt(_nuc == 0)
        _masks_cells = skimage.segmentation.watershed(distance, _nuc, mask=mask, watershed_line=True)

    return _masks_cells


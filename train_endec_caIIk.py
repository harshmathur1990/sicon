import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import h5py
from torch.autograd import Variable
import torchvision
import scipy.io as io
import scipy.ndimage as nd
from tqdm import tqdm
import model_encdec as model
import time
import shutil
# import database
import pandas as pd
import os
from pathlib import Path


quiet_nodes = [
    np.array([65, 95, 112, 127, 140]),
    np.array([140]),
    np.array([68, 97, 112, 128])
]

emission_nodes = [
    np.array([95, 112, 127, 140]),
    np.array([50, 95, 112, 127, 140]),
    np.array([68, 97, 112, 128])
]

generic_nodes = [
    np.array([95, 112, 119, 127, 132, 140]),
    np.array([50, 95, 112, 119, 127, 132, 140]),
    np.array([95, 112, 119, 127, 132, 140])
]

def get_nodes(nodename='emission'):
    if nodename == 'emission':
        return emission_nodes
    elif nodename == 'quiet':
        return quiet_nodes
    else:
        return generic_nodes


def get_fov1():

    input_profile_path = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/inversions/frame_0_21_x_662_712_y_708_758.nc'
    )

    finputprofiles = h5py.File(input_profile_path, 'r')

    output_atmos_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1/frame_0_21_x_662_712_y_708_758_cycle_1_t_4_vl_7_vt_4_atmos.nc')

    output_atmos_quiet_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_quiet_v1/quiet_frame_0_21_x_662_712_y_708_758_cycle_1_t_5_vl_1_vt_4_atmos.nc')

    output_atmos_reverse_shock_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_reverse_shock_v1/reverse_shock_frame_0_21_x_662_712_y_708_758_cycle_1_t_5_vl_5_vt_4_atmos.nc')

    output_atmos_other_emission_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_other_emission_v1/other_emission_frame_0_21_x_662_712_y_708_758_cycle_1_t_5_vl_5_vt_4_atmos.nc')

    output_atmos_failed_inversion_falc_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_failed_inversions_falc/failed_inversions_falc_frame_0_21_x_662_712_y_708_758_cycle_1_t_5_vl_1_vt_4_atmos.nc')

    output_atmos_failed_inversion_falc_2_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_failed_inversions_falc_2/failed_inversions_falc_2_frame_0_21_x_662_712_y_708_758_cycle_1_t_4_vl_1_vt_4_atmos.nc')

    reverse_shock_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/pixel_indices_reverse_shock.h5')

    quiet_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/pixel_indices_new.h5')

    other_emission_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/pixel_indices_other_emission.h5')

    failed_inversions_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/pixel_indices_failed_inversions.h5')

    failed_inversions_falc_2_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/pixel_indices_failed_inversions_falc_2.h5')

    ind = np.where(finputprofiles['profiles'][0, 0, 0, :, 0] != 0)[0]

    fquiet = h5py.File(quiet_pixel_file, 'r')

    a1, b1, c1 = fquiet['pixel_indices'][0:3]

    freverse = h5py.File(reverse_shock_pixel_file, 'r')

    d1, e1, g1 = freverse['pixel_indices'][0:3]

    fother = h5py.File(other_emission_pixel_file, 'r')

    h1, i1, j1 = fother['pixel_indices'][0:3]

    ffailed = h5py.File(failed_inversions_pixel_file, 'r')

    k1, l1, m1 = ffailed['pixel_indices'][0:3]

    ffailed_falc_2 = h5py.File(failed_inversions_falc_2_pixel_file, 'r')

    n1, o1, p1 = ffailed_falc_2['pixel_indices'][0:3]

    fout = h5py.File(output_atmos_filepath, 'r')

    fout_quiet = h5py.File(output_atmos_quiet_filepath, 'r')

    fout_reverse = h5py.File(output_atmos_reverse_shock_filepath, 'r')

    fout_other = h5py.File(output_atmos_other_emission_filepath, 'r')

    fout_failed_falc = h5py.File(
        output_atmos_failed_inversion_falc_filepath,
        'r'
    )

    fout_failed_falc_2 = h5py.File(
        output_atmos_failed_inversion_falc_2_filepath,
        'r'
    )

    all_temp = fout['temp'][()]

    all_vlos = fout['vlos'][()]

    all_vturb = fout['vturb'][()]

    all_pgas = fout['pgas'][()]

    all_temp[a1, b1, c1] = fout_quiet['temp'][0, 0]
    all_temp[d1, e1, g1] = fout_reverse['temp'][0, 0]
    all_temp[h1, i1, j1] = fout_other['temp'][0, 0]
    all_temp[k1, l1, m1] = fout_failed_falc['temp'][0, 0]
    all_temp[n1, o1, p1] = fout_failed_falc_2['temp'][0, 0]

    all_vlos[a1, b1, c1] = fout_quiet['vlos'][0, 0]
    all_vlos[d1, e1, g1] = fout_reverse['vlos'][0, 0]
    all_vlos[h1, i1, j1] = fout_other['vlos'][0, 0]
    all_vlos[k1, l1, m1] = fout_failed_falc['vlos'][0, 0]
    all_vlos[n1, o1, p1] = fout_failed_falc_2['vlos'][0, 0]

    all_vturb[a1, b1, c1] = fout_quiet['vturb'][0, 0]
    all_vturb[d1, e1, g1] = fout_reverse['vturb'][0, 0]
    all_vturb[h1, i1, j1] = fout_other['vturb'][0, 0]
    all_vturb[k1, l1, m1] = fout_failed_falc['vturb'][0, 0]
    all_vturb[n1, o1, p1] = fout_failed_falc_2['vturb'][0, 0]

    all_pgas[a1, b1, c1] = fout_quiet['pgas'][0, 0]
    all_pgas[d1, e1, g1] = fout_reverse['pgas'][0, 0]
    all_pgas[h1, i1, j1] = fout_other['pgas'][0, 0]
    all_pgas[k1, l1, m1] = fout_failed_falc['pgas'][0, 0]
    all_pgas[n1, o1, p1] = fout_failed_falc_2['pgas'][0, 0]

    return finputprofiles['profiles'][:, :, :, ind, 0], all_temp, all_vlos, all_vturb, all_pgas


def get_fov2():

    x = [770, 820]

    y = [338, 388]

    frames = [56, 77]

    input_profile_quiet = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/quiet_profiles_frame_56_77_x_770_820_y_338_388.nc'
    )

    input_profile_shock = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/shock_spicule_profiles_frame_56_77_x_770_820_y_338_388.nc'
    )

    input_profile_shock_78 = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/shock_78_frame_56_77_x_770_820_y_338_388.nc'
    )

    input_profile_reverse = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/reverse_shock_profiles_frame_56_77_x_770_820_y_338_388.nc'
    )

    input_profile_retry = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/retry_shock_spicule_frame_56_77_x_770_820_y_338_388.nc'
    )

    input_profile_other = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/other_emission_profiles_frame_56_77_x_770_820_y_338_388.nc'
    )

    output_atmos_quiet_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/quiet_profiles_frame_56_77_x_770_820_y_338_388_cycle_1_t_5_vl_1_vt_4_atmos.nc')

    output_atmos_shock_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/shock_spicule_profiles_frame_56_77_x_770_820_y_338_388_cycle_1_t_4_vl_5_vt_4_atmos.nc')

    output_atmos_shock_78_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/shock_78_frame_56_77_x_770_820_y_338_388_cycle_1_t_4_vl_5_vt_4_atmos.nc')

    output_atmos_reverse_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/reverse_shock_profiles_frame_56_77_x_770_820_y_338_388_cycle_1_t_4_vl_5_vt_4_atmos.nc')

    output_atmos_retry_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/retry_shock_spicule_frame_56_77_x_770_820_y_338_388_cycle_1_t_4_vl_5_vt_4_atmos.nc')

    output_atmos_other_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/other_emission_profiles_frame_56_77_x_770_820_y_338_388_cycle_1_t_5_vl_5_vt_4_atmos.nc')

    quiet_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/pixel_indices_quiet_profiles_frame_56_77_x_770_820_y_338_388.h5')

    shock_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/pixel_indices_shock_spicule_profiles_frame_56_77_x_770_820_y_338_388.h5')

    shock_78_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/pixel_indices_shock_78_frame_56_77_x_770_820_y_338_388.h5')

    reverse_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/pixel_indices_reverse_shock_profiles_frame_56_77_x_770_820_y_338_388.h5')

    retry_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/pixel_indices_retry_shock_spicule_frame_56_77_x_770_820_y_338_388.h5')

    other_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_second_fov/pixel_indices_other_emission_profiles_frame_56_77_x_770_820_y_338_388.h5')

    finputprofiles_quiet = h5py.File(input_profile_quiet, 'r')

    finputprofiles_shock = h5py.File(input_profile_shock, 'r')

    finputprofiles_shock_78 = h5py.File(input_profile_shock_78, 'r')

    finputprofiles_reverse = h5py.File(input_profile_reverse, 'r')

    finputprofiles_retry = h5py.File(input_profile_retry, 'r')

    finputprofiles_other = h5py.File(input_profile_other, 'r')

    ind = np.where(finputprofiles_quiet['profiles'][0, 0, 0, :, 0] != 0)[0]

    fout_atmos_quiet = h5py.File(output_atmos_quiet_filepath, 'r')

    fout_atmos_shock = h5py.File(output_atmos_shock_filepath, 'r')

    fout_atmos_shock_78 = h5py.File(output_atmos_shock_78_filepath, 'r')

    fout_atmos_reverse = h5py.File(output_atmos_reverse_filepath, 'r')

    fout_atmos_retry = h5py.File(
        output_atmos_retry_filepath,
        'r'
    )

    fout_atmos_other = h5py.File(
        output_atmos_other_filepath,
        'r'
    )

    fquiet = h5py.File(quiet_pixel_file, 'r')

    fshock = h5py.File(shock_pixel_file, 'r')

    fshock_78 = h5py.File(shock_78_pixel_file, 'r')

    freverse = h5py.File(reverse_pixel_file, 'r')

    fretry = h5py.File(retry_pixel_file, 'r')

    fother = h5py.File(other_pixel_file, 'r')

    a1, b1, c1 = fquiet['pixel_indices'][0:3]

    a2, b2, c2 = fshock['pixel_indices'][0:3]

    a3, b3, c3 = fshock_78['pixel_indices'][0:3]

    a4, b4, c4 = freverse['pixel_indices'][0:3]

    a5, b5, c5 = fretry['pixel_indices'][0:3]

    a6, b6, c6 = fother['pixel_indices'][0:3]

    all_profiles = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            30
        )
    )

    all_profiles[a1, b1, c1] = finputprofiles_quiet['profiles'][0, 0, :, ind, 0]
    all_profiles[a2, b2, c2] = finputprofiles_shock['profiles'][0, 0, :, ind, 0]
    all_profiles[a3, b3, c3] = finputprofiles_shock_78['profiles'][0, 0, :, ind, 0]
    all_profiles[a4, b4, c4] = finputprofiles_reverse['profiles'][0, 0, :, ind, 0]
    all_profiles[a5, b5, c5] = finputprofiles_retry['profiles'][0, 0, :, ind, 0]
    all_profiles[a6, b6, c6] = finputprofiles_other['profiles'][0, 0, :, ind, 0]

    all_temp = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    all_vlos = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    all_vturb = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    all_pgas = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    all_temp[a1, b1, c1] = fout_atmos_quiet['temp'][0, 0]
    all_temp[a2, b2, c2] = fout_atmos_shock['temp'][0, 0]
    all_temp[a3, b3, c3] = fout_atmos_shock_78['temp'][0, 0]
    all_temp[a4, b4, c4] = fout_atmos_reverse['temp'][0, 0]
    all_temp[a5, b5, c5] = fout_atmos_retry['temp'][0, 0]
    all_temp[a6, b6, c6] = fout_atmos_other['temp'][0, 0]

    all_vlos[a1, b1, c1] = fout_atmos_quiet['vlos'][0, 0]
    all_vlos[a2, b2, c2] = fout_atmos_shock['vlos'][0, 0]
    all_vlos[a3, b3, c3] = fout_atmos_shock_78['vlos'][0, 0]
    all_vlos[a4, b4, c4] = fout_atmos_reverse['vlos'][0, 0]
    all_vlos[a5, b5, c5] = fout_atmos_retry['vlos'][0, 0]
    all_vlos[a6, b6, c6] = fout_atmos_other['vlos'][0, 0]

    all_vturb[a1, b1, c1] = fout_atmos_quiet['vturb'][0, 0]
    all_vturb[a2, b2, c2] = fout_atmos_shock['vturb'][0, 0]
    all_vturb[a3, b3, c3] = fout_atmos_shock_78['vturb'][0, 0]
    all_vturb[a4, b4, c4] = fout_atmos_reverse['vturb'][0, 0]
    all_vturb[a5, b5, c5] = fout_atmos_retry['vturb'][0, 0]
    all_vturb[a6, b6, c6] = fout_atmos_other['vturb'][0, 0]

    all_pgas[a1, b1, c1] = fout_atmos_quiet['pgas'][0, 0]
    all_pgas[a2, b2, c2] = fout_atmos_shock['pgas'][0, 0]
    all_pgas[a3, b3, c3] = fout_atmos_shock_78['pgas'][0, 0]
    all_pgas[a4, b4, c4] = fout_atmos_reverse['pgas'][0, 0]
    all_pgas[a5, b5, c5] = fout_atmos_retry['pgas'][0, 0]
    all_pgas[a6, b6, c6] = fout_atmos_other['pgas'][0, 0]

    return all_profiles, all_temp, all_vlos, all_vturb, all_pgas


def get_fov3():

    x = [520, 570]

    y = [715, 765]

    frames = [0, 21]

    input_profile_quiet = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/quiet_profiles_frame_0_21_x_520_570_y_715_765.nc'
    )

    input_profile_shock = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/shock_spicule_profiles_frame_0_21_x_520_570_y_715_765.nc'
    )

    input_profile_shock_78 = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/shock_78_profiles_frame_0_21_x_520_570_y_715_765.nc'
    )

    input_profile_reverse = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/reverse_shock_profiles_frame_0_21_x_520_570_y_715_765.nc'
    )

    input_profile_retry = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/retry_shock_spicule_profiles_frame_0_21_x_520_570_y_715_765.nc'
    )

    input_profile_other = Path(
        '/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/other_emission_profiles_frame_0_21_x_520_570_y_715_765.nc'
    )

    output_atmos_quiet_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/quiet_profiles_frame_0_21_x_520_570_y_715_765_cycle_1_t_5_vl_1_vt_4_atmos.nc')

    output_atmos_shock_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/shock_spicule_profiles_frame_0_21_x_520_570_y_715_765_cycle_1_t_4_vl_5_vt_4_atmos.nc')

    output_atmos_shock_78_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/shock_78_profiles_frame_0_21_x_520_570_y_715_765_cycle_1_t_4_vl_5_vt_4_atmos.nc')

    output_atmos_reverse_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/reverse_shock_profiles_frame_0_21_x_520_570_y_715_765_cycle_1_t_5_vl_5_vt_4_atmos.nc')

    output_atmos_retry_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/retry_shock_spicule_profiles_frame_0_21_x_520_570_y_715_765_cycle_1_t_4_vl_5_vt_4_atmos.nc')

    output_atmos_other_filepath = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/other_emission_profiles_frame_0_21_x_520_570_y_715_765_cycle_1_t_5_vl_5_vt_4_atmos.nc')

    quiet_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/pixel_indices_quiet_profiles_frame_0_21_x_520_570_y_715_765.h5')

    shock_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/pixel_indices_shock_spicule_profiles_frame_0_21_x_520_570_y_715_765.h5')

    shock_78_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/pixel_indices_shock_78_frame_0_21_x_520_570_y_715_765.h5')

    reverse_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/pixel_indices_reverse_shock_profiles_frame_0_21_x_520_570_y_715_765.h5')

    retry_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/pixel_indices_retry_shock_spicule_frame_0_21_x_520_570_y_715_765.h5')

    other_pixel_file = Path('/home/harsh/OsloAnalysis/new_kmeans/inversions/plots_v1_third_fov/pixel_indices_other_emission_profiles_frame_0_21_x_520_570_y_715_765.h5')

    finputprofiles_quiet = h5py.File(input_profile_quiet, 'r')

    finputprofiles_shock = h5py.File(input_profile_shock, 'r')

    finputprofiles_shock_78 = h5py.File(input_profile_shock_78, 'r')

    finputprofiles_reverse = h5py.File(input_profile_reverse, 'r')

    finputprofiles_retry = h5py.File(input_profile_retry, 'r')

    finputprofiles_other = h5py.File(input_profile_other, 'r')

    ind = np.where(finputprofiles_quiet['profiles'][0, 0, 0, :, 0] != 0)[0]

    fout_atmos_quiet = h5py.File(output_atmos_quiet_filepath, 'r')

    fout_atmos_shock = h5py.File(output_atmos_shock_filepath, 'r')

    fout_atmos_shock_78 = h5py.File(output_atmos_shock_78_filepath, 'r')

    fout_atmos_reverse = h5py.File(output_atmos_reverse_filepath, 'r')

    fout_atmos_retry = h5py.File(
        output_atmos_retry_filepath,
        'r'
    )

    fout_atmos_other = h5py.File(
        output_atmos_other_filepath,
        'r'
    )

    fquiet = h5py.File(quiet_pixel_file, 'r')

    fshock = h5py.File(shock_pixel_file, 'r')

    fshock_78 = h5py.File(shock_78_pixel_file, 'r')

    freverse = h5py.File(reverse_pixel_file, 'r')

    fretry = h5py.File(retry_pixel_file, 'r')

    fother = h5py.File(other_pixel_file, 'r')

    a1, b1, c1 = fquiet['pixel_indices'][0:3]

    a2, b2, c2 = fshock['pixel_indices'][0:3]

    a3, b3, c3 = fshock_78['pixel_indices'][0:3]

    a4, b4, c4 = freverse['pixel_indices'][0:3]

    a5, b5, c5 = fretry['pixel_indices'][0:3]

    a6, b6, c6 = fother['pixel_indices'][0:3]

    all_profiles = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            30
        )
    )

    all_profiles[a1, b1, c1] = finputprofiles_quiet['profiles'][0, 0, :, ind, 0]
    all_profiles[a2, b2, c2] = finputprofiles_shock['profiles'][0, 0, :, ind, 0]
    all_profiles[a3, b3, c3] = finputprofiles_shock_78['profiles'][0, 0, :, ind, 0]
    all_profiles[a4, b4, c4] = finputprofiles_reverse['profiles'][0, 0, :, ind, 0]
    all_profiles[a5, b5, c5] = finputprofiles_retry['profiles'][0, 0, :, ind, 0]
    all_profiles[a6, b6, c6] = finputprofiles_other['profiles'][0, 0, :, ind, 0]

    all_temp = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    all_vlos = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    all_vturb = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    all_pgas = np.zeros(
        (
            frames[1] - frames[0],
            x[1] - x[0],
            y[1] - y[0],
            150
        )
    )

    all_temp[a1, b1, c1] = fout_atmos_quiet['temp'][0, 0]
    all_temp[a2, b2, c2] = fout_atmos_shock['temp'][0, 0]
    all_temp[a3, b3, c3] = fout_atmos_shock_78['temp'][0, 0]
    all_temp[a4, b4, c4] = fout_atmos_reverse['temp'][0, 0]
    all_temp[a5, b5, c5] = fout_atmos_retry['temp'][0, 0]
    all_temp[a6, b6, c6] = fout_atmos_other['temp'][0, 0]

    all_vlos[a1, b1, c1] = fout_atmos_quiet['vlos'][0, 0]
    all_vlos[a2, b2, c2] = fout_atmos_shock['vlos'][0, 0]
    all_vlos[a3, b3, c3] = fout_atmos_shock_78['vlos'][0, 0]
    all_vlos[a4, b4, c4] = fout_atmos_reverse['vlos'][0, 0]
    all_vlos[a5, b5, c5] = fout_atmos_retry['vlos'][0, 0]
    all_vlos[a6, b6, c6] = fout_atmos_other['vlos'][0, 0]

    all_vturb[a1, b1, c1] = fout_atmos_quiet['vturb'][0, 0]
    all_vturb[a2, b2, c2] = fout_atmos_shock['vturb'][0, 0]
    all_vturb[a3, b3, c3] = fout_atmos_shock_78['vturb'][0, 0]
    all_vturb[a4, b4, c4] = fout_atmos_reverse['vturb'][0, 0]
    all_vturb[a5, b5, c5] = fout_atmos_retry['vturb'][0, 0]
    all_vturb[a6, b6, c6] = fout_atmos_other['vturb'][0, 0]

    all_pgas[a1, b1, c1] = fout_atmos_quiet['pgas'][0, 0]
    all_pgas[a2, b2, c2] = fout_atmos_shock['pgas'][0, 0]
    all_pgas[a3, b3, c3] = fout_atmos_shock_78['pgas'][0, 0]
    all_pgas[a4, b4, c4] = fout_atmos_reverse['pgas'][0, 0]
    all_pgas[a5, b5, c5] = fout_atmos_retry['pgas'][0, 0]
    all_pgas[a6, b6, c6] = fout_atmos_other['pgas'][0, 0]

    return all_profiles, all_temp, all_vlos, all_vturb, all_pgas


class dataset_spot(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        global nodes, activation_nodes
        super(dataset_spot, self).__init__()

        print (activation_nodes)
        print (nodes)

        if mode == 'train':
            profiles1, temp1, vlos1, vturb1, pgas1 = get_fov1()

            profiles2, temp2, vlos2, vturb2, pgas2 = get_fov2()

            self.profiles = np.transpose(
                np.vstack([profiles1, profiles2]),
                axes=(3, 0, 1, 2)
            )

            self.temp = np.transpose(
                np.vstack([temp1, temp2]),
                axes=(3, 0, 1, 2)
            )[nodes[0]]

            self.vlos = np.transpose(
                np.vstack([vlos1, vlos2]),
                axes=(3, 0, 1, 2)
            )[nodes[1]]

            self.vturb = np.transpose(
                np.vstack([vturb1, vturb2]),
                axes=(3, 0, 1, 2)
            )[nodes[2]]

            self.pgas = np.transpose(
                np.vstack([pgas1, pgas2]),
                axes=(3, 0, 1, 2)
            )

            self.model = np.vstack([self.temp, self.vlos, self.vturb])  #, self.pgas])

            self.mean_profile = np.mean(self.profiles, axis=(1, 2, 3))

            self.std_profile = np.std(self.profiles, axis=(1, 2, 3))

            self.mean_model = np.mean(self.model, axis=(1, 2, 3))

            self.std_model = np.std(self.model, axis=(1, 2, 3))

            normalise_profile_params = np.zeros((self.profiles.shape[0], 2), dtype=np.float64)

            normalise_model_params = np.zeros((self.model.shape[0], 2), dtype=np.float64)

            normalise_profile_params[:, 0] = self.mean_profile

            normalise_profile_params[:, 1] = self.std_profile

            normalise_model_params[:, 0] = self.mean_model

            normalise_model_params[:, 1] = self.std_model

            np.savetxt(
                'weights_encdec/normalise_profile_params_{}.txt'.format(
                    activation_nodes
                ),
                normalise_profile_params
            )

            np.savetxt(
                'weights_encdec/normalise_model_params_{}.txt'.format(
                    activation_nodes
                ),
                normalise_model_params
            )

        else:
            profiles1, temp1, vlos1, vturb1, pgas1 = get_fov3()

            self.profiles = np.transpose(
                profiles1,
                axes=(3, 0, 1, 2)
            )

            self.temp = np.transpose(
               temp1,
                axes=(3, 0, 1, 2)
            )[nodes[0]]

            self.vlos = np.transpose(
                vlos1,
                axes=(3, 0, 1, 2)
            )[nodes[1]]

            self.vturb = np.transpose(
                vturb1,
                axes=(3, 0, 1, 2)
            )[nodes[2]]

            self.pgas = np.transpose(
                pgas1,
                axes=(3, 0, 1, 2)
            )

            self.model = np.vstack([self.temp, self.vlos, self.vturb])  #  , self.pgas])

            normalise_profile_params = np.loadtxt(
                'weights_encdec/normalise_profile_params_{}.txt'.format(
                    activation_nodes
                )
            )

            normalise_model_params = np.loadtxt(
                'weights_encdec/normalise_model_params_{}.txt'.format(
                    activation_nodes
                )
            )

            self.mean_profile = normalise_profile_params[:, 0]

            self.std_profile = normalise_profile_params[:, 1]

            self.mean_model = normalise_model_params[:, 0]

            self.std_model = normalise_model_params[:, 1]

        self.profiles = (self.profiles - self.mean_profile[:, np.newaxis, np.newaxis, np.newaxis]) / self.std_profile[:, np.newaxis, np.newaxis, np.newaxis]

        self.model = (self.model - self.mean_model[:, np.newaxis, np.newaxis, np.newaxis]) / self.std_model[:, np.newaxis, np.newaxis, np.newaxis]

        self.profiles = torch.from_numpy(self.profiles.astype('float32'))

        self.model = torch.from_numpy(self.model.astype('float32'))

        self.profiles = nn.functional.pad(self.profiles, (7, 7, 7, 7), mode='reflect')

        self.model = nn.functional.pad(self.model, (7, 7, 7, 7), mode='reflect')

        self.in_planes = self.profiles.shape[0]

        self.out_planes = self.model.shape[0]

        print (self.in_planes)

        print (self.out_planes)

    def __getitem__(self, index):

        return self.profiles[:, index], self.model[:, index]

    def __len__(self):
        return self.profiles.shape[1]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')


class deep_3d_inversor(object):
    def __init__(self):

        self.dataset_train = dataset_spot(mode='train')
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, shuffle=True)

        self.dataset_test = dataset_spot(mode='test')
        self.test_loader = torch.utils.data.DataLoader(self.dataset_test, shuffle=True)  

        self.in_planes = self.dataset_train.in_planes   
        self.out_planes = self.dataset_train.out_planes

        self.model = model.block(in_planes=self.in_planes, out_planes=self.out_planes)

    def optimize(self, epochs, lr=1e-4):

        self.lr = lr
        self.n_epochs = epochs

        root = 'weights_encdec'

        if not os.path.exists(root):
            os.makedirs(root)

        current_time = time.strftime("%Y-%m-%d-%H:%M")
        self.out_name = '{2}/{0}_-lr_{1}'.format(current_time, self.lr, root)

        print("Network name : {0}".format(self.out_name))

        # Copy model
        shutil.copyfile(model.__file__, '{}.model_{}.py'.format(self.out_name, activation_nodes))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lossfn_L2 = nn.MSELoss()
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                            step_size=30,
                                            gamma=0.5)

        self.loss_L2 = []        
        self.loss_L2_val = []
        best_loss = -1e10

        trainF = open('{}.loss_{}.csv'.format(self.out_name, activation_nodes), 'w')

        data = {'epoch':[], 'loss_l2': [], 'loss_l2_val': []}

        self.db = pd.DataFrame(data)

        for epoch in range(1, epochs + 1):
            self.scheduler.step()

            self.train(epoch)
            self.test()

            trainF.write('{},{},{}\n'.format(
                epoch, self.loss_L2[-1], self.loss_L2_val[-1]))
            trainF.flush()

            is_best = self.loss_L2_val[-1] > best_loss
            best_loss = max(self.loss_L2_val[-1], best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, filename='{}_{}.pth'.format(self.out_name, activation_nodes))

            # self.db.update(epoch, self.loss_L2[-1], self.loss_L2_val[-1])

            data = {'epoch':[epoch], 'loss_l2': [self.loss_L2[-1]], 'loss_l2_val': [self.loss_L2_val[-1]]}

            self.db = self.db.append(pd.DataFrame(data))

        self.db.to_hdf(
            '{}_{}.h5'.format(
                self.out_name,
                activation_nodes
            ),
            'data',
            mode='w'
        )
        trainF.close()

    def train(self, epoch):
        self.model.train()
        print("Epoch {0}/{1} - {2}".format(epoch, self.n_epochs, time.strftime("%Y-%m-%d-%H:%M:%S")))
        t = tqdm(self.train_loader)
        
        loss_L2_avg = 0.0
        
        n = 1

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (data, target) in enumerate(t):
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            loss_L2 = self.lossfn_L2(output, target)
                        
            loss_L2_avg += (loss_L2.item() - loss_L2_avg) / n            
            n += 1

            self.loss_L2.append(loss_L2_avg)

            loss_L2.backward()
            self.optimizer.step()

            t.set_postfix(loss=loss_L2_avg, lr=current_lr)

    def test(self):
        self.model.eval()
                
        loss_L2_avg = 0.0
        
        n = 1
        t = tqdm(self.test_loader)

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(t):
            
                output = self.model(data)
            
                # sum up batch loss
                loss_L2 = self.lossfn_L2(output, target)
                                
                loss_L2_avg += (loss_L2.item() - loss_L2_avg) / n                
                n += 1

                self.loss_L2_val.append(loss_L2_avg)
        
                t.set_postfix(loss=loss_L2_avg, lr=current_lr)
            
if __name__ == '__main__':
    activation_nodes = sys.argv[1]
    nodes = get_nodes(activation_nodes)
    deep_inversor = deep_3d_inversor()
    deep_inversor.optimize(50, lr=3e-4)

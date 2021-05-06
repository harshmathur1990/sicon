import numpy as np, h5py, matplotlib.pyplot as plt
from pathlib import Path


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


def make_plots():
    all_profiles, all_temp, all_vlos, all_vturb, all_pgas = get_fov3()

    f = h5py.File('weights_encdec_v2_gpu/output_fov_3_generic_v2_gpu_neural_net.nc', 'r')

    plt.close('all')

    plt.clf()

    plt.cla()

    fig, axs = plt.subplots(1, 3, figsize=(19.2, 10.8))

    axs[0].plot(f['ltau500'][15, 40, 25], all_temp[15, 40, 25], label='STiC inversion', color='#364f6B')
    axs[0].plot(f['ltau500'][15, 40, 25], f['temp'][15, 40, 25], label='Neural-Net', color='#3fC1C9')
    axs[1].plot(f['ltau500'][15, 40, 25], all_vturb[15, 40, 25] / 1e5, label='STiC inversion', color='#364f6B')
    axs[1].plot(f['ltau500'][15, 40, 25], f['vturb'][15, 40, 25] / 1e5, label='Neural-Net', color='#3fC1C9')
    axs[2].plot(f['ltau500'][15, 40, 25], all_vlos[15, 40, 25] / 1e5, label='STiC inversion', color='#364f6B')
    axs[2].plot(f['ltau500'][15, 40, 25], f['vlos'][15, 40, 25] / 1e5, label='Neural-Net', color='#3fC1C9')
    
    axs[0].set_aspect(1.0 / axs[0].get_data_ratio(), adjustable='box')
    axs[1].set_aspect(1.0 / axs[1].get_data_ratio(), adjustable='box')
    axs[2].set_aspect(1.0 / axs[2].get_data_ratio(), adjustable='box')

    axs[0].set_xlabel(r'$log$ $\tau$')
    axs[1].set_xlabel(r'$log$ $\tau$')
    axs[2].set_xlabel(r'$log$ $\tau$')

    axs[0].set_ylabel(r'$T$ $(K)$')
    axs[1].set_ylabel(r'$V_{los}$ $(km/sec)$')
    axs[2].set_ylabel(r'$V_{turb}$ $(km/sec)$')

    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    axs[2].legend(loc="upper right")

    axs[0].set_title('Temperature Variation')
    axs[1].set_title('Line of Sight Velocity Variation')
    axs[2].set_title('MIcroturbulence Variation')

    fig.suptitle('Comparison of results with Neural Network')

    fig.tight_layout()

    plt.savefig('param_compare.png', dpi=100, format='png')

    f.close()

if __name__ == '__main__':
    make_plots()

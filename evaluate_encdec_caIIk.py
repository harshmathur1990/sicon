import sys
import os
sys.path.insert(1, '/home/harsh/CourseworkRepo/stic/example')
from prepare_data import *
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.module import _addindent
import h5py
import time
import sys
import model_encdec as model
from pathlib import Path
from cubic_bezier import prepare_evaluate_bezier


ltau = np.array(
    [
        -8.       , -7.78133  , -7.77448  , -7.76712  , -7.76004  ,
        -7.75249  , -7.74429  , -7.7356   , -7.72638  , -7.71591  ,
        -7.70478  , -7.69357  , -7.68765  , -7.68175  , -7.67589  ,
        -7.66997  , -7.66374  , -7.65712  , -7.64966  , -7.64093  ,
        -7.63093  , -7.6192   , -7.6053   , -7.58877  , -7.56925  ,
        -7.54674  , -7.52177  , -7.49317  , -7.4585   , -7.41659  ,
        -7.36725  , -7.31089  , -7.24834  , -7.18072  , -7.1113   ,
        -7.04138  , -6.97007  , -6.89698  , -6.82299  , -6.74881  ,
        -6.67471  , -6.60046  , -6.52598  , -6.45188  , -6.37933  ,
        -6.30927  , -6.24281  , -6.17928  , -6.11686  , -6.05597  ,
        -5.99747  , -5.94147  , -5.88801  , -5.84684  , -5.81285  ,
        -5.78014  , -5.74854  , -5.71774  , -5.68761  , -5.65825  ,
        -5.6293   , -5.60066  , -5.57245  , -5.54457  , -5.51687  ,
        -5.48932  , -5.46182  , -5.43417  , -5.40623  , -5.37801  ,
        -5.3496   , -5.32111  , -5.29248  , -5.26358  , -5.23413  ,
        -5.20392  , -5.17283  , -5.14073  , -5.1078   , -5.07426  ,
        -5.03999  , -5.00492  , -4.96953  , -4.93406  , -4.89821  ,
        -4.86196  , -4.82534  , -4.78825  , -4.75066  , -4.71243  ,
        -4.67439  , -4.63696  , -4.59945  , -4.5607   , -4.52212  ,
        -4.48434  , -4.44653  , -4.40796  , -4.36863  , -4.32842  ,
        -4.28651  , -4.24205  , -4.19486  , -4.14491  , -4.09187  ,
        -4.03446  , -3.97196  , -3.90451  , -3.83088  , -3.7496   ,
        -3.66     , -3.56112  , -3.4519   , -3.33173  , -3.20394  ,
        -3.07448  , -2.94444  , -2.8139   , -2.68294  , -2.55164  ,
        -2.42002  , -2.28814  , -2.15605  , -2.02377  , -1.89135  ,
        -1.7588   , -1.62613  , -1.49337  , -1.36127  , -1.23139  ,
        -1.10699  , -0.99209  , -0.884893 , -0.782787 , -0.683488 ,
        -0.584996 , -0.485559 , -0.383085 , -0.273456 , -0.152177 ,
        -0.0221309,  0.110786 ,  0.244405 ,  0.378378 ,  0.51182  ,
        0.64474  ,  0.777188 ,  0.909063 ,  1.04044  ,  1.1711
    ]
)

quiet_nodes = [
    np.array([65, 95, 112, 119, 127, 140]),
    np.array([140]),
    np.array([68, 97, 112, 128])
]

emission_nodes = [
    np.array([95, 112, 119, 127, 140]),
    np.array([50, 95, 112, 127, 140]),
    np.array([68, 97, 112, 128])
]

def get_nodes(nodename='emission'):
    if nodename == 'emission':
        return emission_nodes
    else:
        return quiet_nodes

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

    return all_profiles


class deep_3d_inversor(object):
    def __init__(self, checkpoint=None):

        global nodes, activation_nodes
        # Instantiate the model
        self.model = model.block(in_planes=30, out_planes=nodes[0].size + nodes[1].size + nodes[2].size)
        self.checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(self.checkpoint['state_dict'])    

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

        self.min_profile = normalise_profile_params[:, 0]

        self.max_profile = normalise_profile_params[:, 1]

        self.min_model = normalise_model_params[:, 0]

        self.max_model = normalise_model_params[:, 1]
    
    def evaluate(self, save_output=False):

        # shape = time, wavelength, x, y
        all_profiles = np.transpose(get_fov3(), axes=(0, 3, 1, 2))

        all_profiles = (all_profiles - self.min_profile[np.newaxis, :, np.newaxis, np.newaxis]) / (self.max_profile[np.newaxis, :, np.newaxis, np.newaxis] - self.min_profile[np.newaxis, :, np.newaxis, np.newaxis])

        all_profiles = torch.from_numpy(all_profiles.astype('float32'))

        input = nn.functional.pad(all_profiles, (7, 7, 7, 7), mode='reflect')

        # Put the model in evaulation mode                                                       
        self.model.eval()

        # Put PyTorch in evaluation mode
        with torch.no_grad():
            
            # Input tensor
            
            # Evluate the model and rescale the output
            start = time.time()
            output = self.model(input).data
            output = (output * (self.max_model[np.newaxis, :, np.newaxis, np.newaxis] - self.min_model[np.newaxis, :, np.newaxis, np.newaxis]) ) + self.min_model[np.newaxis, :, np.newaxis, np.newaxis]
            print('Elapsed time : {0} s'.format(time.time()-start))

            # shape = time, x, y, logtau
            output = np.transpose(output, axes=(0, 2, 3, 1))

            output = output[:, 7:57, 7:57, :]

            temp = output[:, :, :, 0:nodes[0].size]

            vlos = output[:, :, :, nodes[0].size:nodes[0].size + nodes[1].size]

            vturb = output[:, :, :, nodes[0].size + nodes[1].size:nodes[0].size + nodes[1].size + nodes[2].size]

            interp_temp = prepare_evaluate_bezier(ltau[nodes[0]], ltau)
            interp_vlos = prepare_evaluate_bezier(ltau[nodes[1]], ltau)
            interp_vturb = prepare_evaluate_bezier(ltau[nodes[2]], ltau)

            vec_evaluate_temp = np.vectorize(interp_temp, signature='(m)->(n)')
            vec_evaluate_vlos = np.vectorize(interp_vlos, signature='(m)->(n)')
            vec_evaluate_vturb = np.vectorize(interp_vturb, signature='(m)->(n)')

            all_temp = vec_evaluate_temp(temp)
            all_vlos = vec_evaluate_vlos(vlos)
            all_vturb = vec_evaluate_vturb(vturb)

            m = sp.model(nx=all_temp.shape[2], ny=all_temp.shape[1], nt=all_temp.shape[0], ndep=150)

            m.ltau[:, :, :] = ltau[indices]

            m.temp = all_temp

            m.pgas[:, :, :] = pgas[indices]

            m.vlos = all_vlos

            m.vturb = all_vturb

            m.write('output_fov_3_{}_neural_net.nc'.format(activation_nodes))

if __name__ == '__main__':
    activation_nodes = sys.argv[1]
    nodes = get_nodes(activation_nodes)
    deep_network = deep_3d_inversor(checkpoint=sys.argv[2])
    deep_network.evaluate()

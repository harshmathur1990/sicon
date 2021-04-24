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
import os
import model_encdec as model
from pathlib import Path


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

pgas = np.array(
    [
        1.92480162e-01, 2.10335478e-01, 2.11099088e-01, 2.11936578e-01,
        2.12759241e-01, 2.13655323e-01, 2.14651138e-01, 2.15732709e-01,
        2.16910645e-01, 2.18287319e-01, 2.19797805e-01, 2.21369892e-01,
        2.22221285e-01, 2.23084763e-01, 2.23957345e-01, 2.24854335e-01,
        2.25815415e-01, 2.26856306e-01, 2.28054076e-01, 2.29489878e-01,
        2.31181175e-01, 2.33230412e-01, 2.35753953e-01, 2.38895416e-01,
        2.42812648e-01, 2.47627556e-01, 2.53370225e-01, 2.60512590e-01,
        2.70063460e-01, 2.83080816e-01, 3.00793827e-01, 3.24782073e-01,
        3.57132912e-01, 4.00601238e-01, 4.56965476e-01, 5.29293299e-01,
        6.24008834e-01, 7.49855459e-01, 9.16294158e-01, 1.13508153e+00,
        1.42164230e+00, 1.79735219e+00, 2.28865838e+00, 2.92158079e+00,
        3.71474624e+00, 4.67952347e+00, 5.81029558e+00, 7.12056494e+00,
        8.65885639e+00, 1.04314861e+01, 1.24189425e+01, 1.46121168e+01,
        1.69987698e+01, 1.90496445e+01, 2.08928623e+01, 2.28027058e+01,
        2.47815571e+01, 2.68437004e+01, 2.89950733e+01, 3.12255573e+01,
        3.35608482e+01, 3.60105362e+01, 3.85653992e+01, 4.12354431e+01,
        4.40376701e+01, 4.69793587e+01, 5.00764923e+01, 5.33600731e+01,
        5.68586159e+01, 6.05851173e+01, 6.45416412e+01, 6.87255859e+01,
        7.31585388e+01, 7.78767090e+01, 8.29482269e+01, 8.84405975e+01,
        9.44148178e+01, 1.00943398e+02, 1.08040993e+02, 1.15709175e+02,
        1.24027237e+02, 1.33072845e+02, 1.42778915e+02, 1.53122635e+02,
        1.64239197e+02, 1.76195251e+02, 1.89045700e+02, 2.02897873e+02,
        2.17846008e+02, 2.34042114e+02, 2.51210236e+02, 2.69186646e+02,
        2.88341248e+02, 3.09395111e+02, 3.31706787e+02, 3.54934540e+02,
        3.79624237e+02, 4.06384979e+02, 4.35409149e+02, 4.67003418e+02,
        5.02125732e+02, 5.41979980e+02, 5.87387085e+02, 6.39156738e+02,
        6.98575623e+02, 7.68398621e+02, 8.51376160e+02, 9.49656250e+02,
        1.06806201e+03, 1.21330579e+03, 1.39257166e+03, 1.61584387e+03,
        1.89664856e+03, 2.25192163e+03, 2.69018799e+03, 3.20748486e+03,
        3.81380298e+03, 4.52528320e+03, 5.36149463e+03, 6.34581787e+03,
        7.50641797e+03, 8.87612695e+03, 1.04934688e+04, 1.24038418e+04,
        1.46598135e+04, 1.73236582e+04, 2.04680117e+04, 2.41766445e+04,
        2.85175430e+04, 3.35073477e+04, 3.90268633e+04, 4.48068984e+04,
        5.07996992e+04, 5.70373594e+04, 6.35809258e+04, 7.05180547e+04,
        7.79622109e+04, 8.61010938e+04, 9.53656875e+04, 1.06372070e+05,
        1.19236258e+05, 1.33759531e+05, 1.49984500e+05, 1.68110281e+05,
        1.88273656e+05, 2.10752016e+05, 2.35874125e+05, 2.63987969e+05,
        2.95532281e+05, 3.30931750e+05
    ]
)


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

        # Instantiate the model
        self.model = model.block(in_planes=30, out_planes=450)
        self.checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(self.checkpoint['state_dict'])    

        normalise_profile_params = np.loadtxt('normalise_profile_params.txt')

        normalise_model_params = np.loadtxt('normalise_model_params.txt')

        self.min_profile = normalise_profile_params[:, 0]

        self.max_profile = normalise_profile_params[:, 1]

        self.min_model = normalise_model_params[:, 0]

        self.max_model = normalise_model_params[:, 1]
    
    def evaluate(self, save_output=False):

        all_profiles = get_fov3()

        self.stokes = np.transpose(all_profiles, axes=(0, 3, 1, 2))

        self.stokes = (self.stokes - self.min_profile[np.newaxis, :, np.newaxis, np.newaxis]) / (self.max_profile[np.newaxis, :, np.newaxis, np.newaxis] - self.min_profile[np.newaxis, :, np.newaxis, np.newaxis])

        # Put the model in evaulation mode                                                       
        self.model.eval()

        # Put PyTorch in evaluation mode
        with torch.no_grad():
            
            # Input tensor
            input = torch.from_numpy(self.stokes.astype('float32'))
            
            # Evluate the model and rescale the output
            start = time.time()
            output = self.model(input).data
            output = (output * (self.max_model[np.newaxis, :, np.newaxis, np.newaxis] - self.min_model[np.newaxis, :, np.newaxis, np.newaxis]) ) + self.min_model[np.newaxis, :, np.newaxis, np.newaxis]
            print('Elapsed time : {0} s'.format(time.time()-start))            

            output = np.transpose(output, axes=(0, 2, 3, 1))

            temp = output[:, :, :, 0:150]

            vlos = output[:, :, :, 150:300]

            vturb = output[:, :, :, 300:450]

            m = sp.model(nx=temp.shape[2], ny=temp.shape[1], nt=temp.shape[0], ndep=150)

            m.ltau[:, :, :] = ltau

            m.temp = temp

            m.pgas[:, :, :] = pgas

            m.vlos = vlos

            m.vturb = vturb

            m.write('output_fov_3_neural_net.nc')

if __name__ == '__main__':
    plt.close('all')
    os.chdir('/home/harsh/CourseworkRepo/stic/example')
    from prepare_data import *
    os.chdir('/home/harsh/CourseworkRepo/sicon')
    deep_network = deep_3d_inversor(checkpoint='')
    deep_network.evaluate()

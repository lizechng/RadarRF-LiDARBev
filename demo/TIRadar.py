import numpy as np

from .utils import load_json

# =============================== radar ======================================
def read_raw_adcdata_from_bin(adc_path, radar_config):
    '''
    read raw adcdata from .adcdata.bin
    :param adc_path:
    :param radar_config:
    :return:
    '''
    num_sample = radar_config['num_sample']
    num_loop = radar_config['num_loop']
    num_RX = radar_config['num_RX']
    num_chirp = radar_config['num_chirp']

    adcdata = np.fromfile(adc_path, dtype='int16')
    adcdata = adcdata.reshape((num_sample, num_loop, num_RX, num_chirp, 2, 2), order='F')
    adcdata = adcdata[..., 0, 0] + adcdata[..., 1, 0] * 1j

    return adcdata

def calibrate_raw_adcdata(adcdata, radar_config):
    '''
    calibrate raw adcdata
    :param adcdata:
    :param radar_config:
    :return:
    '''
    out_data = np.zeros(adcdata.shape, dtype=adcdata.dtype)

    transfer_order = radar_config['transfer_order']
    num_tx = len(transfer_order)
    adc_sample_rate = radar_config['adc_sample_rate']
    chirp_slope = radar_config['chirp_slope']
    num_sample = radar_config['num_sample']
    num_loop = radar_config['num_loop']

    range_mat = np.array(radar_config['calib_data']['RangeMat'])
    fs_calib = radar_config['calib_data']['Sampling_Rate_sps']
    slope_calib = radar_config['calib_data']['Slope_MHzperus'] * 1e12
    calibration_interp = radar_config['calibration_interp']
    peak_val_mat = np.array(radar_config['calib_data']['PeakValMat']['_ArrayData_'][0]).reshape((12, 16)) + np.array(
        radar_config['calib_data']['PeakValMat']['_ArrayData_'][1]).reshape((12, 16)) * 1j
    phase_calib_only = radar_config['phase_calib_only']

    tx_id_ref = transfer_order[0]
    for i_tx in range(num_tx):
        tx_id = transfer_order[i_tx]

        # construct the frequency compensation matrix
        freq_calib = (range_mat[tx_id-1, :] - range_mat[tx_id_ref-1, 0]) * fs_calib / adc_sample_rate * chirp_slope / slope_calib
        freq_calib = 2 * np.pi * freq_calib / (num_sample * calibration_interp)
        correction_vec = np.exp(np.matmul(np.array(range(num_sample)).reshape((-1, 1)), freq_calib.reshape((1, -1))) * 1j).conj().T

        freq_correction_mat = np.tile(correction_vec, (num_loop, 1, 1))
        freq_correction_mat = np.swapaxes(freq_correction_mat, 0, 1)
        freq_correction_mat = np.swapaxes(freq_correction_mat, 0, 2)
        out_data[:, :, :, i_tx] = adcdata[:, :, :, i_tx] * freq_correction_mat

        # construct the phase compensation matrix
        phase_calib = peak_val_mat[tx_id_ref-1, 0] / peak_val_mat[tx_id-1, :]
        # remove amplitude calibration
        if phase_calib_only == 1:
            phase_calib = phase_calib / np.abs(phase_calib)
        phase_correction_mat = np.tile(phase_calib, (num_sample, num_loop, 1))
        out_data[:, :, :, i_tx] *= phase_correction_mat

    return out_data

def rangeFFT(adcdata, radar_config):
    '''
    rangeFFT
    :param adcdata:
    :param radar_config:
    :return:
    '''
    output = adcdata

    # DC offset compensation
    output = output - output.mean(axis=0)

    output = np.fft.fft(output, n=radar_config['rangeFFT_size'], axis=0)

    return output

def dopplerFFT(output_rangeFFT, radar_config):
    '''
    dopplerFFT
    :param output_rangeFFT:
    :param radar_config:
    :return:
    '''
    output = output_rangeFFT
    output = np.fft.fft(output, n=radar_config['dopplerFFT_size'], axis=1)
    output = np.fft.fftshift(output, axes=1)

    return output

def rdm_cfar(output_dopplerFFT, radar_config):
    '''
    do CFAR on range doppler map
    :param output_dopplerFFT:
    :param radar_config:
    :return: output_detection
    '''

    n_obj_range, index_obj_range, _, _, _ = range_cfar_os(output_dopplerFFT, radar_config)
    if n_obj_range > 0:
        n_obj, index_obj, energy_obj, noise_obj, snr_obj = doppler_cfar_os_cyclicity(output_dopplerFFT, radar_config, index_obj_range)

        range_index = np.array(index_obj)[:, 0]
        doppler_index = np.array(index_obj)[:, 1]
        range_obj = list(range_index * radar_config['range_resolution'])
        velocity_obj = list((doppler_index - output_dopplerFFT.shape[1]/2) * radar_config['velocity_resolution'])
        if radar_config['apply_vmax_extend'] == 1:
            # doppler phase correction due to TDM MIMO with applying vmax extention algorithm

            # mask_for_apply_vmax_extend = np.array(range_obj) >= radar_config['min_range_for_apply_vmax_extend']
            mask_v_positive = np.array(velocity_obj) > 0
            mask_v_unpositive = np.logical_not(mask_v_positive)

            num_tx = radar_config['num_TX']
            dopplerFFT_size = radar_config['dopplerFFT_size']
            doppler_index_unwrap = np.tile(doppler_index, (num_tx, 1)).T + dopplerFFT_size * (
                        np.tile(np.arange(num_tx) - num_tx / 2, (n_obj, 1)) + np.tile(mask_v_unpositive, (num_tx, 1)).T)

            sig = output_dopplerFFT[range_index, doppler_index, :, :]

            # Doppler phase correction due to TDM MIMO
            delta_phi = 2 * np.pi * (doppler_index_unwrap - dopplerFFT_size / 2) / (num_tx * dopplerFFT_size)

            # construct all possible signal vectors based on the number of possible hypothesis
            tx = np.arange(num_tx)
            correct_matrix = np.exp(-1j * np.matmul(delta_phi.reshape((delta_phi.shape[0], delta_phi.shape[1], 1)),
                                                    np.tile(tx, (n_obj, 1, 1))))
            sig_correct_all = np.matmul(
                np.expand_dims(sig.swapaxes(1, 2), len(sig.shape)),
                np.expand_dims(correct_matrix.swapaxes(1, 2), len(correct_matrix.shape)).swapaxes(2, 3)
            ).swapaxes(1, 2)

            # find the overlap antenna ID that can be used for phase compensation
            overlap_antenna_info_1tx = get_overlap_antenna_info(radar_config)

            # use overlap antenna to do max velocity unwrap
            sig_associated = sig[:, overlap_antenna_info_1tx[:, 0], overlap_antenna_info_1tx[:, 1]]
            sig_overlaped = sig[:, overlap_antenna_info_1tx[:, 2], overlap_antenna_info_1tx[:, 3]]
            # check the phase difference of each overlap antenna pair for each hypothesis
            angle_sum_test = np.zeros((n_obj, overlap_antenna_info_1tx.shape[0], delta_phi.shape[1]))
            for i_sig in range(angle_sum_test.shape[1]):
                signal2 = np.matmul(
                    np.expand_dims(sig_overlaped[:, :i_sig + 1], len(sig_overlaped.shape)),
                    np.exp(-1j * np.expand_dims(delta_phi, 1))
                )
                signal1 = np.tile(np.expand_dims(sig_associated[:, :i_sig + 1], len(sig_associated.shape)), (1, 1, signal2.shape[2]))
                angle_sum_test[:, i_sig, :] = np.angle(np.sum(signal1 * signal2.conj(), axis=1))

            # chosee the hypothesis with minimum phase difference to estimate the unwrap factor
            doppler_unwrap_integ_overlap_val = np.min(np.abs(angle_sum_test), axis=2)
            doppler_unwrap_integ_overlap_index = np.argmin(np.abs(angle_sum_test), axis=2)

            # test the angle FFT SNR
            rx_tx_noredundant_row1 = get_rx_tx_noredundant_row1(radar_config)
            sig_correct_row1 = sig_correct_all[:, rx_tx_noredundant_row1[:, 0], rx_tx_noredundant_row1[:, 1], :]
            angleFFT_size = radar_config['angleFFT_size']
            sig_correct_row1_azimuthFFT = np.fft.fftshift(
                np.fft.fft(sig_correct_row1, n=angleFFT_size, axis=1),
                axes=1
            )
            angle_bin_skip_left = radar_config['angle_bin_skip_left']
            angle_bin_skip_right = radar_config['angle_bin_skip_right']
            sig_correct_row1_azimuthFFT_abs_cut = np.abs(
                sig_correct_row1_azimuthFFT[:, angle_bin_skip_left - 1:angleFFT_size - angle_bin_skip_right-1, :])
            doppler_unwrap_integ_FFT_max_index= np.argmax(np.max(sig_correct_row1_azimuthFFT_abs_cut, axis=1), axis=1)

            doppler_unwrap_integ_index = [np.argmax(np.bincount(doppler_unwrap_integ_overlap_index[i, :])) for i in
                                          range(doppler_unwrap_integ_overlap_index.shape[0])]
            doppler_unwrap_integ_index = np.array(doppler_unwrap_integ_index)

            # overlap antenna method is applied by default
            sig_correct = sig_correct_all[np.arange(n_obj), :, :, doppler_unwrap_integ_index]

            # corret velocity after applying the integer value
            doppler_index_unwraped = doppler_index_unwrap[np.arange(n_obj), doppler_unwrap_integ_index]
            doppler_index_FFT = doppler_index_unwrap[np.arange(n_obj), doppler_unwrap_integ_FFT_max_index]

            output_detection = {
                'n_obj': n_obj,
                'range_index': list(range_index.astype('int')),
                'range': range_obj,
                'energy': energy_obj,
                'noise': noise_obj,
                'snr': snr_obj,
                'sig_correct': sig_correct,
                'doppler_index_origin': list(doppler_index.astype('int')),
                'velocity_origin': velocity_obj,
                'doppler_index_unwraped': list(doppler_index_unwraped.astype('int')),
                'velocity_unwraped': list(
                    (doppler_index_unwraped - output_dopplerFFT.shape[1] / 2) * radar_config['velocity_resolution']),
                # 'doppler_index_FFT': list(doppler_index_FFT.astype('int')),
                # 'velocity_FFT': list(
                #     (doppler_index_FFT - output_dopplerFFT.shape[1] / 2) * radar_config['velocity_resolution']),
            }
            output_detection['doppler_index'] = output_detection['doppler_index_unwraped']
            output_detection['velocity'] = output_detection['velocity_unwraped']

            return output_detection

        else:
            # doppler phase correction due to TDM MIMO without applying vmax extention algorithm
            sig = output_dopplerFFT[range_index, doppler_index, :, :]

            delta_phi = 2 * np.pi * (doppler_index - output_dopplerFFT.shape[1] / 2) / (
                        output_dopplerFFT.shape[3] * output_dopplerFFT.shape[1])
            delta_phi = np.tile(delta_phi, (1, 1)).T

            tx = np.array(range(output_dopplerFFT.shape[3]))
            tx = np.tile(tx, (1, 1))

            correct_matrix = np.exp(-1j * np.matmul(delta_phi, tx))
            correct_matrix = np.tile(correct_matrix, (output_dopplerFFT.shape[2], 1, 1))
            correct_matrix = correct_matrix.swapaxes(0, 1)

            sig_correct = sig * correct_matrix

            output_detection = {
                'n_obj': n_obj,
                'range_index': list(range_index.astype('int')),
                'range': range_obj,
                'doppler_index': list(doppler_index.astype('int')),
                'velocity': velocity_obj,
                'energy': energy_obj,
                'noise': noise_obj,
                'snr': snr_obj,
                'sig_correct': sig_correct,
            }
            return output_detection
    else:
        return None

def get_rx_tx_noredundant_row1(radar_config):
    '''
    get unique positions' (i_rx, i_chirp) in 1th row
    :param radar_config:
    :return: rx_tx_noredundant_row1: 86x2, (i_rx, i_chirp)
    '''
    tx_position_azimuth = radar_config['tx_position_azimuth']
    tx_position_elevation = radar_config['tx_position_elevation']
    rx_position_azimuth = radar_config['rx_position_azimuth']
    rx_position_elevation = radar_config['rx_position_elevation']
    num_tx = radar_config['num_TX']
    num_rx = radar_config['num_RX']

    virtual_position_azimuth = np.tile(np.array(tx_position_azimuth), (len(rx_position_azimuth), 1)) + np.tile(
        np.array(rx_position_azimuth), (len(tx_position_azimuth), 1)).T
    virtual_position_elevation = np.tile(np.array(tx_position_elevation), (len(rx_position_elevation), 1)) + np.tile(
        np.array(rx_position_elevation), (len(tx_position_elevation), 1)).T
    rx_id = np.tile(np.arange(num_rx), (num_tx, 1)).T
    tx_id = np.tile(np.array(radar_config['transfer_order']) - 1, (num_rx, 1))

    virtual_position_redundant = np.hstack((virtual_position_azimuth.reshape((-1, 1)), virtual_position_elevation.reshape((-1, 1))))
    rx_tx = np.hstack((rx_id.reshape((-1, 1)), tx_id.reshape((-1, 1))))
    mask = virtual_position_redundant[:, 1] == 0
    virtual_position_redundant_row1 = virtual_position_redundant[mask]
    rx_tx_row1 = rx_tx[mask]
    virtual_position_noredundant_row1, virtual_position_index_row1 = np.unique(
        virtual_position_redundant_row1, axis=0, return_index=True)
    rx_tx_noredundant_row1 = rx_tx_row1[virtual_position_index_row1, :]

    return rx_tx_noredundant_row1

def get_overlap_antenna_info(radar_config):
    '''
    find overlaped rx_tx pairs and associated pairs
    :param radar_config:
    :return: overlap_antenna_info: rx_associated, tx_associated, rx_overlaped, tx_overlaped, dif_tx, position_azimuth, position_elevation
    '''
    tx_position_azimuth = radar_config['tx_position_azimuth']
    tx_position_elevation = radar_config['tx_position_elevation']
    rx_position_azimuth = radar_config['rx_position_azimuth']
    rx_position_elevation = radar_config['rx_position_elevation']
    num_tx = radar_config['num_TX']
    num_rx = radar_config['num_RX']

    virtual_position_azimuth = np.tile(np.array(tx_position_azimuth), (len(rx_position_azimuth), 1)) + np.tile(
        np.array(rx_position_azimuth), (len(tx_position_azimuth), 1)).T
    virtual_position_elevation = np.tile(np.array(tx_position_elevation),
                                         (len(rx_position_elevation), 1)) + np.tile(
        np.array(rx_position_elevation), (len(tx_position_elevation), 1)).T
    rx_id = np.tile(np.array(range(num_rx)), (num_tx, 1)).T
    # tx_id = np.tile(np.array(range(num_tx)), (num_rx, 1))
    tx_id = np.tile(np.array(radar_config['transfer_order']) - 1, (num_rx, 1))

    virtual_position_redundant = np.hstack(
        (virtual_position_azimuth.reshape((-1, 1)), virtual_position_elevation.reshape((-1, 1))))
    rx_tx = np.hstack((rx_id.reshape((-1, 1)), tx_id.reshape((-1, 1))))
    virtual_position_noredundant, virtual_position_noredundant_index = np.unique(virtual_position_redundant, axis=0,
                                                                                 return_index=True)
    virtual_position_overlap_index = np.setxor1d(np.arange(0, virtual_position_redundant.shape[0]),
                                                 virtual_position_noredundant_index)
    virtual_position_overlap = virtual_position_redundant[virtual_position_overlap_index, :]
    rx_tx_overlaped = rx_tx[virtual_position_overlap_index, :]
    rx_tx_associated = []
    for i in range(rx_tx_overlaped.shape[0]):
        mask = (virtual_position_noredundant == virtual_position_overlap[i])
        mask = np.logical_and(mask[:, 0], mask[:, 1])
        rx_tx_index_associated = virtual_position_noredundant_index[mask][0]
        rx_tx_associated.append(list(rx_tx[rx_tx_index_associated]))
    rx_tx_associated = np.array(rx_tx_associated)
    # rx_associated, tx_associated, rx_overlaped, tx_overlaped, dif_tx, position_azimuth, position_elevation
    overlap_antenna_info = np.hstack((rx_tx_associated, rx_tx_overlaped,
                                      np.abs(rx_tx_associated[:, 1] - rx_tx_overlaped[:, 1]).reshape((-1, 1)),
                                      virtual_position_overlap[:, 0].reshape((-1, 1)),
                                      virtual_position_overlap[:, 1].reshape((-1, 1))))

    overlap_antenna_info_1tx = overlap_antenna_info[overlap_antenna_info[:, 4] == 1]
    # sort
    sorted_index = np.argsort(overlap_antenna_info_1tx[:, 5])
    overlap_antenna_info_1tx = overlap_antenna_info_1tx[sorted_index, :]


    return overlap_antenna_info_1tx

def doppler_cfar_os_cyclicity(output_dopplerFFT, radar_config, index_obj_range):
    '''
    as for doppler dimension, use doppler cyclicity to run cfar_os
    :param output_dopplerFFT:
    :param radar_config:
    :param index_obj_range:
    :return: n_obj: the number of detected objects
    :return: index_obj: the index of detected objects in sig_integrate
    :return: energy_obj: the energy of detected objects
    :return: noise_obj: the noise of detected objects
    :return: snr_obj: the snr of detected objects
    '''

    refWinSize = radar_config['doppler_cfar_os_refWinSize']
    guardWinSize = radar_config['doppler_cfar_os_guardWinSize']
    K0 = radar_config['doppler_cfar_os_K0']
    maxEnable = radar_config['doppler_cfar_os_maxEnable']
    sortSelectFactor = radar_config['doppler_cfar_os_sortSelectFactor']
    gaptot = refWinSize + guardWinSize
    n_obj = 0
    index_obj = []
    energy_obj = []
    noise_obj = []
    snr_obj = []

    sig_integrate = np.sum(
        np.power(np.abs(output_dopplerFFT.reshape((output_dopplerFFT.shape[0], output_dopplerFFT.shape[1], -1))), 2),
        axis=2) + 1

    index_obj_range = np.array(index_obj_range)
    index_obj_range_unique = np.unique(index_obj_range[:, 0])
    for i_range in index_obj_range_unique:
        sigv = sig_integrate[i_range, :]
        # cyclicity
        vecMid = sigv
        vecLeft = sigv[-gaptot:]
        vecRight = sigv[0: gaptot]
        vec = np.hstack((vecLeft, vecMid, vecRight))

        for j in range(len(vecMid)):
            index_cur = j + gaptot
            index_left = list(range(index_cur - gaptot, index_cur - guardWinSize))
            index_right = list(range(index_cur + guardWinSize + 1, index_cur + gaptot + 1))

            sorted_res = np.sort(np.hstack((vec[index_left], vec[index_right])), axis=0)
            value_selected = sorted_res[int(np.ceil(sortSelectFactor * len(sorted_res))-1)]

            if maxEnable == 1:
                # whether value_selected is the local max value
                value_local = vec[index_cur - gaptot: index_cur + gaptot + 1]
                value_max = value_local.max()
                if vec[index_cur] >= K0*value_selected and vec[index_cur] >= value_max:
                    if j in index_obj_range[index_obj_range[:, 0] == i_range, 1]:  # whether j also detected in range_cfar
                        n_obj += 1
                        index_obj.append([i_range, j])
                        energy_obj.append(vec[index_cur])
                        noise_obj.append(value_selected)
                        snr_obj.append(vec[index_cur]/value_selected)
            else:
                if vec[index_cur] >= K0 * value_selected:
                    if j in index_obj_range[index_obj_range[:, 0] == i_range, 1]:
                        n_obj += 1
                        index_obj.append([i_range, j])
                        energy_obj.append(vec[index_cur])
                        noise_obj.append(value_selected)
                        snr_obj.append(vec[index_cur] / value_selected)

    return n_obj, index_obj, energy_obj, noise_obj, snr_obj

def range_cfar_os(output_dopplerFFT, radar_config):
    '''
    as for range dimension, run cfar_os
    :param output_dopplerFFT:
    :param radar_config:
    :return: n_obj: the number of detected objects
    :return: index_obj: the index of detected objects in sig_integrate
    :return: energy_obj: the energy of detected objects
    :return: noise_obj: the noise of detected objects
    :return: snr_obj: the snr of detected objects
    '''

    sig_integrate = np.sum(
        np.power(np.abs(output_dopplerFFT.reshape((output_dopplerFFT.shape[0], output_dopplerFFT.shape[1], -1))), 2),
        axis=2) + 1


    refWinSize = radar_config['range_cfar_os_refWinSize']
    guardWinSize = radar_config['range_cfar_os_guardWinSize']
    K0 = radar_config['range_cfar_os_K0']
    discardCellLeft = radar_config['range_cfar_os_discardCellLeft']
    discardCellRight = radar_config['range_cfar_os_discardCellRight']
    maxEnable = radar_config['range_cfar_os_maxEnable']
    sortSelectFactor = radar_config['range_cfar_os_sortSelectFactor']
    gaptot = refWinSize + guardWinSize
    n_obj = 0
    index_obj = []
    energy_obj = []
    noise_obj = []
    snr_obj = []

    n_range = sig_integrate.shape[0]
    n_doppler = sig_integrate.shape[1]
    for i_doppler in range(n_doppler):
        sigv = sig_integrate[:, i_doppler]
        vecMid = sigv[discardCellLeft: n_range-discardCellRight]
        vecLeft = vecMid[0: gaptot]
        vecRight = vecMid[-gaptot:]
        vec = np.hstack((vecLeft, vecMid, vecRight))

        for j in range(len(vecMid)):
            index_cur = j + gaptot
            index_left = list(range(index_cur - gaptot, index_cur - guardWinSize))
            index_right = list(range(index_cur + guardWinSize + 1, index_cur + gaptot + 1))

            sorted_res = np.sort(np.hstack((vec[index_left], vec[index_right])), axis=0)
            value_selected = sorted_res[int(np.ceil(sortSelectFactor * len(sorted_res))-1)]

            if maxEnable == 1:
                # whether value_selected is the local max value
                value_local = vec[index_cur - gaptot: index_cur + gaptot + 1]
                value_max = value_local.max()
                if vec[index_cur] >= K0*value_selected and vec[index_cur] >= value_max:
                    n_obj += 1
                    index_obj.append([discardCellLeft+j, i_doppler])
                    energy_obj.append(vec[index_cur])
                    noise_obj.append(value_selected)
                    snr_obj.append(vec[index_cur]/value_selected)
            else:
                if vec[index_cur] >= K0 * value_selected:
                    n_obj += 1
                    index_obj.append([discardCellLeft + j, i_doppler])
                    energy_obj.append(vec[index_cur])
                    noise_obj.append(value_selected)
                    snr_obj.append(vec[index_cur] / value_selected)

    return n_obj, index_obj, energy_obj, noise_obj, snr_obj

def doa(output_detection, radar_config):
    '''

    :param output_detection:
    :param radar_config:
    :return:
    '''
    n_obj = 0
    range_index = []
    range_val = []
    doppler_index = []
    velocity = []
    azimuth_index = []
    azimuth = []
    elevation_index = []
    elevation = []
    intensity = []
    if radar_config['apply_vmax_extend'] == 1:
        doppler_index_origin = []
        velocity_origin = []

    for i in range(output_detection['n_obj']):
        sig = output_detection['sig_correct'][i, :, :]
        angle_obj, output_elevationFFT = DOA_beamformingFFT_2D(sig, radar_config)
        # plt.imshow(np.abs(output_elevationFFT))
        # plt.show()

        for j in range(len(angle_obj)):
            n_obj += 1
            range_index.append(output_detection['range_index'][i])
            range_val.append(output_detection['range'][i])
            doppler_index.append(output_detection['doppler_index'][i])
            velocity.append(output_detection['velocity'][i])
            if radar_config['apply_vmax_extend'] == 1:
                doppler_index_origin.append(output_detection['doppler_index_origin'][i])
                velocity_origin.append(output_detection['velocity_origin'][i])

            azimuth_index.append(angle_obj[j][2])
            azimuth.append(angle_obj[j][0])
            elevation_index.append(angle_obj[j][3])
            elevation.append(angle_obj[j][1])

            intensity.append(np.abs(output_elevationFFT[angle_obj[j][2], angle_obj[j][3]]))

    if n_obj > 0:
        output_doa = {
            'range_index': range_index,
            'range': range_val,
            'doppler_index': doppler_index,
            'velocity': velocity,
            'azimuth_index': azimuth_index,
            'azimuth': azimuth,
            'elevation_index': elevation_index,
            'elevation': elevation,
            'intensity': intensity,
        }
        if radar_config['apply_vmax_extend'] == 1:
            output_doa['doppler_index_origin'] = doppler_index_origin
            output_doa['velocity_origin'] = velocity_origin

        return output_doa
    else:
        return None

def DOA_beamformingFFT_2D(sig, radar_config):
    '''

    :param sig: 16x12
    :param radar_config:
    :return:
    '''
    sidelobeLevel_dB_azim = radar_config['doa_sidelobeLevel_dB_azim']
    sidelobeLevel_dB_elev = radar_config['doa_sidelobeLevel_dB_elev']
    doa_fov_azim = radar_config['doa_fov_azim']
    doa_fov_elev = radar_config['doa_fov_elev']
    doa_unitDis = radar_config['doa_unitDis']
    azimuthFFT_size = radar_config['azimuthFFT_size']
    elevationFFT_size = radar_config['elevationFFT_size']

    sig_space = single_obj_signal_space_mapping(sig, radar_config)

    output_azimuthFFT = np.fft.fftshift(np.fft.fft(sig_space, n=azimuthFFT_size, axis=0), axes=0)
    output_elevationFFT = np.fft.fftshift(np.fft.fft(output_azimuthFFT, n=elevationFFT_size, axis=1), axes=1)
    azimuth_bins = np.array(range(-azimuthFFT_size, azimuthFFT_size, 2)) * np.pi / azimuthFFT_size
    elevation_bins = np.array(range(-elevationFFT_size, elevationFFT_size, 2)) * np.pi / elevationFFT_size

    spec_azim = np.abs(output_azimuthFFT[:, 0])
    _, peak_loc_azim = DOA_BF_PeakDet_loc(spec_azim, radar_config, sidelobeLevel_dB_azim)

    n_obj = 0
    angle_obj = []
    for i in range(len(peak_loc_azim)):
        spec_elev = abs(output_elevationFFT[peak_loc_azim[i], :])
        peak_val_elev, peak_loc_elev = DOA_BF_PeakDet_loc(spec_elev, radar_config, sidelobeLevel_dB_elev)
        for j in range(len(peak_loc_elev)):
            est_azimuth = np.arcsin(azimuth_bins[peak_loc_azim[i]] / 2 / np.pi / doa_unitDis) / np.pi * 180
            est_elevation = np.arcsin(elevation_bins[peak_loc_elev[j]] / 2 / np.pi / doa_unitDis) / np.pi * 180
            if est_azimuth >= doa_fov_azim[0] and est_azimuth <= doa_fov_azim[1] and est_elevation >= doa_fov_elev[0] and est_elevation <= doa_fov_elev[1]:
                n_obj += 1
                angle_obj.append([est_azimuth, est_elevation, peak_loc_azim[i], peak_loc_elev[j]])
    return angle_obj, output_elevationFFT

def DOA_BF_PeakDet_loc(input_data, radar_config, sidelobeLevel_dB):
    gamma = radar_config['doa_gamma']

    min_val = np.inf
    max_val = -np.inf
    max_loc = 0
    max_data = []
    locate_max = 0
    num_max = 0
    extend_loc = 0
    init_stage = 1
    abs_max_value = 0
    i = 0
    N = len(input_data)

    while (i < (N + extend_loc)):
        i_loc = np.mod(i-1, N)
        current_val = input_data[i_loc]
        # record the maximum abs value
        if current_val > abs_max_value:
            abs_max_value = current_val
        # record the maximum value and loc
        if current_val > max_val:
            max_val = current_val
            max_loc = i_loc
            max_loc_r = i
        # record the minimum value
        if current_val < min_val:
            min_val = current_val

        if locate_max == 1:
            if current_val < max_val / gamma:
                num_max += 1
                bwidth = i - max_loc_r
                max_data.append([max_loc, max_val, bwidth, max_loc_r])
                min_val = current_val
                locate_max = 0
        else:
            if current_val > min_val * gamma:
                locate_max = 1
                max_val = current_val
                max_loc = i_loc
                max_loc_r = i

                if init_stage == 1:
                    extend_loc = i
                    init_stage = 0

        i += 1

    max_data = np.array(max_data)
    if len(max_data.shape) < 2:
        max_data = np.zeros((0, 4))

    max_data = max_data[max_data[:, 1] >= abs_max_value * pow(10, -sidelobeLevel_dB / 10), :]

    peak_val = max_data[:, 1]
    peak_loc = max_data[:, 0].astype('int')
    return peak_val, peak_loc

def single_obj_signal_space_mapping(sig, radar_config):
    '''

    :param sig: 16x12
    :param radar_config:
    :return: sig_space: 86x7
    '''
    tx_position_azimuth = radar_config['tx_position_azimuth']
    tx_position_elevation = radar_config['tx_position_elevation']
    rx_position_azimuth = radar_config['rx_position_azimuth']
    rx_position_elevation = radar_config['rx_position_elevation']

    virtual_position_azimuth = np.tile(np.array(tx_position_azimuth), (len(rx_position_azimuth), 1)) + np.tile(
        np.array(rx_position_azimuth), (len(tx_position_azimuth), 1)).T
    virtual_position_elevation = np.tile(np.array(tx_position_elevation), (len(rx_position_elevation), 1)) + np.tile(
        np.array(rx_position_elevation), (len(tx_position_elevation), 1)).T
    rx_id = np.tile(np.array(range(sig.shape[0])), (sig.shape[1], 1)).T
    tx_id = np.tile(np.array(radar_config['transfer_order']) - 1, (sig.shape[0], 1))

    virtual_position_redundant = np.hstack((virtual_position_azimuth.reshape((-1, 1)), virtual_position_elevation.reshape((-1, 1))))
    rx_tx = np.hstack((rx_id.reshape((-1, 1)), tx_id.reshape((-1, 1))))
    virtual_position_noredundant, virtual_position_index = np.unique(virtual_position_redundant, axis=0, return_index=True)

    sig_space = np.zeros((virtual_position_azimuth.max()+1, virtual_position_elevation.max()+1), dtype='complex128')
    sig_space[virtual_position_noredundant[:, 0], virtual_position_noredundant[:, 1]] = sig[
        rx_tx[virtual_position_index, 0], rx_tx[virtual_position_index, 1]]

    return sig_space

def generate_pointcloud(output_doa):
    '''

    :param output_detection:
    :return:
    '''
    r = np.array(output_doa['range'])
    v = np.array(output_doa['velocity'])
    azimuth = np.array(output_doa['azimuth'])
    elevation = -np.array(output_doa['elevation'])
    intensity = np.array(output_doa['intensity'])

    x = r * np.cos(elevation / 180 * np.pi) * np.sin(azimuth / 180 * np.pi)
    y = r * np.cos(elevation / 180 * np.pi) * np.cos(azimuth / 180 * np.pi)
    z = r * np.sin(elevation / 180 * np.pi)

    pointcloud = np.vstack((x, y, z, v, intensity)).T
    aer = np.vstack((azimuth, elevation, r)).T

    return pointcloud, aer

def radar_pointcloud_generation_FFT_based(adcdata_path, radar_config_path):
    radar_config = load_json(radar_config_path)

    # read data
    adcdata = read_raw_adcdata_from_bin(adcdata_path, radar_config)

    # calibrate raw data
    adcdata = calibrate_raw_adcdata(adcdata, radar_config)

    # rangeFFT
    output_rangeFFT = rangeFFT(adcdata, radar_config)

    # dopplerFFT
    output_dopplerFFT = dopplerFFT(output_rangeFFT, radar_config)

    # range doppler map CFAR
    # doppler correction (w/o vmax extension)
    output_detection = rdm_cfar(output_dopplerFFT, radar_config)

    # DOA
    if output_detection is not None:
        output_doa = doa(output_detection, radar_config)
    else:
        return np.zeros((0, 5)), np.zeros((0, 3))

    # generate pointcloud
    if output_doa is not None:
        pointcloud, aer = generate_pointcloud(output_doa)
        return pointcloud, aer
    else:
        return np.zeros((0, 5)), np.zeros((0, 3))

def radar_preprocess(adcdata_path, radar_config_path):
    pointcloud, aer = radar_pointcloud_generation_FFT_based(adcdata_path, radar_config_path)
    xyz = pointcloud[:, :3]
    other = pointcloud[:, 3:]

    return xyz, aer, other

def get_radar_bins(radar_config_path):
    radar_config = load_json(radar_config_path)

    rangeFFT_size = radar_config['rangeFFT_size']
    range_resolution = radar_config['range_resolution']
    range_bins = range_resolution * (np.arange(rangeFFT_size) + 1)

    dopplerFFT_size = radar_config['dopplerFFT_size']
    velocity_resolution = radar_config['velocity_resolution']
    velocity_bins = velocity_resolution * (np.arange(dopplerFFT_size) - dopplerFFT_size / 2)

    doa_unitDis = radar_config['doa_unitDis']
    azimuthFFT_size = radar_config['azimuthFFT_size']
    azimuthFFT_bins = np.arange(-azimuthFFT_size, azimuthFFT_size, 2) * np.pi / azimuthFFT_size
    azimuth_bins = np.arcsin(azimuthFFT_bins / 2 / np.pi / doa_unitDis) / np.pi * 180

    elevationFFT_size = radar_config['elevationFFT_size']
    elevationFFT_bins = np.arange(-elevationFFT_size, elevationFFT_size, 2) * np.pi / elevationFFT_size
    elevation_bins = np.arcsin(elevationFFT_bins / 2 / np.pi / doa_unitDis) / np.pi * 180
    elevation_bins = -elevation_bins

    return range_bins, velocity_bins, azimuth_bins, elevation_bins

def get_azimuth_elevation_edges(radar_config_path):
    radar_config = load_json(radar_config_path)

    doa_unitDis = radar_config['doa_unitDis']
    azimuthFFT_size = radar_config['azimuthFFT_size']
    azimuth_edges = np.arange(-azimuthFFT_size-1, azimuthFFT_size, 2) * np.pi / azimuthFFT_size
    azimuth_edges = np.arcsin(azimuth_edges / 2 / np.pi / doa_unitDis) / np.pi * 180

    elevationFFT_size = radar_config['elevationFFT_size']
    elevation_edges = np.arange(-elevationFFT_size-1, elevationFFT_size, 2) * np.pi / elevationFFT_size
    elevation_edges = np.arcsin(elevation_edges / 2 / np.pi / doa_unitDis) / np.pi * 180
    elevation_edges = -elevation_edges

    return azimuth_edges, elevation_edges

def get_radar_range_bin_width(radar_config_path):
    radar_config = load_json(radar_config_path)

    return np.array(radar_config['range_resolution'])

def get_radar_velocity_bin_width(radar_config_path):
    radar_config = load_json(radar_config_path)

    return np.array(radar_config['velocity_resolution'])

def get_azimuth_elevation_index_roi(roi_azimuth, roi_elevation, radar_config_path):
    radar_config = load_json(radar_config_path)
    azimuth_min, azimuth_max = roi_azimuth
    elevation_min, elevation_max = roi_elevation
    doa_unitDis = radar_config['doa_unitDis']
    azimuthFFT_size = radar_config['azimuthFFT_size']
    elevationFFT_size = radar_config['elevationFFT_size']

    azimuthFFT_bins = np.arange(-azimuthFFT_size, azimuthFFT_size, 2) * np.pi / azimuthFFT_size
    azimuth_bins = np.arcsin(azimuthFFT_bins / 2 / np.pi / doa_unitDis) / np.pi * 180
    azimuth_bins_roi_mask = np.logical_and(azimuth_bins >= azimuth_min, azimuth_bins <= azimuth_max)
    azimuth_index_roi = np.nonzero(azimuth_bins_roi_mask)[0]

    elevationFFT_bins = np.arange(-elevationFFT_size, elevationFFT_size, 2) * np.pi / elevationFFT_size
    elevation_bins = np.arcsin(elevationFFT_bins / 2 / np.pi / doa_unitDis) / np.pi * 180
    elevation_bins = -elevation_bins
    elevation_bins_roi_mask = np.logical_and(elevation_bins >= elevation_min, elevation_bins <= elevation_max)
    elevation_index_roi = np.nonzero(elevation_bins_roi_mask)[0]

    return azimuth_index_roi, elevation_index_roi

def get_radar_azimuth_elevation_bin_width(roi_azimuth, roi_elevation, radar_config_path):
    radar_config = load_json(radar_config_path)
    azimuth_min, azimuth_max = roi_azimuth
    elevation_min, elevation_max = roi_elevation
    doa_unitDis = radar_config['doa_unitDis']
    azimuthFFT_size = radar_config['azimuthFFT_size']
    elevationFFT_size = radar_config['elevationFFT_size']

    azimuthFFT_bins = np.arange(-azimuthFFT_size, azimuthFFT_size, 2) * np.pi / azimuthFFT_size
    azimuth_bins = np.arcsin(azimuthFFT_bins / 2 / np.pi / doa_unitDis) / np.pi * 180
    azimuth_bins_roi_mask = np.logical_and(azimuth_bins >= azimuth_min, azimuth_bins <= azimuth_max)
    azimuth_index_roi = np.nonzero(azimuth_bins_roi_mask)[0]
    radar_azimuth_bin_width = np.stack((
        azimuth_bins[azimuth_index_roi + 1] - azimuth_bins[azimuth_index_roi],
        azimuth_bins[azimuth_index_roi] - azimuth_bins[azimuth_index_roi-1]
    ))

    elevationFFT_bins = np.arange(-elevationFFT_size, elevationFFT_size, 2) * np.pi / elevationFFT_size
    elevation_bins = np.arcsin(elevationFFT_bins / 2 / np.pi / doa_unitDis) / np.pi * 180
    elevation_bins = -elevation_bins
    elevation_bins_roi_mask = np.logical_and(elevation_bins >= elevation_min, elevation_bins <= elevation_max)
    elevation_index_roi = np.nonzero(elevation_bins_roi_mask)[0]
    radar_elevation_bin_width = np.stack((
        elevation_bins[elevation_index_roi - 1] - elevation_bins[elevation_index_roi],
        elevation_bins[elevation_index_roi] - elevation_bins[elevation_index_roi + 1]
    ))

    return radar_azimuth_bin_width, radar_elevation_bin_width

def multi_obj_signal_space_mapping(sig, radar_config):
    '''

    :param sig: nx16x12
    :param radar_config:
    :return: sig_space: nx86x7
    '''
    tx_position_azimuth = radar_config['tx_position_azimuth']
    tx_position_elevation = radar_config['tx_position_elevation']
    rx_position_azimuth = radar_config['rx_position_azimuth']
    rx_position_elevation = radar_config['rx_position_elevation']
    num_rx = radar_config['num_RX']
    num_tx = radar_config['num_TX']
    transfer_order = radar_config['transfer_order']

    virtual_position_azimuth = np.tile(np.array(tx_position_azimuth), (len(rx_position_azimuth), 1)) + np.tile(
        np.array(rx_position_azimuth), (len(tx_position_azimuth), 1)).T
    virtual_position_elevation = np.tile(np.array(tx_position_elevation), (len(rx_position_elevation), 1)) + np.tile(
        np.array(rx_position_elevation), (len(tx_position_elevation), 1)).T
    rx_id = np.tile(np.arange(num_rx), (len(transfer_order), 1)).T
    tx_id = np.tile(np.array(transfer_order) - 1, (num_rx, 1))

    virtual_position_redundant = np.hstack((virtual_position_azimuth.reshape((-1, 1)), virtual_position_elevation.reshape((-1, 1))))
    rx_tx = np.hstack((rx_id.reshape((-1, 1)), tx_id.reshape((-1, 1))))
    virtual_position_noredundant, virtual_position_index = np.unique(virtual_position_redundant, axis=0, return_index=True)

    sig_space = np.zeros((sig.shape[0], virtual_position_azimuth.max()+1, virtual_position_elevation.max()+1), dtype='complex128')
    sig_space[:, virtual_position_noredundant[:, 0], virtual_position_noredundant[:, 1]] = sig[
        :, rx_tx[virtual_position_index, 0], rx_tx[virtual_position_index, 1]]

    return sig_space

def radar_angle_heatmap_generation(adcdata_path, radar_config_path):
    radar_config = load_json(radar_config_path)

    # read data
    adcdata = read_raw_adcdata_from_bin(adcdata_path, radar_config)

    # calibrate raw data
    adcdata = calibrate_raw_adcdata(adcdata, radar_config)

    # rangeFFT
    output_rangeFFT = rangeFFT(adcdata, radar_config)

    # dopplerFFT
    output_dopplerFFT = dopplerFFT(output_rangeFFT, radar_config)

    # range doppler map doppler correction (w/o vmax extension)
    n_obj = output_dopplerFFT.shape[0] * output_dopplerFFT.shape[1]
    range_index = np.tile(np.arange(output_dopplerFFT.shape[0]), (output_dopplerFFT.shape[1], 1)).T.reshape((-1))
    doppler_index = np.tile(np.arange(output_dopplerFFT.shape[1]), (output_dopplerFFT.shape[0], 1)).reshape((-1))
    range_obj = list(range_index * radar_config['range_resolution'])
    velocity_obj = list((doppler_index - output_dopplerFFT.shape[1] / 2) * radar_config['velocity_resolution'])
    if radar_config['apply_vmax_extend'] == 1:
        # doppler phase correction due to TDM MIMO with applying vmax extention algorithm

        # mask_for_apply_vmax_extend = np.array(range_obj) >= radar_config['min_range_for_apply_vmax_extend']
        mask_v_positive = np.array(velocity_obj) > 0
        mask_v_unpositive = np.logical_not(mask_v_positive)

        num_tx = radar_config['num_TX']
        dopplerFFT_size = radar_config['dopplerFFT_size']
        doppler_index_unwrap = np.tile(doppler_index, (num_tx, 1)).T + dopplerFFT_size * (
                np.tile(np.arange(num_tx) - num_tx / 2, (n_obj, 1)) + np.tile(mask_v_unpositive, (num_tx, 1)).T)

        sig = output_dopplerFFT[range_index, doppler_index, :, :]

        # Doppler phase correction due to TDM MIMO
        delta_phi = 2 * np.pi * (doppler_index_unwrap - dopplerFFT_size / 2) / (num_tx * dopplerFFT_size)

        # construct all possible signal vectors based on the number of possible hypothesis
        tx = np.arange(num_tx)
        correct_matrix = np.exp(-1j * np.matmul(delta_phi.reshape((delta_phi.shape[0], delta_phi.shape[1], 1)),
                                                np.tile(tx, (n_obj, 1, 1))))
        sig_correct_all = np.matmul(
            np.expand_dims(sig.swapaxes(1, 2), len(sig.shape)),
            np.expand_dims(correct_matrix.swapaxes(1, 2), len(correct_matrix.shape)).swapaxes(2, 3)
        ).swapaxes(1, 2)

        # find the overlap antenna ID that can be used for phase compensation
        overlap_antenna_info_1tx = get_overlap_antenna_info(radar_config)

        # use overlap antenna to do max velocity unwrap
        sig_associated = sig[:, overlap_antenna_info_1tx[:, 0], overlap_antenna_info_1tx[:, 1]]
        sig_overlaped = sig[:, overlap_antenna_info_1tx[:, 2], overlap_antenna_info_1tx[:, 3]]
        # check the phase difference of each overlap antenna pair for each hypothesis
        angle_sum_test = np.zeros((n_obj, overlap_antenna_info_1tx.shape[0], delta_phi.shape[1]))
        for i_sig in range(angle_sum_test.shape[1]):
            signal2 = np.matmul(
                np.expand_dims(sig_overlaped[:, :i_sig + 1], len(sig_overlaped.shape)),
                np.exp(-1j * np.expand_dims(delta_phi, 1))
            )
            signal1 = np.tile(np.expand_dims(sig_associated[:, :i_sig + 1], len(sig_associated.shape)),
                              (1, 1, signal2.shape[2]))
            angle_sum_test[:, i_sig, :] = np.angle(np.sum(signal1 * signal2.conj(), axis=1))

        # chosee the hypothesis with minimum phase difference to estimate the unwrap factor
        doppler_unwrap_integ_overlap_val = np.min(np.abs(angle_sum_test), axis=2)
        doppler_unwrap_integ_overlap_index = np.argmin(np.abs(angle_sum_test), axis=2)

        # test the angle FFT SNR
        rx_tx_noredundant_row1 = get_rx_tx_noredundant_row1(radar_config)
        sig_correct_row1 = sig_correct_all[:, rx_tx_noredundant_row1[:, 0], rx_tx_noredundant_row1[:, 1], :]
        angleFFT_size = radar_config['angleFFT_size']
        sig_correct_row1_azimuthFFT = np.fft.fftshift(
            np.fft.fft(sig_correct_row1, n=angleFFT_size, axis=1),
            axes=1
        )
        angle_bin_skip_left = radar_config['angle_bin_skip_left']
        angle_bin_skip_right = radar_config['angle_bin_skip_right']
        sig_correct_row1_azimuthFFT_abs_cut = np.abs(
            sig_correct_row1_azimuthFFT[:, angle_bin_skip_left - 1:angleFFT_size - angle_bin_skip_right - 1, :])
        doppler_unwrap_integ_FFT_max_index = np.argmax(np.max(sig_correct_row1_azimuthFFT_abs_cut, axis=1), axis=1)

        doppler_unwrap_integ_index = [np.argmax(np.bincount(doppler_unwrap_integ_overlap_index[i, :])) for i in
                                      range(doppler_unwrap_integ_overlap_index.shape[0])]
        doppler_unwrap_integ_index = np.array(doppler_unwrap_integ_index)

        # overlap antenna method is applied by default
        sig_correct = sig_correct_all[np.arange(n_obj), :, :, doppler_unwrap_integ_index]

        # corret velocity after applying the integer value
        doppler_index_unwraped = doppler_index_unwrap[np.arange(n_obj), doppler_unwrap_integ_index]
        doppler_index_FFT = doppler_index_unwrap[np.arange(n_obj), doppler_unwrap_integ_FFT_max_index]

    else:
        # doppler phase correction due to TDM MIMO without applying vmax extention algorithm
        sig = output_dopplerFFT[range_index, doppler_index, :, :]

        delta_phi = 2 * np.pi * (doppler_index - output_dopplerFFT.shape[1] / 2) / (
                output_dopplerFFT.shape[3] * output_dopplerFFT.shape[1])
        delta_phi = np.tile(delta_phi, (1, 1)).T

        tx = np.array(range(output_dopplerFFT.shape[3]))
        tx = np.tile(tx, (1, 1))

        correct_matrix = np.exp(-1j * np.matmul(delta_phi, tx))
        correct_matrix = np.tile(correct_matrix, (output_dopplerFFT.shape[2], 1, 1))
        correct_matrix = correct_matrix.swapaxes(0, 1)

        sig_correct = sig * correct_matrix

    azimuthFFT_size = radar_config['azimuthFFT_size']
    elevationFFT_size = radar_config['elevationFFT_size']
    sig_space = multi_obj_signal_space_mapping(sig_correct, radar_config)
    output_azimuthFFT = np.fft.fftshift(np.fft.fft(sig_space, n=azimuthFFT_size, axis=1), axes=1)
    output_elevationFFT = np.fft.fftshift(np.fft.fft(output_azimuthFFT, n=elevationFFT_size, axis=2), axes=2)

    heatmap = np.abs(output_elevationFFT).reshape((
        output_dopplerFFT.shape[0], output_dopplerFFT.shape[1], azimuthFFT_size, elevationFFT_size
    ))

    return heatmap



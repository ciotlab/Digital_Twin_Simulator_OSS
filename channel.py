import yaml
import numpy as np
import torch
from einops import rearrange
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .simulator.ray import RayGroup
from .simulator.ue import UEGroup
from .simulator.background import Background


class Channel:
    @torch.no_grad()
    def __init__(self, scenario_name, simulation_name, OFDM_params, array_params, device='cpu'):
        self.background = Background.load(scenario_name=scenario_name, simulation_name=simulation_name)
        self.bs_name_list = self.background.get_bs_name_list()
        self.bs_attitude_dict = self.background.get_bs_attitude_dict()
        self.ue = UEGroup.load(scenario_name=scenario_name, simulation_name=simulation_name)
        self.ue_name_list = self.ue.get_ue_name_list()
        self.ue_time_idx_dict = self.ue.get_ue_time_idx_dict()
        self.ray = RayGroup.load(scenario_name=scenario_name, simulation_name=simulation_name)
        self.wavelength = self.ray.wavelength
        self.device = device
        # OFDM parameters
        self.fft_size = OFDM_params['fft_size']
        self.subcarrier_spacing = OFDM_params['subcarrier_spacing']
        self.sampling_rate = self.fft_size * self.subcarrier_spacing
        start = -self.fft_size / 2 if self.fft_size % 2 == 0 else -(self.fft_size - 1) / 2
        limit = self.fft_size / 2 if self.fft_size % 2 == 0 else (self.fft_size - 1) / 2 + 1
        self._frequencies = torch.arange(start=start, end=limit).to(device) * self.subcarrier_spacing
        # Array parameters
        self.array_params = array_params
        self.array_position = {}
        for n in array_params:
            position = None
            if array_params[n]['type'] == 'array':
                position = torch.tensor(array_params[n]['position'], device=device).to(torch.float64)
            elif array_params[n]['type'] == 'planar':
                num_rows = array_params[n]['num_rows']
                num_cols = array_params[n]['num_cols']
                vertical_spacing = array_params[n]['vertical_spacing']
                horizontal_spacing = array_params[n]['horizontal_spacing']
                row_coord = (torch.arange(start=0, end=num_rows).to(device) - (num_rows - 1) / 2) * vertical_spacing
                col_coord = (torch.arange(start=0, end=num_cols).to(device) - (num_cols - 1) / 2) * horizontal_spacing
                yz = torch.stack(tensors=(col_coord[:, None].expand(-1, num_rows),
                                          row_coord[None, :].expand(num_cols, -1)), dim=2)
                yz = torch.flatten(yz, start_dim=0, end_dim=1)
                position = torch.concatenate(tensors=(torch.zeros((yz.shape[0], 1)).to(device), yz), dim=1).to(torch.float64)
            self.array_position[n] = position * torch.tensor(self.wavelength).to(device).to(torch.float64)
        bs_attitude_list = list(self.bs_attitude_dict.items())
        bs_attitude = torch.tensor(np.stack([x[1] for x in bs_attitude_list], axis=0), device=device)
        bs_rot_mat = self._rotation_matrix(bs_attitude)
        bs_array = torch.einsum('brc, ac -> bar', bs_rot_mat, self.array_position['bs'])  # bs, ant, coord
        self._bs_array_dict = {}
        for idx, (name, _) in enumerate(bs_attitude_list):
            self._bs_array_dict[name] = bs_array[idx, :, :]

    @torch.no_grad()
    def get_sparse_domain_channel(self, bs_name_list, ue_name_list, time_idx_range):
        freq_coef, array_coef, data = self.get_coef(bs_name_list, ue_name_list, time_idx_range)
        a = data['a']  # bs, ue, time, path
        freq_coef = torch.fft.ifft(torch.fft.ifftshift(freq_coef, dim=-1), dim=-1, norm='ortho')  # bs, ue, time, path, delay
        if self.array_params['bs']['type'] == 'planar':
            num_rows = self.array_params['bs']['num_rows']
            num_cols = self.array_params['bs']['num_cols']
            array_coef = rearrange(array_coef, 'b u t p (bac bar) ua -> b u t p bac bar ua', bac=num_cols, bar=num_rows)  # bs, ue, time, path, bs_ant_col, bs_ant_row, ue_ant
            array_coef = torch.fft.fftshift(torch.fft.fft(array_coef, dim=-3, norm='ortho'), dim=-3)  # column fft
            array_coef = torch.fft.fftshift(torch.fft.fft(array_coef, dim=-2, norm='ortho'), dim=-2)  # row fft
        else:
            raise Exception('BS should be a planar array.')
        ch = torch.einsum('butp, butpf, butpijk -> butijkf', a, freq_coef, array_coef)  # bs, ue, time, bs_ant_col(i), bs_ant_row(j), ue_ant(k), freq
        return ch, data

    @torch.no_grad()
    def plot_sparce_domain_channel(self, ch, data, start_time_idx, time_idx_step, n_time_plot_row, n_time_plot_col, scale=3):
        ch = ch[0, 0, :, :, 0, 0, :]  # time, azimuth, delay
        ch = 20 * torch.log10(torch.abs(ch)).cpu().numpy()
        num_plot = n_time_plot_row * n_time_plot_col
        time_index_list = list(np.arange(start_time_idx, start_time_idx + time_idx_step * num_plot, time_idx_step))
        time = data['time']
        time_list = [time[i].cpu().numpy().item() for i in time_index_list]
        fig = plt.figure(figsize=(scale * n_time_plot_col, scale * n_time_plot_row))
        gs = GridSpec(nrows=n_time_plot_row, ncols=n_time_plot_col)
        for idx, (t_idx, t) in enumerate(zip(time_index_list, time_list)):
            ax = fig.add_subplot(gs[idx])
            ax.set_title(f'{t}')
            c = ch[t_idx, :, :]
            ax.pcolor(c, shading='auto')
        plt.show()

    @torch.no_grad()
    def get_channel(self, bs_name_list, ue_name_list, time_idx_range):
        freq_coef, array_coef, data = self.get_coef(bs_name_list, ue_name_list, time_idx_range)
        a = data['a']  # bs, ue, time, path
        ch = torch.einsum('butp, butpf, butpij -> butijf', a, freq_coef, array_coef)  # bs, ue, time, bs_ant(i), ue_ant(j), freq
        return ch, data

    @torch.no_grad()
    def get_coef(self, bs_name_list, ue_name_list, time_idx_range):
        data = self.ray.get_data_tensor(bs_name_list, ue_name_list, time_idx_range)  # bs, ue, time, path
        for k in data:
            if k not in ['bs_name_list', 'ue_name_list']:
                data[k] = torch.tensor(data[k], device=self.device)
        freq_coef = self._get_freq_coefficient(tau=data['tau'])  # bs, ue, time, path, frequency
        array_coef = self._get_array_coefficient(bs_name_list=data['bs_name_list'], ue_att=data['ue_att'],
                                                 phi_r=data['phi_r'], phi_t=data['phi_t'], theta_r=data['theta_r'],
                                                 theta_t=data['theta_t'])  # bs, ue, time, path, bs_ant, ue_ant
        return freq_coef, array_coef, data

    @torch.no_grad()
    def _get_freq_coefficient(self, tau):
        freq_coef = torch.exp(- 2j * np.pi * self._frequencies[None, None, None, None, :] * tau[:, :, :, :, None])  # bs, ue, time, path, frequency
        return freq_coef

    @torch.no_grad()
    def _get_array_coefficient(self, bs_name_list, ue_att, phi_r, phi_t, theta_r, theta_t):
        bs_array = []
        for name in bs_name_list:
            bs_array.append(self._bs_array_dict[name])
        bs_array = torch.stack(bs_array, dim=0)  # bs, bs_ant, coord
        ue_rot_mat = self._rotation_matrix(ue_att)  # ue, time, row, col
        ue_array = torch.einsum('utrc, ac -> utar', ue_rot_mat, self.array_position['ue'])  # ue, time, ue_ant, coord
        k_bs = torch.stack(tensors=[torch.sin(theta_t) * torch.cos(phi_t),
                                    torch.sin(theta_t) * torch.sin(phi_t),
                                    torch.cos(theta_t)], dim=-1)  # bs, ue, time, path, coord
        k_ue = torch.stack(tensors=[torch.sin(theta_r) * torch.cos(phi_r),
                                    torch.sin(theta_r) * torch.sin(phi_r),
                                    torch.cos(theta_r)], dim=-1)  # bs, ue, time, path, coord
        bs_phase_shifts = torch.einsum('bic, butpc -> butpi', bs_array, k_bs)  # bs, ue, time, path, bs_ant(i)
        ue_phase_shifts = torch.einsum('utjc, butpc -> butpj', ue_array, k_ue)  # bs, ue, time, path, ue_ant(j)
        phase_shifts = bs_phase_shifts[:, :, :, :, :, None] + ue_phase_shifts[:, :, :, :, None, :]  # bs, ue, time, path, bs_ant, ue_ant
        phase_shifts = 2 * np.pi * phase_shifts / self.wavelength
        array_coef = torch.exp(1j * phase_shifts)
        return array_coef

    @torch.no_grad()
    def _rotation_matrix(self, angles):
        # angles : (z, y, x) intrinsic rotation angle
        a = angles[..., 0]
        b = angles[..., 1]
        c = angles[..., 2]
        cos_a = torch.cos(a)
        cos_b = torch.cos(b)
        cos_c = torch.cos(c)
        sin_a = torch.sin(a)
        sin_b = torch.sin(b)
        sin_c = torch.sin(c)

        r_11 = cos_a * cos_b
        r_12 = cos_a * sin_b * sin_c - sin_a * cos_c
        r_13 = cos_a * sin_b * cos_c + sin_a * sin_c
        r_1 = torch.stack((r_11, r_12, r_13), dim=-1)

        r_21 = sin_a * cos_b
        r_22 = sin_a * sin_b * sin_c + cos_a * cos_c
        r_23 = sin_a * sin_b * cos_c - cos_a * sin_c
        r_2 = torch.stack((r_21, r_22, r_23), dim=-1)

        r_31 = -sin_b
        r_32 = cos_b * sin_c
        r_33 = cos_b * cos_c
        r_3 = torch.stack((r_31, r_32, r_33), dim=-1)

        rot_mat = torch.stack((r_1, r_2, r_3), dim=-2)
        return rot_mat


if __name__ == '__main__':
    device = 'cuda:0'
    #OFDM_params = {'fft_size': 64, 'subcarrier_spacing': 15e3, 'normalize_delay': False}
    #array_params = {'bs': {'type': 'planar', 'num_rows': 4, 'num_cols': 8,
    #                       'vertical_spacing': 0.5, 'horizontal_spacing': 0.5},
    #                'ue': {'type': 'array', 'position': [[0, 0.25, 0], [0, -0.25, 0]]}}
    OFDM_params = {'fft_size': 256, 'subcarrier_spacing': 120e3}
    array_params = {'bs': {'type': 'planar', 'num_rows': 1, 'num_cols': 128,
                           'vertical_spacing': 0.5, 'horizontal_spacing': 0.5},
                    'ue': {'type': 'array', 'position': [[0, 0, 0]]}}
    ch = Channel(scenario_name='Suwon', simulation_name='sim_1', OFDM_params=OFDM_params, array_params=array_params,
                 device=device)
    # bs_name_list = ['bs0', 'bs1']
    bs_name_list = ['bs9']
    # bs_name_list = None
    # ue_name_list = ['ue0', 'ue1']
    ue_name_list = ['ue1']
    # ue_name_list = None
    time_range = (0.0, 300.0)
    #c = ch.get_channel(bs_name_list, ue_name_list, time_range)
    c, data = ch.get_sparse_domain_channel(bs_name_list, ue_name_list, time_range)
    ch.plot_sparce_domain_channel(c, data, start_time_idx=0, time_idx_step=11, n_time_plot_row=10, n_time_plot_col=10, scale=3)
    pass


import yaml
import numpy as np
from einops import rearrange
import mitsuba as mi
import drjit as dr
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, AntennaArray, Antenna
from pathlib import Path
from tqdm import tqdm
from ray import RayGroup
from ue import UEGroup
from background import Background


class RaySimulator:
    def __init__(self, scenario_name, simulation_name):
        self.scenario_name = scenario_name
        self.simulation_name = simulation_name
        self.scenario_dir = Path(__file__).parents[1] / 'scenario' / scenario_name
        self.sionna_dir = self.scenario_dir / 'sionna'
        self.simulation_dir = self.scenario_dir / 'simulation' / simulation_name
        with open(self.simulation_dir / 'config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)['ray']
        self.frequency = float(self.config['frequency'])
        speed_of_light = 299792458.0
        self.wavelength = speed_of_light / self.frequency
        self.ue = UEGroup.load(scenario_name=scenario_name, simulation_name=simulation_name)
        self.background = Background()
        # Construct scene
        self.scene = load_scene(str(self.sionna_dir / f"{scenario_name}.xml"))
        self.scene.frequency = self.frequency
        self.scene.synthetic_array = True
        self.scene.tx_array = AntennaArray(antenna=Antenna(pattern=self.config['tx_antenna_pattern'],
                                                           polarization=self.config['tx_antenna_polarization']),
                                           positions=np.array([[0.0, 0.0, 0.0]]))
        self.scene.rx_array = AntennaArray(antenna=Antenna(pattern=self.config['rx_antenna_pattern'],
                                                           polarization=self.config['rx_antenna_polarization']),
                                           positions=np.array([[0.0, 0.0, 0.0]]))
        with open(self.scenario_dir / 'config.yaml', 'r') as f:
            self.bs_config = yaml.safe_load(f)['bs']
        for idx, (position, direction, tilt) in enumerate(zip(self.bs_config['position'], self.bs_config['direction'],
                                                              self.bs_config['tilt'])):
            name = 'bs' + str(idx)
            position = np.array(position)
            orientation = np.array([direction / 180.0 * np.pi, tilt / 180.0 * np.pi, 0.0])
            self.background.append_bs(name=name, position=position, attitude=orientation)

    def extract_mesh(self):
        scene = mi.load_file(str(self.sionna_dir / f"{self.scenario_name}.xml"))
        shapes = scene.shapes()
        vertices, faces, albedos = [], [], []
        f_offset = 0
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = mi.Vector3f(0, 0, 1)
        for i, s in enumerate(shapes):
            n_vertices = s.vertex_count()
            v = s.vertex_position(dr.arange(mi.UInt32, n_vertices))
            vertices.append(v.numpy())
            f = s.face_indices(dr.arange(mi.UInt32, s.face_count()))
            faces.append(f.numpy() + f_offset)
            f_offset += n_vertices
            albedo = s.bsdf().eval_diffuse_reflectance(si).numpy()
            albedos.append(np.tile(albedo, (n_vertices, 1)))
        vertices, faces, albedos = np.concatenate(vertices, axis=0), np.concatenate(faces, axis=0), np.concatenate(albedos, axis=0)
        self.background.set_mesh(vertices, faces, albedos)

    def save_background(self, file_name='background.pkl'):
        self.extract_mesh()
        self.background.save(self.scenario_name, self.simulation_name, file_name)

    def run(self, bs_name_list=None, ue_name_list=None, file_mode='single'):
        if bs_name_list is None:
            bs_name_list = list(self.background.bs.keys())
        if ue_name_list is None:
            ue_name_list = list(self.ue.get_ue_name_list())
        ray_group = RayGroup(self.scenario_name, self.simulation_name, bs_name_list, ue_name_list)
        ray_group.time = self.ue.time
        ray_group.wavelength = self.wavelength
        if file_mode == 'single':
            for bs_name in tqdm(bs_name_list, desc="Progress", total=len(bs_name_list)):
                self.run_single_bs(ray_group, bs_name, ue_name_list)
            ray_group.save_ray(file_name='ray.pkl')
            ray_group.save()
        elif file_mode == 'per_ue':
            with tqdm(total=len(bs_name_list) * len(ue_name_list), desc="Progress") as pbar:
                for ue_name in ue_name_list:
                    for bs_name in bs_name_list:
                        self.run_single_bs(ray_group, bs_name, ue_name_list=[ue_name])
                        pbar.update(1)
                    ray_group.save_ray(file_name=f'ray_{ue_name}.pkl')
                    ray_group.clear_rays()
            ray_group.save()

    def run_single_bs(self, ray_group, bs_name, ue_name_list=None):
        # Add BS to Sionna
        bs_position = self.background.bs[bs_name].position
        bs_orientation = self.background.bs[bs_name].attitude
        bs = Transmitter(name=bs_name, position=bs_position, orientation=bs_orientation)
        self.scene.add(bs)
        # Load UE
        ue_name, ue_num_time, ue_time, ue_time_index, ue_position, ue_attitude = self.ue.get_data(ue_name_list)
        ue_position = np.concatenate(ue_position, axis=0)
        ue_attitude = np.concatenate(ue_attitude, axis=0)
        num_step = np.ceil(ue_position.shape[0] / self.config['max_rx_positions'])
        ue_position_list = np.array_split(ue_position, num_step)
        ue_attitude_list = np.array_split(ue_attitude, num_step)
        # Run Sionna simulation
        mask, a, phi_r, phi_t, theta_r, theta_t, tau, objects, vertices = [], [], [], [], [], [], [], [], []
        for ue_pos, ue_att in zip(ue_position_list, ue_attitude_list):
            for idx, (pos, att) in enumerate(zip(ue_pos, ue_att)):
                self.scene.add(Receiver(name="ue" + str(idx), position=pos, orientation=att))
            path = self.scene.compute_paths(max_depth=self.config['max_depth'],
                                            num_samples=float(self.config['num_samples']),
                                            diffraction=self.config['diffraction'],
                                            edge_diffraction=self.config['edge_diffraction'],
                                            scattering=self.config['scattering'],
                                            scat_keep_prob=self.config['scat_keep_prob'],
                                            scat_random_phases=self.config['scat_random_phases'],
                                            check_scene=False)
            path.normalize_delays = False
            mask.append(path.mask.numpy()[0, :, 0, :])  # rx, path
            a_tmp, tau_tmp = path.cir()
            a.append(rearrange(a_tmp.numpy(), 'b r ra t ta p s -> b s r t p ra ta')[0, 0, :, 0, :, 0, 0])  # rx, path
            tau.append(tau_tmp.numpy()[0, :, 0, :])  # rx, path
            phi_r.append(path.phi_r.numpy()[0, :, 0, :])  # rx, path
            phi_t.append(path.phi_t.numpy()[0, :, 0, :])  # rx, path
            theta_r.append(path.theta_r.numpy()[0, :, 0, :])  # rx, path
            theta_t.append(path.theta_t.numpy()[0, :, 0, :])  # rx, path
            objects.append(rearrange(path.objects.numpy(), 'd r t p -> r t p d')[:, 0, :, :])  # rx, path, depth
            vertices.append(rearrange(path.vertices.numpy(), 'd r t p c -> r t p d c')[:, 0, :, :, :])  # rx, path, depth, coordinate
            for idx in range(ue_position.shape[0]):
                self.scene.remove("ue" + str(idx))
            self.scene.remove("paths")
        self.scene.remove(bs_name)
        # Save simulation results
        max_num_path = max([m.shape[1] for m in mask])
        for i in range(len(mask)):
            pl = 2 * max_num_path - mask[i].shape[1]
            mask[i] = np.pad(mask[i], pad_width=((0, 0), (0, pl)), constant_values=False)
            a[i] = np.pad(a[i], pad_width=((0, 0), (0, pl)), constant_values=0.0)
            phi_r[i] = np.pad(phi_r[i], pad_width=((0, 0), (0, pl)), constant_values=0.0)
            phi_t[i] = np.pad(phi_t[i], pad_width=((0, 0), (0, pl)), constant_values=0.0)
            theta_r[i] = np.pad(theta_r[i], pad_width=((0, 0), (0, pl)), constant_values=0.0)
            theta_t[i] = np.pad(theta_t[i], pad_width=((0, 0), (0, pl)), constant_values=0.0)
            tau[i] = np.pad(tau[i], pad_width=((0, 0), (0, pl)), constant_values=-1.0)
            dl = self.config['max_depth'] - objects[i].shape[2]
            objects[i] = np.pad(objects[i], pad_width=((0, 0), (0, pl), (0, dl)), constant_values=-1)
            vertices[i] = np.pad(vertices[i], pad_width=((0, 0), (0, pl), (0, dl), (0, 0)), constant_values=0)
        mask, a, phi_r, phi_t, theta_r, theta_t, tau, objects, vertices = (
            np.concatenate(mask, axis=0), np.concatenate(a, axis=0), np.concatenate(phi_r, axis=0),
            np.concatenate(phi_t, axis=0), np.concatenate(theta_r, axis=0), np.concatenate(theta_t, axis=0),
            np.concatenate(tau, axis=0), np.concatenate(objects, axis=0), np.concatenate(vertices, axis=0))
        for ue_name, l, time_slice, time_index_slice in zip(ue_name, ue_num_time, ue_time, ue_time_index):
            mask_slice, mask = np.split(mask, indices_or_sections=[l], axis=0)
            a_slice, a = np.split(a, indices_or_sections=[l], axis=0)
            phi_r_slice, phi_r = np.split(phi_r, indices_or_sections=[l], axis=0)
            phi_t_slice, phi_t = np.split(phi_t, indices_or_sections=[l], axis=0)
            theta_r_slice, theta_r = np.split(theta_r, indices_or_sections=[l], axis=0)
            theta_t_slice, theta_t = np.split(theta_t, indices_or_sections=[l], axis=0)
            tau_slice, tau = np.split(tau, indices_or_sections=[l], axis=0)
            objects_slice, objects = np.split(objects, indices_or_sections=[l], axis=0)
            vertices_slice, vertices = np.split(vertices, indices_or_sections=[l], axis=0)
            ue_position = self.ue.ue[ue_name].position
            ue_attitude = self.ue.ue[ue_name].attitude
            res_dict = self.preprocess(mask=mask_slice, a=a_slice, phi_r=phi_r_slice, phi_t=phi_t_slice,
                                       theta_r=theta_r_slice, theta_t=theta_t_slice, tau=tau_slice,
                                       objects=objects_slice, vertices=vertices_slice,
                                       score_thresh=self.config['preprocess_score_thresh'])
            ray_group.append_ray(name=(bs_name, ue_name), time=time_slice, time_index=time_index_slice,
                                 ue_pos=ue_position, ue_att=ue_attitude,
                                 bs_pos=bs_position, bs_att=bs_orientation, **res_dict)

    def preprocess(self, mask, a, phi_r, phi_t, theta_r, theta_t, tau, objects, vertices, score_thresh=10.0):
        path_index = np.ones_like(mask) * -1
        cur_path_index = 0
        for p in range(mask.shape[1]):
            if mask[0, p]:
                path_index[0, p] = cur_path_index
                cur_path_index += 1
        for t in range(1, mask.shape[0]):
            prev_msk = mask[t - 1]
            msk = mask[t]
            prev_obj = objects[t - 1, prev_msk]
            obj = objects[t, msk]
            prev_vert = vertices[t - 1, prev_msk]
            prev_vert[np.broadcast_to(prev_obj[:, :, None], prev_vert.shape) < 0] = 0
            vert = vertices[t, msk]
            vert[np.broadcast_to(obj[:, :, None], vert.shape) < 0] = 0
            score = np.mean(np.sum(np.square(prev_vert[:, None, :, :] - vert[None, :, :, :]), axis=3), axis=2)
            obj_equality = np.all((prev_obj[:, None, :] >= 0) == (obj[None, :, :] >= 0), axis=-1)
            score[~obj_equality] = np.inf
            score[score > score_thresh] = np.inf
            mapping = np.zeros((len(msk),), dtype=int)
            exist_idx = list(np.nonzero(msk)[0])
            non_exist_idx = list(np.nonzero(~msk)[0])
            prev_exist_idx = list(np.nonzero(prev_msk)[0])
            prev_non_exist_idx = list(np.nonzero(~prev_msk)[0])
            for i, ei in enumerate(exist_idx):
                if prev_exist_idx and (not np.isinf(np.min(score[:, i]))):
                    mi = np.argmin(score[:, i])
                    mapping[ei] = prev_exist_idx.pop(mi)
                    score = np.delete(score, mi, axis=0)
                    path_index[t, mapping[ei]] = path_index[t - 1, mapping[ei]]
                else:
                    mapping[ei] = prev_non_exist_idx.pop(0)
                    path_index[t, mapping[ei]] = cur_path_index
                    cur_path_index += 1
            prev_idx = sorted(prev_exist_idx + prev_non_exist_idx)
            for nei in non_exist_idx:
                mapping[nei] = prev_idx.pop(0)
            mask[t, mapping] = mask[t].copy()
            a[t, mapping] = a[t].copy()
            phi_r[t, mapping] = phi_r[t].copy()
            phi_t[t, mapping] = phi_t[t].copy()
            theta_r[t, mapping] = theta_r[t].copy()
            theta_t[t, mapping] = theta_t[t].copy()
            tau[t, mapping] = tau[t].copy()
            objects[t, mapping] = objects[t].copy()
            vertices[t, mapping] = vertices[t].copy()
        time_loc = np.repeat(np.arange(mask.shape[0])[:, np.newaxis], mask.shape[1], axis=1)
        num_path = np.sum(mask.astype(int), axis=1)
        path_loc = []
        for n in num_path:
            path_loc.append(np.arange(n))
        path_loc = np.concatenate(path_loc, axis=0)
        mask = mask.flatten()
        return {'num_path': num_path, 'time_loc': time_loc.flatten()[mask], 'path_loc': path_loc,
                'path_index': path_index.flatten()[mask],
                'a': a.flatten()[mask],
                'phi_r': phi_r.flatten()[mask], 'phi_t': phi_t.flatten()[mask],
                'theta_r': theta_r.flatten()[mask], 'theta_t': theta_t.flatten()[mask],
                'tau': tau.flatten()[mask],
                'objects': objects.reshape(-1, objects.shape[-1])[mask],
                'vertices': vertices.reshape(-1, *vertices.shape[-2:])[mask]}


if __name__ == '__main__':
    scenario_name = 'Suwon'
    simulation_name = 'sim_1'
    rs = RaySimulator(scenario_name, simulation_name)
    rs.save_background()
    #bs_name_list = ['bs0', 'bs1', 'bs2', 'bs3']
    bs_name_list = None
    #ue_name_list = ['ue0', 'ue1', 'ue2']
    ue_name_list = None
    rs.run(bs_name_list=bs_name_list, ue_name_list=ue_name_list, file_mode='per_ue')



        

        
    

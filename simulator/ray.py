import numpy as np
import dill
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pythreejs as p3s


class RayGroup:
    def __init__(self, scenario_name, simulation_name, bs_name_list=None, ue_name_list=None):
        self._scenario_name = scenario_name
        self._simulation_name = simulation_name
        self.bs_name_list = bs_name_list
        self.ue_name_list = ue_name_list
        self.time = np.empty(0)
        self.wavelength = None
        self._rays = {}
        self._ray_files = None
        self._ray_time_index_start = None
        self._ray_time_index_end = None
        self._ray_max_num_path = None
        self._ray_dir = (Path(__file__).parents[1] / 'scenario' / scenario_name / 'simulation' /
                         simulation_name / 'ray')
        Path.mkdir(self._ray_dir, exist_ok=True)
        if bs_name_list is not None and ue_name_list is not None:
            self._ray_files = np.zeros(shape=(len(bs_name_list), len(ue_name_list)), dtype='<S256')
            self._ray_time_index_start = np.zeros(shape=(len(bs_name_list), len(ue_name_list)), dtype=int)
            self._ray_time_index_end = np.zeros(shape=(len(bs_name_list), len(ue_name_list)), dtype=int)
            self._ray_max_num_path = np.zeros(shape=(len(bs_name_list), len(ue_name_list)), dtype=int)
        self._anim_bs_name_list = None
        self._anim_ue_name_list = None

    def get_data_tensor(self, bs_name_list=None, ue_name_list=None, time_idx_range=None):
        bs_name_list = bs_name_list if bs_name_list is not None else self.bs_name_list
        ue_name_list = ue_name_list if ue_name_list is not None else self.ue_name_list
        if time_idx_range is None:
            time_idx_start = 0
            time_idx_end = len(self.time)
        else:
            time_idx_start, time_idx_end = time_idx_range
        start_list = []
        end_list = []
        max_num_path = 1
        ray_file_dict = {}
        for bs_loc, bs_name in enumerate(bs_name_list):
            for ue_loc, ue_name in enumerate(ue_name_list):
                bs_idx, ue_idx = self.bs_name_list.index(bs_name), self.ue_name_list.index(ue_name)
                start = self._ray_time_index_start[bs_idx, ue_idx]
                end = self._ray_time_index_end[bs_idx, ue_idx]
                start_list.append(start)
                end_list.append(end)
                num_path = self._ray_max_num_path[bs_idx, ue_idx]
                max_num_path = max(max_num_path, num_path)
                file = self._ray_files[bs_idx, ue_idx].decode('UTF-8')
                if file in ray_file_dict:
                    ray_file_dict[file].append((bs_name, ue_name, bs_loc, ue_loc))
                else:
                    ray_file_dict[file] = [(bs_name, ue_name, bs_loc, ue_loc)]
        time_idx_start = max(time_idx_start, min(start_list))
        time_idx_end = min(time_idx_end, max(end_list))
        time_idx_range = (time_idx_start, time_idx_end)
        time_idx = np.arange(time_idx_start, time_idx_end)
        time_len = time_idx_end - time_idx_start
        n_bs, n_ue = len(bs_name_list), len(ue_name_list)
        ue_pos = np.full(shape=(n_ue, time_len, 3), fill_value=0.0)
        ue_att = np.full(shape=(n_ue, time_len, 3), fill_value=0.0)
        ue_mask = np.full(shape=(n_ue, time_len), fill_value=False)
        shape = (n_bs, n_ue, time_len, max_num_path)
        mask = np.full(shape=shape, fill_value=False)
        a = np.full(shape=shape, fill_value=0.0 + 0.0j)
        phi_r = np.full(shape=shape, fill_value=0.0)
        phi_t = np.full(shape=shape, fill_value=0.0)
        theta_r = np.full(shape=shape, fill_value=0.0)
        theta_t = np.full(shape=shape, fill_value=0.0)
        tau = np.full(shape=shape, fill_value=0.0)
        max_num_path = 1
        for file_name in ray_file_dict:
            self.load_ray(file_name)
            for bs_name, ue_name, bs_idx, ue_idx in ray_file_dict[file_name]:
                x = self._rays[(bs_name, ue_name)].get_data_tensor(time_idx_range)
                num_path = x['mask'].shape[1]
                max_num_path = max(max_num_path, num_path)
                ue_pos[ue_idx, :, :] = x['ue_pos']
                ue_att[ue_idx, :, :] = x['ue_att']
                ue_mask[ue_idx, :] = x['ue_mask']
                mask[bs_idx, ue_idx, :, :num_path] = x['mask']
                a[bs_idx, ue_idx, :, :num_path] = x['a']
                phi_r[bs_idx, ue_idx, :, :num_path] = x['phi_r']
                phi_t[bs_idx, ue_idx, :, :num_path] = x['phi_t']
                theta_r[bs_idx, ue_idx, :, :num_path] = x['theta_r']
                theta_t[bs_idx, ue_idx, :, :num_path] = x['theta_t']
                tau[bs_idx, ue_idx, :, :num_path] = x['tau']
            self.clear_rays()
        mask = mask[:, :, :, :max_num_path]
        a = a[:, :, :, :max_num_path]
        phi_r = phi_r[:, :, :, :max_num_path]
        phi_t = phi_t[:, :, :, :max_num_path]
        theta_r = theta_r[:, :, :, :max_num_path]
        theta_t = theta_t[:, :, :, :max_num_path]
        tau = tau[:, :, :, :max_num_path]
        a[~mask] = 0.0 + 0.0j
        return {'time_idx': time_idx, 'bs_name_list': bs_name_list, 'ue_name_list': ue_name_list,
                'ue_pos': ue_pos, 'ue_att': ue_att, 'ue_mask': ue_mask,
                'mask': mask, 'a': a, 'phi_r': phi_r, 'phi_t': phi_t, 'theta_r': theta_r, 'theta_t': theta_t,
                'tau': tau}

    def get_ray(self, bs_name, ue_name):
        return self._rays[(bs_name, ue_name)]

    def append_ray(self, name, time, time_index, ue_pos, ue_att, bs_pos, bs_att, num_path, time_loc, path_loc,
                   path_index, a, phi_r, phi_t, theta_r, theta_t, tau, objects, vertices):
        self._rays[name] = Ray(name, time, time_index, ue_pos, ue_att, bs_pos, bs_att, num_path, time_loc, path_loc,
                               path_index, a, phi_r, phi_t, theta_r, theta_t, tau, objects, vertices)
        bs_idx, ue_idx = self.bs_name_list.index(name[0]), self.ue_name_list.index(name[1])
        self._ray_time_index_start[bs_idx, ue_idx] = time_index[0]
        self._ray_time_index_end[bs_idx, ue_idx] = time_index[-1] + 1
        self._ray_max_num_path[bs_idx, ue_idx] = np.max(num_path)

    def save_ray(self, file_name):
        with open(self._ray_dir / file_name, 'wb') as f:
            rays = list(self._rays.items())
            ray_name_list = []
            for r in rays:
                ray_name_list.append(r[0])
                bs_name, ue_name = r[0][0], r[0][1]
                bs_idx, ue_idx = self.bs_name_list.index(bs_name), self.ue_name_list.index(ue_name)
                self._ray_files[bs_idx, ue_idx] = file_name
            dill.dump(ray_name_list, f)
            for r in rays:
                r[1].save(f)

    def load_ray(self, file_name):
        with open(self._ray_dir / file_name, 'rb') as f:
            names = dill.load(f)
            for n in names:
                self._rays[n] = Ray.load(f)

    def clear_rays(self):
        self._rays.clear()

    def load_ray_bs_ue_name(self, bs_name_list, ue_name_list):
        file_name_list = []
        for bs_name in bs_name_list:
            for ue_name in ue_name_list:
                bs_idx, ue_idx = self.bs_name_list.index(bs_name), self.ue_name_list.index(ue_name)
                file_name = self._ray_files[bs_idx, ue_idx].decode('UTF-8')
                if file_name not in file_name_list:
                    file_name_list.append(file_name)
        for file_name in file_name_list:
            self.load_ray(file_name)

    def save(self):
        with open(self._ray_dir / 'ray_group.pkl', 'wb') as f:
            dill.dump(self.bs_name_list, f)
            dill.dump(self.ue_name_list, f)
            dill.dump(self.time, f)
            dill.dump(self.wavelength, f)
            dill.dump(self._ray_files, f)
            dill.dump(self._ray_time_index_start, f)
            dill.dump(self._ray_time_index_end, f)
            dill.dump(self._ray_max_num_path, f)
            dill.dump(self._ray_dir, f)

    @classmethod
    def load(cls, scenario_name, simulation_name):
        path = (Path(__file__).parents[1] / 'scenario' / scenario_name / 'simulation' /
                simulation_name / 'ray' / 'ray_group.pkl')
        with open(path, 'rb') as f:
            obj = cls(scenario_name, simulation_name)
            obj.bs_name_list = dill.load(f)
            obj.ue_name_list = dill.load(f)
            obj.time = dill.load(f)
            obj.wavelength = dill.load(f)
            obj._ray_files = dill.load(f)
            obj._ray_time_index_start = dill.load(f)
            obj._ray_time_index_end = dill.load(f)
            obj._ray_max_num_path = dill.load(f)
            obj._ray_dir = dill.load(f)
        return obj

    def plot(self, name, start_time=None, end_time=None):
        self.load_ray_bs_ue_name(bs_name_list=[name[0]], ue_name_list=[name[1]])
        self._rays[name].plot(start_time, end_time)

    def init_animation(self, scene, bs_name_list=None, ue_name_list=None, max_num_path=20, linewidth=1.0):
        self._anim_bs_name_list = bs_name_list if bs_name_list is not None else self.bs_name_list
        self._anim_ue_name_list = ue_name_list if ue_name_list is not None else self.ue_name_list
        self.load_ray_bs_ue_name(self._anim_bs_name_list, self._anim_ue_name_list)
        for bs_name in self._anim_bs_name_list:
            for ue_name in self._anim_ue_name_list:
                self._rays[(bs_name, ue_name)].init_animation(scene, max_num_path, linewidth)

    def animate(self, time):
        for bs_name in self._anim_bs_name_list:
            for ue_name in self._anim_ue_name_list:
                self._rays[(bs_name, ue_name)].animate(time)


class Ray:
    def __init__(self, name, time, time_index, ue_pos, ue_att, bs_pos, bs_att, num_path, time_loc, path_loc, path_index,
                 a, phi_r, phi_t, theta_r, theta_t, tau, objects, vertices):
        self._name = name
        self._time = time
        self._time_index = time_index
        self._ue_pos = ue_pos
        self._ue_att = ue_att
        self._bs_pos = bs_pos
        self._bs_att = bs_att
        self._num_path = num_path
        self._time_loc = time_loc
        self._path_loc = path_loc
        self._path_index = path_index
        self._a = a
        self._phi_r = phi_r
        self._phi_t = phi_t
        self._theta_r = theta_r
        self._theta_t = theta_t
        self._tau = tau
        self._objects = objects
        self._vertices = vertices  # index, depth, coordinate
        self._ani_mesh = []
        self._ani_prev_time = 0.0
        self._ani_cur_idx = 0

    def save(self, f):
        d = {'name': self._name, 'time': self._time, 'time_index': self._time_index,
             'ue_pos': self._ue_pos, 'ue_att': self._ue_att,
             'bs_pos': self._bs_pos, 'bs_att': self._bs_att,
             'num_path': self._num_path, 'time_loc': self._time_loc, 'path_loc': self._path_loc,
             'path_index': self._path_index, 'a': self._a, 'phi_r': self._phi_r,
             'phi_t': self._phi_t, 'theta_r': self._theta_r, 'theta_t': self._theta_t, 'tau': self._tau,
             'objects': self._objects, 'vertices': self._vertices}
        dill.dump(d, f)

    @classmethod
    def load(cls, f):
        return cls(**dill.load(f))

    @property
    def time(self):
        return self._time

    def get_time_index_range(self):
        return np.min(self._time_index), np.max(self._time_index) + 1

    def get_num_paths(self, time_idx_range):
        num_path = self._num_path[np.logical_and(self._time_index >= time_idx_range[0],
                                                 self._time_index < time_idx_range[1])]
        max_num_path = np.max(num_path) if num_path.size > 0 else 0
        return max_num_path

    def get_data_tensor(self, time_idx_range):
        time_idx_start, time_idx_end = time_idx_range
        rel_time_index = self._time_index - time_idx_start
        time_len = time_idx_end - time_idx_start
        time_mask = np.logical_and(0 <= rel_time_index, rel_time_index < time_len)
        path_mask = time_mask[self._time_loc]
        num_path = self._num_path[time_mask]
        max_num_path = np.max(num_path) if num_path.size > 0 else 0
        ue_pos = np.full(shape=(time_len, 3), fill_value=0.0)
        ue_pos[rel_time_index[time_mask]] = self._ue_pos[time_mask, :]
        ue_att = np.full(shape=(time_len, 3), fill_value=0.0)
        ue_att[rel_time_index[time_mask]] = self._ue_att[time_mask, :]
        ue_mask = np.full(shape=(time_len,), fill_value=False)
        ue_mask[rel_time_index[time_mask]] = True
        time_loc, path_loc = rel_time_index[self._time_loc[path_mask]], self._path_loc[path_mask]
        shape = (time_len, max_num_path)
        mask = np.full(shape=shape, fill_value=False)
        mask[time_loc, path_loc] = True
        a = np.full(shape=shape, fill_value=0.0 + 0.0j)
        a[time_loc, path_loc] = self._a[path_mask]
        phi_r = np.full(shape=shape, fill_value=0.0)
        phi_r[time_loc, path_loc] = self._phi_r[path_mask]
        phi_t = np.full(shape=shape, fill_value=0.0)
        phi_t[time_loc, path_loc] = self._phi_t[path_mask]
        theta_r = np.full(shape=shape, fill_value=0.0)
        theta_r[time_loc, path_loc] = self._theta_r[path_mask]
        theta_t = np.full(shape=shape, fill_value=0.0)
        theta_t[time_loc, path_loc] = self._theta_t[path_mask]
        tau = np.full(shape=shape, fill_value=0.0)
        tau[time_loc, path_loc] = self._tau[path_mask]
        return {'ue_pos': ue_pos, 'ue_att': ue_att, 'ue_mask': ue_mask,
                'mask': mask, 'a': a, 'phi_r': phi_r, 'phi_t': phi_t, 'theta_r': theta_r, 'theta_t': theta_t,
                'tau': tau}

    def get_path_data(self, idx, start_time, end_time):
        sel = (self._path_index == idx)
        t = self._time[self._time_loc]
        sel = np.logical_and(sel, t >= start_time)
        sel = np.logical_and(sel, t <= end_time)
        time = t[sel]
        a = self._a[sel]
        phi_r = self._phi_r[sel]
        phi_t = self._phi_t[sel]
        theta_r = self._theta_r[sel]
        theta_t = self._theta_t[sel]
        tau = self._tau[sel]
        return time, a, phi_r, phi_t, theta_r, theta_t, tau

    def plot(self, start_time=None, end_time=None):
        title = ['a_power', 'a_phase', 'tau', 'phi_r', 'phi_t', 'theta_r', 'theta_t']
        num_plot = len(title)
        fig = plt.figure(figsize=(18, num_plot*3))
        gs = GridSpec(nrows=num_plot, ncols=1)
        ax = {}
        for idx, t in enumerate(title):
            ax[t] = fig.add_subplot(gs[idx])
            ax[t].set_title(t)
        num_path = np.max(self._path_index) + 1
        start_time = self._time[0] if start_time is None else start_time
        end_time = self._time[-1] if end_time is None else end_time
        for p in range(num_path):
            time, a, phi_r, phi_t, theta_r, theta_t, tau = self.get_path_data(p, start_time, end_time)
            for t in title:
                if t == 'a_power':
                    a_abs = np.abs(a)
                    ax[t].plot(time, 20 * np.log10(a_abs, out=np.ones_like(a_abs) * -np.inf, where=(a_abs > 0.0)))
                if t == 'a_phase':
                    ax[t].plot(time, np.angle(a))
                if t == 'tau':
                    ax[t].plot(time, tau)
                if t == 'phi_r':
                    ax[t].plot(time, phi_r)
                if t == 'phi_t':
                    ax[t].plot(time, phi_t)
                if t == 'theta_r':
                    ax[t].plot(time, theta_r)
                if t == 'theta_t':
                    ax[t].plot(time, theta_t)
                ax[t].set_xlim(start_time, end_time)
        plt.show()

    def init_animation(self, scene, max_num_path=20, linewidth=1.0):
        for _ in range(max_num_path):
            segment = np.zeros((1, 2, 3), dtype=np.float32)
            geo = p3s.LineSegmentsGeometry(positions=segment)
            mat = p3s.LineMaterial(linewidth=linewidth, color="black")
            mesh = p3s.LineSegments2(geo, mat)
            mesh.visible = False
            scene.add(mesh)
            self._ani_mesh.append(mesh)
        self._ani_cur_idx = 0

    def animate(self, time):
        while self._ani_cur_idx < len(self._time) and self._ani_prev_time > self._time[self._ani_cur_idx]:
            self._ani_cur_idx += 1
        if self._ani_cur_idx < len(self._time) and (self._ani_prev_time <= self._time[self._ani_cur_idx] < time):
            mask = (self._time_loc == self._ani_cur_idx)
            objects = self._objects[mask]
            vertices = self._vertices[mask]
            bs_pos = self._bs_pos
            ue_pos = self._ue_pos[self._ani_cur_idx]
            for idx, mesh in enumerate(self._ani_mesh):
                if idx < objects.shape[0]:
                    obj = objects[idx]
                    vert = vertices[idx]
                    pos = vert[obj >= 0]
                    pos = np.concatenate((bs_pos[None, :], pos, ue_pos[None, :]))
                    pos = np.stack((pos[:-1], pos[1:]), axis=1).astype(dtype=np.float32)
                    mesh.geometry = p3s.LineSegmentsGeometry(positions=pos)
                    mesh.visible = True
                else:
                    mesh.visible = False
        else:
            for mesh in self._ani_mesh:
                mesh.visible = False
        self._ani_prev_time = time


if __name__ == '__main__':
    rg = RayGroup.load(scenario_name='Suwon', simulation_name='sim_1')
    start_time = None
    end_time = None
    #rg.plot(name=('bs0', 'ue1'), start_time=start_time, end_time=end_time)
    data = rg.get_data_tensor(bs_name_list=None, ue_name_list=['ue0', 'ue1'], time_idx_range=(0, 500))
    pass

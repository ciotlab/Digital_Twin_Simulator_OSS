import numpy as np
import dill
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pythreejs as p3s


class UEGroup:
    def __init__(self):
        self.ue = {}
        self.time = np.empty(0)
        self.x_range = None
        self.y_range = None
        self._iter_idx = 0
        self._anim_ue_name_list = None

    def __len__(self):
        n = 0
        for k in self.ue:
            n += len(self.ue[k])
        return n

    def get_ue_name_list(self):
        return list(self.ue.keys())

    def get_ue_time_idx_dict(self):
        return {k: self.ue[k].time_index for k in self.ue}

    def get_data(self, name_list=None):
        name, length, time, time_index, position, attitude = [], [], [], [], [], []
        if name_list is None:
            name_list = list(self.ue.keys())
        for n in name_list:
            rd = self.ue[n]
            name.append(n)
            length.append(len(rd))
            time.append(rd.time)
            time_index.append(rd.time_index)
            position.append(rd.position)
            attitude.append(rd.attitude)
        return name, length, time, time_index, position, attitude

    def append_mobility_data(self, name, time, time_index, position, attitude, velocity):
        if name not in self.ue:
            self.ue[name] = UE(name)
        self.ue[name].append_mobility_data(time, time_index, position, attitude, velocity)

    def save(self, scenario_name, simulation_name, file_name='ue.pkl'):
        path = Path(__file__).parents[1] / 'scenario' / scenario_name / 'simulation' / simulation_name / file_name
        with open(path, 'wb') as f:
            ue = list(self.ue.items())
            dill.dump([u[0] for u in ue], f)
            for u in ue:
                u[1].save(f)
            dill.dump(self.time, f)
            dill.dump(self.x_range, f)
            dill.dump(self.y_range, f)

    @classmethod
    def load(cls, scenario_name, simulation_name, file_name='ue.pkl'):
        path = Path(__file__).parents[1] / 'scenario' / scenario_name / 'simulation' / simulation_name / file_name
        with open(path, 'rb') as f:
            obj = cls()
            names = dill.load(f)
            for n in names:
                obj.ue[n] = UE.load(f)
            obj.time = dill.load(f)
            obj.x_range = dill.load(f)
            obj.y_range = dill.load(f)
        return obj

    def plot(self, name_list=None, start_time=0, end_time=np.inf, velocity_vector=True, velocity_vector_step=10):
        fig = plt.figure()
        ax = fig.add_subplot()
        if name_list is None:
            name_list = list(self.ue.keys())
        for name in name_list:
            self.ue[name].plot(ax, start_time, end_time, velocity_vector, velocity_vector_step)
        fig.show()

    def init_animation(self, scene, ue_name_list=None, ue_scale=1.0):
        self._anim_ue_name_list = ue_name_list if ue_name_list is not None else self.get_ue_name_list()
        for name in self._anim_ue_name_list:
            self.ue[name].init_animation(scene, ue_scale=ue_scale)

    def animate(self, time):
        for name in self._anim_ue_name_list:
            self.ue[name].animate(time)


class UE:
    def __init__(self, name):
        self.name = name
        self.time = np.empty(0)
        self.time_index = np.empty(0).astype(int)
        self.next_time_index = -1
        self.position = np.empty((0, 3))
        self.attitude = np.empty((0, 3))  # z-y-x intrinsic rotation
        self.velocity = np.empty(0)
        self.velocity_vector = None
        self._ani_mesh = None
        self._ani_prev_time = 0.0
        self._ani_cur_idx = None

    def save(self, f):
        d = {'name': self.name, 'time': self.time, 'time_index': self.time_index,
             'position': self.position, 'attitude': self.attitude,
             'velocity': self.velocity, 'velocity_vector': self.velocity_vector}
        dill.dump(d, f)

    @classmethod
    def load(cls, f):
        d = dill.load(f)
        obj = cls(d['name'])
        obj.time, obj.time_index, obj.position, obj.attitude, obj.velocity, obj.velocity_vector = (
            d['time'], d['time_index'], d['position'], d['attitude'], d['velocity'], d['velocity_vector'])
        return obj

    def __len__(self):
        return self.time.shape[0]

    def append_mobility_data(self, time, time_index, position, attitude, velocity):
        self.time = np.concatenate((self.time, np.array(time)[None]), axis=0)
        self.time_index = np.concatenate((self.time_index, np.array(time_index)[None]), axis=0)
        self.position = np.concatenate((self.position, np.array(position)[None, :]), axis=0)
        self.attitude = np.concatenate((self.attitude, np.array(attitude)[None, :]), axis=0)
        self.velocity = np.concatenate((self.velocity, np.array(velocity)[None]), axis=0)

    def cal_velocity_vector(self):
        self.velocity_vector = np.empty((0, 3))
        for attitude, velocity in zip(self.attitude, self.velocity):
            rot = R.from_euler('ZYX', attitude)
            v = rot.apply((velocity, 0, 0))
            self.velocity_vector = np.concatenate((self.velocity_vector, v[None]), axis=0)

    def plot(self, ax, start_time=0, end_time=np.inf, velocity_vector=True, velocity_vector_step=10):
        time_idx = np.logical_and(start_time <= self.time, self.time <= end_time)
        pos = self.position[time_idx, :]
        ax.plot(pos[:, 0], pos[:, 1])
        if velocity_vector:
            if self.velocity_vector is None:
                self.cal_velocity_vector()
            vec = self.velocity_vector[time_idx, :]
            vec = vec[::velocity_vector_step, :]
            pos = pos[::velocity_vector_step, :]
            ax.quiver(pos[:, 0], pos[:, 1], vec[:, 0], vec[:, 1], angles='xy', width=0.002, headwidth=6, headlength=8)

    def init_animation(self, scene, ue_scale=1.0):
        width = 3.0 * ue_scale
        height = 1.5 * ue_scale
        depth = 1.5 * ue_scale
        geo = p3s.BoxGeometry(width=width, height=height, depth=depth)
        mat = p3s.MeshStandardMaterial(color='blue')
        self._ani_mesh = p3s.Mesh(geometry=geo, material=mat)
        self._ani_mesh.visible = False
        scene.add(self._ani_mesh)
        self._ani_cur_idx = 0

    def animate(self, time):
        while self._ani_cur_idx < len(self.time) and self._ani_prev_time > self.time[self._ani_cur_idx]:
            self._ani_cur_idx += 1
        if self._ani_cur_idx < len(self.time) and (self._ani_prev_time <= self.time[self._ani_cur_idx] < time):
            position = self.position[self._ani_cur_idx]
            attitude = self.attitude[self._ani_cur_idx]
            self._ani_mesh.position = tuple(position)
            theta = attitude[0]
            rot_mat = [np.cos(theta), np.sin(theta), 0, -np.sin(theta), np.cos(theta), 0, 0, 0, 1]
            self._ani_mesh.setRotationFromMatrix(rot_mat)
            self._ani_mesh.visible = True
        else:
            self._ani_mesh.visible = False
        self._ani_prev_time = time


if __name__ == '__main__':
    ue = UEGroup.load(scenario_name='Suwon', simulation_name='sim_1')
    #name_list = ['ue0', 'ue1', 'ue2', 'ue3']
    name_list = None
    ue.plot(name_list=name_list, start_time=0, end_time=300, velocity_vector=True, velocity_vector_step=10)

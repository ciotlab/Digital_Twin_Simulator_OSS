import numpy as np
import pythreejs as p3s
import time
from simulator.ray import RayGroup
from simulator.ue import UEGroup
from simulator.background import Background


class Animation:
    def __init__(self, scenario_name, simulation_name, bs_name_list=None, ue_name_list=None):
        self.background = Background.load(scenario_name=scenario_name, simulation_name=simulation_name)
        self.ue = UEGroup.load(scenario_name=scenario_name, simulation_name=simulation_name)
        self.ray = RayGroup.load(scenario_name=scenario_name, simulation_name=simulation_name)
        self._bs_name_list = bs_name_list if bs_name_list is not None else self.background.get_bs_name_list()
        self._ue_name_list = ue_name_list if ue_name_list is not None else self.ue.get_ue_name_list()
        self._scene = None
        self._renderer = None
        self._show_ray_path = True

    @property
    def renderer(self):
        return self._renderer

    def init_animation(self, bs_scale=100.0, ue_scale=10.0, ray_linewidth=3.0, max_num_ray_path=20, show_ray_path=True):
        resolution = (1440, 1024)
        fov = 45
        background = '#ffffff'

        ambient_light = p3s.AmbientLight(intensity=0.80)
        camera_light = p3s.DirectionalLight(position=[0, 0, 0], intensity=0.25)
        camera = p3s.PerspectiveCamera(fov=fov, aspect=resolution[0] / resolution[1], up=[0, 0, 1], far=10000,
                                       children=[camera_light], position=[0, 0, 1000])
        orbit = p3s.OrbitControls(controlling=camera)
        self._scene = p3s.Scene(background=background, children=[camera, ambient_light])
        self._renderer = p3s.Renderer(camera=camera, scene=self._scene, controls=[orbit], width=resolution[0],
                                      height=resolution[1], antialias=True)
        orbit.exec_three_obj_method('update')
        camera.exec_three_obj_method('updateProjectionMatrix')

        self.background.init_animation(self._scene, self._bs_name_list, bs_scale=bs_scale)
        self.ue.init_animation(self._scene, self._ue_name_list, ue_scale=ue_scale)
        self._show_ray_path = show_ray_path
        if self._show_ray_path:
           self.ray.init_animation(self._scene, self._bs_name_list, self._ue_name_list,
                                   max_num_path=max_num_ray_path, linewidth=ray_linewidth)

    def animate(self, start_time, end_time, time_step, sleep_time=0.1):
        time_steps = np.arange(start_time + time_step / 2.0, end_time + time_step / 2.0, time_step)
        for t in time_steps:
            print(f"{t}\r", end="")
            self.ue.animate(t)
            if self._show_ray_path:
                self.ray.animate(t)
            time.sleep(sleep_time)


if __name__ == '__main__':
    bs_name_list = ['bs0', 'bs1', 'bs2', 'bs3', 'bs4', 'bs5', 'bs6', 'bs7']
    # bs_name_list = None
    ue_name_list = ['ue0', 'ue1', 'ue2', 'ue3', 'ue4', 'ue5', 'ue6', 'ue7']
    # ue_name_list = None
    s = Animation(scenario_name='Suwon', simulation_name='sim_1', bs_name_list=bs_name_list, ue_name_list=ue_name_list)
    s.init_animation(bs_scale=10.0, ue_scale=3.0, ray_linewidth=2.0, max_num_ray_path=20, show_ray_path=True)
    s.animate(start_time=0.0, end_time=300.0, time_step=0.1, sleep_time=0.1)

import dill
from pathlib import Path
import matplotlib.pyplot as plt
import pythreejs as p3s
import numpy as np


class Background:
    def __init__(self):
        self.mesh_vertices = None
        self.mesh_faces = None
        self.mesh_albedos = None
        self.bs = {}
        self._anim_bs_name_list = None

    def get_bs_name_list(self):
        return list(self.bs.keys())

    def get_bs_attitude_dict(self):
        return {k: self.bs[k].attitude for k in self.bs}

    def set_mesh(self, vertices, faces, albedos):
        self.mesh_vertices = vertices
        self.mesh_faces = faces
        self.mesh_albedos = albedos

    def append_bs(self, name, position, attitude):
        bs = BS(name, position, attitude)
        self.bs[name] = bs

    def save(self, scenario_name, simulation_name, file_name='background.pkl'):
        path = Path(__file__).parents[1] / 'scenario' / scenario_name / 'simulation' / simulation_name / file_name
        with open(path, 'wb') as f:
            bs = list(self.bs.items())
            dill.dump([b[0] for b in bs], f)
            for b in bs:
                b[1].save(f)
            dill.dump(self.mesh_vertices, f)
            dill.dump(self.mesh_faces, f)
            dill.dump(self.mesh_albedos, f)

    @classmethod
    def load(cls, scenario_name, simulation_name, file_name='background.pkl'):
        path = Path(__file__).parents[1] / 'scenario' / scenario_name / 'simulation' / simulation_name / file_name
        with open(path, 'rb') as f:
            obj = cls()
            names = dill.load(f)
            for n in names:
                obj.bs[n] = BS.load(f)
            obj.mesh_vertices = dill.load(f)
            obj.mesh_faces = dill.load(f)
            obj.mesh_albedos = dill.load(f)
        return obj

    def plot(self, z_max=1000):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_zlim(0, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        x, y, z = self.mesh_vertices[:, 0], self.mesh_vertices[:, 1], self.mesh_vertices[:, 2]
        ax.plot_trisurf(x, y, z, triangles=self.mesh_faces)
        for bs in self.bs.values():
            bs.plot(ax)
        plt.show()

    def init_animation(self, scene, bs_name_list=None, bs_scale=10.0):
        colors = self.mesh_albedos.astype(np.float32)
        colors = np.power(colors, 1 / 1.8)
        geo = p3s.BufferGeometry(
            attributes={
                'index': p3s.BufferAttribute(self.mesh_faces.ravel(), normalized=False),
                'position': p3s.BufferAttribute(self.mesh_vertices, normalized=False),
                'color': p3s.BufferAttribute(colors, normalized=False)
            }
        )
        mat = p3s.MeshStandardMaterial(side='DoubleSide', metalness=0., roughness=1.0, vertexColors='VertexColors',
                                       flatShading=True)
        mesh = p3s.Mesh(geo, mat)
        scene.add(mesh)
        self._anim_bs_name_list = bs_name_list if bs_name_list is not None else self.get_bs_name_list()
        for name in self._anim_bs_name_list:
            self.bs[name].init_animation(scene, bs_scale)


class BS:
    def __init__(self, name, position, attitude):
        self.name = name
        self.position = position
        self.attitude = attitude

    def save(self, f):
        d = {'name': self.name, 'position': self.position, 'attitude': self.attitude}
        dill.dump(d, f)

    @classmethod
    def load(cls, f):
        return cls(**dill.load(f))

    def plot(self, ax):
        x, y, z = self.position[0], self.position[1], self.position[2]
        ax.scatter(x, y, z, s=10, color='red')

    def init_animation(self, scene, bs_scale=10.0):
        width = bs_scale
        height = bs_scale
        depth = (width + height) / 8.0
        geo = p3s.BoxGeometry(width=width, height=height, depth=depth)
        mat = p3s.MeshStandardMaterial(color='red')
        mesh = p3s.Mesh(geometry=geo, material=mat)
        mesh.position = tuple(self.position)
        mesh.rotateZ(self.attitude[0])
        mesh.rotateY(np.pi / 2 + self.attitude[1])
        scene.add(mesh)


if __name__ == '__main__':
    bg = Background.load(scenario_name='Suwon', simulation_name='sim_1')
    bg.plot()


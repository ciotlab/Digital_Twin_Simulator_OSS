import os
import subprocess
import sumo
import traci
import yaml
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ue import UEGroup


class MobilitySimulator:
    def __init__(self, scenario_name, simulation_name):
        self.scenario_name = scenario_name
        self.simulation_name = simulation_name
        self.scenario_dir = Path(__file__).parents[1] / 'scenario' / scenario_name
        self.simulation_dir = self.scenario_dir / 'simulation' / simulation_name
        self.sumo_dir = self.scenario_dir / 'sumo'
        with open(self.scenario_dir / 'config.yaml', 'r') as f:
            simulation_area = yaml.safe_load(f)['simulation_area']
        self.lower_left_coord = [simulation_area['lower_left']['lat'], simulation_area['lower_left']['lon']]
        self.upper_right_coord = [simulation_area['upper_right']['lat'], simulation_area['upper_right']['lon']]
        self.center_coord = [
            (self.lower_left_coord[0] + self.upper_right_coord[0]) / 2,
            (self.lower_left_coord[1] + self.upper_right_coord[1]) / 2
        ]
        self._earth_radius = 6371000.0
        self._deg_to_rad = np.pi / 180.0
        delta_min_lat = self.lower_left_coord[0] - self.center_coord[0]
        delta_min_lon = self.lower_left_coord[1] - self.center_coord[1]
        x_min = (delta_min_lon * self._deg_to_rad) * self._earth_radius * np.cos(delta_min_lat * self._deg_to_rad)
        y_min = (delta_min_lat * self._deg_to_rad) * self._earth_radius
        delta_max_lat = self.upper_right_coord[0] - self.center_coord[0]
        delta_max_lon = self.upper_right_coord[1] - self.center_coord[1]
        x_max = (delta_max_lon * self._deg_to_rad) * self._earth_radius * np.cos(delta_max_lat * self._deg_to_rad)
        y_max = (delta_max_lat * self._deg_to_rad) * self._earth_radius
        self.ue_group = UEGroup()
        self.ue_group.x_range = (x_min, x_max)
        self.ue_group.y_range = (y_min, y_max)
        with open(self.simulation_dir / 'config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)['mobility']

    def generate_trip(self):
        # Modify SUMO config file
        tree = ET.parse(self.sumo_dir / 'osm.sumocfg')
        root = tree.getroot()

        time_elem = root.find('time')
        if time_elem is None:
            time_elem = ET.SubElement(root, 'time')

        step_length_elem = time_elem.find('step-length')
        if step_length_elem is None:
            step_length_elem = ET.SubElement(time_elem, 'step-length')
        step_length_elem.set('value', str(self.config['simulation_step']))

        end_elem = time_elem.find('end')
        if end_elem is None:
            end_elem = ET.SubElement(time_elem, 'end')
        end_elem.set('value', str(self.config['simulation_time']))

        tree.write(self.sumo_dir / 'osm.sumocfg')

        # Run random trip generation code
        sumo_home = sumo.SUMO_HOME
        cmd = [
            "python",
            os.path.join(sumo_home, "tools", "randomTrips.py"),
            "-n", self.sumo_dir / "osm.net.xml.gz",
            "--fringe-factor", "5",
            "--insertion-density", str(self.config['insertion_density']),
            "-o", self.sumo_dir / "osm.passenger.trips.xml",
            "-r", self.sumo_dir / "osm.passenger.rou.xml",
            "-b", "0",
            "-e", str(self.config['simulation_time']),
            "--trip-attributes", 'departLane="best"',
            "--fringe-start-attributes", 'departSpeed="max"',
            "--validate",
            "--remove-loops",
            "--via-edge-types", "highway.motorway,highway.motorway_link,highway.trunk_link,highway.primary_link,highway.secondary_link,highway.tertiary_link",
            "--vehicle-class", "passenger",
            "--vclass", "passenger",
            "--prefix", "veh",
            "--min-distance", str(self.config['min_distance']),
            "--min-distance.fringe", "10",
            "--allow-fringe.min-length", "1000",
            "--lanes"
        ]
        subprocess.run(cmd)
    
    def run(self):
        sumo_cmd = ['sumo', '-c', str(self.sumo_dir / 'osm.sumocfg')]
        traci.start(sumo_cmd)
        simulation_time = float(self.config['simulation_time'])
        simulation_step = float(self.config['simulation_step'])
        n_steps = int(simulation_time / simulation_step)
        time_index = 0
        time_list = []
        with tqdm(total=n_steps, desc="Simulation Progress", unit='step') as pbar:
            while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < simulation_time:
                traci.simulationStep()
                time = traci.simulation.getTime()
                time_list.append(time)
                veh_id_list = traci.vehicle.getIDList()
                for veh_id in veh_id_list:
                    _x, _y = traci.vehicle.getPosition(veh_id)
                    lon, lat = traci.simulation.convertGeo(_x, _y)
                    delta_lat = lat - self.center_coord[0]
                    delta_lon = lon - self.center_coord[1]
                    x = (delta_lon * self._deg_to_rad) * self._earth_radius * np.cos(lat * self._deg_to_rad)
                    y = (delta_lat * self._deg_to_rad) * self._earth_radius
                    z = self.config['vehicle_height']
                    position = (x, y, z)
                    angle = traci.vehicle.getAngle(veh_id)
                    angle = np.pi / 2.0 - angle * self._deg_to_rad
                    attitude = (angle, 0, 0)
                    velocity = traci.vehicle.getSpeed(veh_id)
                    if ((self.upper_right_coord[0] > lat > self.lower_left_coord[0])
                            and (self.upper_right_coord[1] > lon > self.lower_left_coord[1])):
                        name = 'ue' + veh_id[3:]
                        if name not in self.ue_group.ue or self.ue_group.ue[name].next_time_index == time_index:
                            self.ue_group.append_mobility_data(name, time, time_index, position, attitude, velocity)
                            self.ue_group.ue[name].next_time_index = time_index + 1
                time_index += 1
                pbar.update(1)
            self.ue_group.time = np.array(time_list)
        traci.close()

    def save(self, file_name='ue.pkl'):
        self.ue_group.save(self.scenario_name, self.simulation_name, file_name)


if __name__ == '__main__':
    scenario_name = 'Suwon'
    simulation_name = 'sim_1'
    sim = MobilitySimulator(scenario_name, simulation_name)
    sim.generate_trip()
    sim.run()
    sim.save()






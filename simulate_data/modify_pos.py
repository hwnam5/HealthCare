import os
import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glfw
#import mujoco_viewer
import mujoco
import pandas as pd
import xml.etree.ElementTree as ET

matplotlib.use("TkAgg")
plt.ion()
os.environ["MUJOCO_GL"] = "egl"

def set_phone_position(xml_path, output_xml, new_pos):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    if worldbody is not None:
        for body in worldbody.findall("body"):
            if body.get("name") == "phone":
                body.set("pos", f"{new_pos[0]} {new_pos[1]} {new_pos[2]}")
                break

    tree.write(output_xml)
    return output_xml

def get_sensor_id(model, sensor_name):
        if sensor_name in model.sensor_names:
            return model.sensor_names.index(sensor_name)
        return None
def get_imu_data(data, imu_accel_id, imu_gyro_id):
        if imu_accel_id is None or imu_gyro_id is None:
            return None, None
        return data.sensordata[imu_accel_id: imu_accel_id+3], data.sensordata[imu_gyro_id: imu_gyro_id+3]

def get_watch_distance(data, watch_dist_id):
        if watch_dist_id is None or watch_dist_id >= len(data.sensordata):
            return None
        return data.sensordata[watch_dist_id]

def NLOS2UWB(watch_dist, NLOS_prob=0.03):
        if watch_dist is None:
            return None
        if np.random.rand() < NLOS_prob:
            watch_dist += np.random.uniform(0.2, 0.5)
        return watch_dist
def run_simulation(xml_path,squat_data4excel, num_repeats=40, num_frames=20):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 1280, 1280)

    standing_qpos = np.array([
        0, 0, 1.25, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0
    ])
    squat_qpos = standing_qpos.copy()
    squat_qpos[0:3] = [0, 0, 0.596]
    squat_qpos[3:7] = [0.988015, 0, 0.154359, 0]
    squat_qpos[7:10] = [0, 0.4, 0]
    squat_qpos[10:16] = [-0.25, -0.5, -2.5, -2.65, -0.8, 0.56]
    squat_qpos[16:22] = [-0.25, -0.5, -2.5, -2.65, -0.8, 0.56]

    imu_accel_id = get_sensor_id(model, "imu_acc")
    imu_gyro_id = get_sensor_id(model, "imu_gyro")
    watch_dist_id = get_sensor_id(model, "distance_to_watch")

    for i in range(num_repeats):
        for t in np.linspace(0, 1, num_frames):
            if t == 0:
                start_range = get_watch_distance()
            interpolated_qpos = standing_qpos * (1 - t) + squat_qpos * t

            data.qpos[:] = interpolated_qpos
            mujoco.mj_step(model, data)

            imu_accel, imu_gyro = get_imu_data(data, imu_accel_id, imu_gyro_id)
            dist = get_watch_distance(data, watch_dist_id)
            distWNoise = NLOS2UWB(dist)

            squat_data4excel.append([0, start_range, distWNoise, imu_accel[0], imu_accel[1], 
                                     imu_accel[2], imu_gyro[0], imu_gyro[1], imu_gyro[2]])

            renderer.update_scene(data)
            img = renderer.render()
            plt.imshow(img)
            plt.axis("off")
            plt.pause(0.005)
        
        for t in np.linspace(0, 1, num_frames):
            interpolated_qpos = squat_qpos * (1 - t) + standing_qpos * t

            data.qpos[:] = interpolated_qpos
            mujoco.mj_step(model, data)

            imu_accel, imu_gyro = get_imu_data(data, imu_accel_id, imu_gyro_id)
            dist = get_watch_distance(data, watch_dist_id)
            distWNoise = NLOS2UWB(dist)

            squat_data4excel.append([1, start_range, distWNoise, imu_accel[0], imu_accel[1], 
                                     imu_accel[2], imu_gyro[0], imu_gyro[1], imu_gyro[2]])

            renderer.update_scene(data)
            img = renderer.render()
            plt.imshow(img)
            plt.axis("off")
            plt.pause(0.005)

    plt.show()

    df = pd.DataFrame(squat_data4excel, columns=["updown", "first_range", "UWB", "accel_x", "accel_y",
                                                 "accel_z", "gyro_x", "gyro_y", "gyro_z"])
    df.to_excel("squat_data.xlsx", index=False)

if __name__ == "__main__":
    original_xml = "humanoid_WSensor.xml"
    modified_xml = "humanoid_WSensor_modified.xml"
    
    squat_data4excel = []
    
    for i in range(100):
        x = np.random.uniform(0.2, 1.5)
        y = np.random.uniform(0.2, 1.5)
        z = np.random.uniform(0, 0.7)
        
        new_phone_pos = [x, y, z]

        modified_xml = set_phone_position(original_xml, modified_xml, new_phone_pos)
        run_simulation(modified_xml, squat_data4excel)
    
    df = pd.DataFrame(squat_data4excel, columns=["updown", "first_range", "UWB", "accel_x", "accel_y",
                                                 "accel_z", "gyro_x", "gyro_y", "gyro_z"])
    df.to_excel("squat_data.xlsx", index=False)

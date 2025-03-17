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

#matplotlib.use("TkAgg")
#plt.ion()
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
    for i in range(model.nsensor):
        if model.sensor(i).name == sensor_name:
            return i
    print("Fail to get sensor id.")
    return None
def get_imu_data(data, imu_accel_id, imu_gyro_id):
    if imu_accel_id is None or imu_gyro_id is None:
        return None, None
    imu_accel = data.sensordata[imu_accel_id * 3: imu_accel_id * 3 + 3]  # 가속도 (x, y, z)
    imu_gyro = data.sensordata[imu_gyro_id * 3: imu_gyro_id * 3 + 3]  # 각속도 (x, y, z)
    return imu_accel, imu_gyro

def imu_accel_noise(bias_drift_before, accel_value):
    BIAS_DRIFT_STD = 0.001 * abs(accel_value)

    bias_drift = np.random.normal(0, BIAS_DRIFT_STD, size=3)
    return bias_drift_before + bias_drift

def imu_gyro_noise(bias_drift_before, gyro_value):
    BIAS_DRIFT_STD = 0.0005 * abs(gyro_value)

    bias_drift = np.random.normal(0, BIAS_DRIFT_STD, size=3)
    return bias_drift_before + bias_drift

def get_distance(data, watch_position_id, phone_position_id):
    if watch_position_id is None or phone_position_id is None:
        return None
    watch_pos = data.sensordata[watch_position_id * 3: watch_position_id * 3 + 3]
    phone_pos = data.sensordata[phone_position_id * 3: phone_position_id * 3 + 3]
    return np.linalg.norm(watch_pos - phone_pos)


def NLOS2UWB(watch_dist, NLOS_prob=0.03):
        UWB_NOISE_STD = 0.05 * abs(watch_dist)
        if watch_dist is None:
            return None
        if np.random.uniform(0.00, 1.00) <= NLOS_prob:
            watch_dist += np.random.uniform(0.2, 0.5)
        watch_dist += np.random.normal(0, UWB_NOISE_STD)
        return watch_dist



def run_simulation(people_id, xml_path, squat_data4excel, num_repeats=1, num_frames=20):
    GYRO_NOISE_STD = 0.05
    ACCEL_NOISE_STD = 0.1

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    #renderer = mujoco.Renderer(model, 1280, 1280)

    standing_qpos = np.array([
        0, 0, 1.25,  # torso 위치
        1, 0, 0, 0,  # torso 방향 (중립)
        0, 0, 0,  # 척추 회전 (곧게)

        0, 0, 0, 0, 0, 0, # 오른쪽 다리 (자연스럽게 폄)
        0, 0, 0, 0, 0, 0,  # 왼쪽 다리 (자연스럽게 폄)

        0, 0,0,0,0,0  # 팔을 내림 (기본 위치)
    ])
    squat_qpos = standing_qpos.copy()
    squat_qpos[0:3] = [0, 0, 0.596]
    squat_qpos[3:7] = [0.988015, 0, 0.154359, 0]
    squat_qpos[7:10] = [0, 0.4, 0]
    squat_qpos[10:16] = [-0.25, -0.5, -2.5, -2.65, -0.8, 0.56]
    squat_qpos[16:22] = [-0.25, -0.5, -2.5, -2.65, -0.8, 0.56]

    imu_accel_id = get_sensor_id(model, "imu_acc")
    imu_gyro_id = get_sensor_id(model, "imu_gyro")
    watch_position_id = get_sensor_id(model, "watch_position")
    phone_position_id = get_sensor_id(model, "phone_position")

    bias_drift_accel = np.zeros(3)
    bias_drift_gyro = np.zeros(3)
    #for i in range(num_repeats):
    for j in range(40):
        for t in np.linspace(0, 1, num_frames):
            interpolated_qpos = standing_qpos * (1 - t) + squat_qpos * t

            data.qpos[:] = interpolated_qpos
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            #print(t)
            if t == 0.0 and j == 0:
                start_range = get_distance(data, watch_position_id, phone_position_id)
            imu_accel, imu_gyro = get_imu_data(data, imu_accel_id, imu_gyro_id)
            bias_drift_accel = imu_accel_noise(bias_drift_accel, imu_accel)
            bias_drift_gyro = imu_gyro_noise(bias_drift_gyro, imu_gyro)
            imu_accel = imu_accel + bias_drift_accel + np.random.normal(0, ACCEL_NOISE_STD * abs(imu_accel), size=3)
            imu_gyro = imu_gyro + bias_drift_gyro + np.random.normal(0, GYRO_NOISE_STD * abs(imu_gyro), size=3)
            dist = get_distance(data, watch_position_id, phone_position_id)
            #print(dist)
            distWNoise = NLOS2UWB(dist)

            squat_data4excel.append([people_id, 0, start_range, distWNoise, imu_accel[0], imu_accel[1], 
                                     imu_accel[2], imu_gyro[0], imu_gyro[1], imu_gyro[2]])
            #print(squat_data4excel[len(squat_data4excel)-1])
            #renderer.update_scene(data)
            #img = renderer.render()
            #plt.imshow(img)
            #plt.axis("off")
            #plt.pause(0.005)
        
        for t in np.linspace(0, 1, num_frames):
            interpolated_qpos = squat_qpos * (1 - t) + standing_qpos * t

            data.qpos[:] = interpolated_qpos
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            imu_accel, imu_gyro = get_imu_data(data, imu_accel_id, imu_gyro_id)
            bias_drift_accel = imu_accel_noise(bias_drift_accel, imu_accel)
            bias_drift_gyro = imu_gyro_noise(bias_drift_gyro, imu_gyro)
            imu_accel = imu_accel + bias_drift_accel + np.random.normal(0, ACCEL_NOISE_STD * abs(imu_accel), size=3)
            imu_gyro = imu_gyro + bias_drift_gyro + np.random.normal(0, GYRO_NOISE_STD * abs(imu_gyro), size=3)
            dist = get_distance(data, watch_position_id, phone_position_id)
            distWNoise = NLOS2UWB(dist)

            squat_data4excel.append([people_id, 1, start_range, distWNoise, imu_accel[0], imu_accel[1], 
                                     imu_accel[2], imu_gyro[0], imu_gyro[1], imu_gyro[2]])
            #print(squat_data4excel[len(squat_data4excel)-1])
            #renderer.update_scene(data)
            #img = renderer.render()
            #plt.imshow(img)
            #plt.axis("off")
            #plt.pause(0.005)
        print(f"{j}th squat done")
    #plt.show()

    #df = pd.DataFrame(squat_data4excel, columns=["updown", "first_range", "UWB", "accel_x", "accel_y",
                                                 #"accel_z", "gyro_x", "gyro_y", "gyro_z"])

if __name__ == "__main__":
    original_xml = "humanoid_WSensor.xml"
    modified_xml = "humanoid_WSensor_modified.xml"
    
    squat_data4excel = []
    
    for i in range(500):
        people_id = i
        print(f"people_id: {people_id}")
        x = np.random.uniform(0.2, 1.5)
        y = np.random.uniform(0.2, 1.5)
        z = np.random.uniform(0, 0.7)
        
        new_phone_pos = [x, y, z]

        modified_xml = set_phone_position(original_xml, modified_xml, new_phone_pos)
        run_simulation(people_id, modified_xml, squat_data4excel)
    
    df = pd.DataFrame(squat_data4excel, columns=["people_id","updown", "first_range", "UWB", "accel_x", "accel_y",
                                                 "accel_z", "gyro_x", "gyro_y", "gyro_z"])
    df.to_excel("squat_data.xlsx", index=False)
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

#env = gym.make("Humanoid-v5", render_mode="rgb_array")
#env.reset()
XML_PATH = "humanoid_WSensor.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 1280, 1280)
#sim = mujoco.MjSim(model, data)

num_frames = 20
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
squat_qpos[3:7] = [0.988015, 0, 0.154359, 0]  # 몸 기울기
squat_qpos[7:10] = [0, 0.4, 0]  # 척추 회전
squat_qpos[10:16] = [-0.25, -0.5, -2.5, -2.65, -0.8, 0.56]  # 오른쪽 다리
squat_qpos[16:22] = [-0.25, -0.5, -2.5, -2.65, -0.8, 0.56]  # 왼쪽 다리

def get_sensor_id(model, sensor_name):
    for i in range(model.nsensor):
        if model.sensor(i).name == sensor_name:
            return i
    return None

imu_accel_id = get_sensor_id(model, "imu_acc")
imu_gyro_id = get_sensor_id(model, "imu_gyro")
watch_dist_id = get_sensor_id(model, "distance_to_watch")

def get_imu_data():
    if imu_accel_id is None or imu_gyro_id is None:
        return None, None
    imu_accel = data.sensordata[imu_accel_id * 3: imu_accel_id * 3 + 3]  # 가속도 (x, y, z)
    imu_gyro = data.sensordata[imu_gyro_id * 3: imu_gyro_id * 3 + 3]  # 각속도 (x, y, z)
    return imu_accel, imu_gyro

def get_watch_distance():
    if watch_dist_id is None:
        return None
    watch_dist = data.sensordata[watch_dist_id]
    return watch_dist

def NLOS2UWB(watch_dist, NLOS_prob = 0.03):
    watch_distWNoise = watch_dist
    if np.random.rand() < NLOS_prob:
        nlos_bias = np.random.uniform(0.2, 0.5)
        watch_distWNoise += nlos_bias
        
    return watch_distWNoise

squat_data4excel = []

for i in range(40):
    for t in np.linspace(0, 1, num_frames):
        if t==0:
            start_range = get_watch_distance()
        interpolated_qpos = standing_qpos * (1 - t) + squat_qpos * t

        data.qpos[:] = interpolated_qpos
        #data.qvel[:] = np.zeros_like(data.qvel)
        #if t == 0:
            #data.qvel[:] = np.zeros_like(data.qvel)
        mujoco.mj_step(model, data)

        imu_accel, imu_gyro = get_imu_data()
        dist = get_watch_distance()
        distWNoise = NLOS2UWB(dist)

        if distWNoise is not None:
            print(f"Frame {t:.2f}: Distance to watch = {distWNoise:.2f}")
        else:
            print(f"Frame {t:.2f}: There is no watch sensor in the model.")

        #print(f"Frame {t:.2f}: x = {data.qpos[0]:.2f}, y = {data.qpos[1]:.2f}, z = {data.qpos[2]:.2f}")
        if imu_accel is not None:
            print(f"Frame {t:.2f}: Accel = {imu_accel}, Gyro = {imu_gyro}")
        else:
            print(f"Frame {t:.2f}: There is no IMU sensor in the model.")
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
        #data.qvel[:] = np.zeros_like(data.qvel)
        #if t == 0:
            #data.qvel[:] = np.zeros_like(data.qvel)
        mujoco.mj_step(model, data)

        imu_accel, imu_gyro = get_imu_data()
        dist = get_watch_distance()
        distWNoise = NLOS2UWB(dist)

        if distWNoise is not None:
            print(f"Frame {t:.2f}: Distance to watch = {distWNoise:.2f}")
        else:
            print(f"Frame {t:.2f}: There is no watch sensor in the model.")
        
        #print(f"Frame {t:.2f}: x = {data.qpos[0]:.2f}, y = {data.qpos[1]:.2f}, z = {data.qpos[2]:.2f}")
        if imu_accel is not None:
            print(f"Frame {t:.2f}: Accel = {imu_accel}, Gyro = {imu_gyro}")
        else:
            print(f"Frame {t:.2f}: There is no IMU sensor in the model.")
            
        squat_data4excel.append([1, start_range, distWNoise, imu_accel[0], imu_accel[1], 
                                 imu_accel[2], imu_gyro[0], imu_gyro[1], imu_gyro[2]])

        renderer.update_scene(data)
        img = renderer.render()
        plt.imshow(img)
        plt.axis("off")
        plt.pause(0.005)

plt.show()
df = pd.DataFrame(squat_data4excel, columns=["updown", "first_range" "UWB", "accel_x", "accel_y",
                                             "gyro_x", "gyro_y", "gyro_z"])
df.to_excel("squat_data_temp.xlsx", index=False)
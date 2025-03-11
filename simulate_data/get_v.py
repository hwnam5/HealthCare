import os
import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glfw
#import mujoco_viewer
import mujoco

matplotlib.use("TkAgg")
plt.ion()
os.environ["MUJOCO_GL"] = "egl"

#env = gym.make("Humanoid-v5", render_mode="rgb_array")
#env.reset()
XML_PATH = "humanoid_WSensor.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 1280, 720)
#sim = mujoco.MjSim(model, data)

num_frames = 30
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

def get_imu_data():
    if imu_accel_id is None or imu_gyro_id is None:
        return None, None
    imu_accel = data.sensordata[imu_accel_id * 3: imu_accel_id * 3 + 3]  # 가속도 (x, y, z)
    imu_gyro = data.sensordata[imu_gyro_id * 3: imu_gyro_id * 3 + 3]  # 각속도 (x, y, z)
    return imu_accel, imu_gyro

for i in range(40):
    for t in np.linspace(0, 1, num_frames):
        interpolated_qpos = standing_qpos * (1 - t) + squat_qpos * t

        data.qpos[:] = interpolated_qpos
        #data.qvel[:] = np.zeros_like(data.qvel)
        #if t == 0:
            #data.qvel[:] = np.zeros_like(data.qvel)
        mujoco.mj_step(model, data)

        imu_accel, imu_gyro = get_imu_data()

        print(f"Frame {t:.2f}: x = {data.qpos[0]:.2f}, y = {data.qpos[1]:.2f}, z = {data.qpos[2]:.2f}")
        if imu_accel is not None:
            print(f"Frame {t:.2f}: Accel = {imu_accel}, Gyro = {imu_gyro}")
        else:
            print(f"Frame {t:.2f}: There is no IMU sensor in the model.")
        
        renderer.update_scene(data)
        img = renderer.render()
        plt.imshow(img)
        plt.axis("off")
        plt.pause(0.02)
    
    for t in np.linspace(0, 1, num_frames):
        interpolated_qpos = squat_qpos * (1 - t) + standing_qpos * t

        data.qpos[:] = interpolated_qpos
        #data.qvel[:] = np.zeros_like(data.qvel)
        #if t == 0:
            #data.qvel[:] = np.zeros_like(data.qvel)
        mujoco.mj_step(model, data)

        imu_accel, imu_gyro = get_imu_data()
        
        print(f"Frame {t:.2f}: x = {data.qpos[0]:.2f}, y = {data.qpos[1]:.2f}, z = {data.qpos[2]:.2f}")
        if imu_accel is not None:
            print(f"Frame {t:.2f}: Accel = {imu_accel}, Gyro = {imu_gyro}")
        else:
            print(f"Frame {t:.2f}: There is no IMU sensor in the model.")

        renderer.update_scene(data)
        img = renderer.render()
        plt.imshow(img)
        plt.axis("off")
        plt.pause(0.03)

plt.show()
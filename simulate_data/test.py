import os
import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Matplotlib 백엔드 설정 (WSL에서 GUI 사용 가능하도록)
matplotlib.use("TkAgg")

# Interactive mode 활성화
plt.ion()

# WSL에서 OpenGL 렌더링 문제 해결
os.environ["MUJOCO_GL"] = "egl"

# Humanoid 환경 생성 (MuJoCo)
env = gym.make("Humanoid-v5", render_mode="rgb_array")

# 모델 객체 직접 접근 (unwrapped 사용)
model = env.unwrapped.model

# 환경 초기화
obs, info = env.reset()

# 모델의 관절 개수 확인
nq = model.nq  # 관절 위치(qpos) 개수 (24)
nv = model.nv  # 관절 속도(qvel) 개수 (23)
action_dim = env.action_space.shape[0]  # 액션 공간 크기 (17)

print(f"✅ qpos 크기: {nq}, qvel 크기: {nv}")
print(f"✅ obs 크기: {obs.shape}")
print(f"✅ 액션 공간 크기: {action_dim}")  # Expected: 17

# PD 컨트롤러의 P, D 게인 값 (더 작은 값으로 조정)
P_GAIN = 50
D_GAIN = 5

# 고관절 제어 게인 값 (다리 벌어짐 방지)
HIP_GAIN = 30
HIP_D_GAIN = 3

# 목표 자세 초기화 (균형 잡힌 상태)
target_qpos = obs[:nq].copy()[1:]  # (23,)

def balance_controller(obs, target_qpos):
    """균형 유지 컨트롤러 (ZMP + Hip Adduction)"""
    qpos = obs[:nq][1:]  # (23,)
    qvel = obs[nq : nq + nv]  # (23,)

    # 무게중심(CoM) 보정 → 사람이 중심에서 벗어나지 않도록 함
    com_offset = obs[0]  # x축 중심 이동
    balance_torque = -P_GAIN * com_offset - D_GAIN * qvel[0]

    # 🔹 **발목 토크(Ankle Torque) 적용 (x축 무게 중심 보정)**
    ankle_torque = np.zeros_like(qpos)
    ankle_torque[0] = balance_torque  # 발목 관절에 힘 적용

    # 🔹 **고관절 토크(Hip Adduction Torque) 적용 (쩍벌 방지)**
    hip_torque = np.zeros_like(qpos)
    
    # qpos[6]: 고관절 벌어짐 각도 (Abduction/Adduction)
    hip_torque[6] = -HIP_GAIN * qpos[6] - HIP_D_GAIN * qvel[6]  # 허벅지 벌어짐 방지

    # PD 제어 계산 (목표 위치로 수렴하도록 힘을 적용)
    torque = P_GAIN * (target_qpos - qpos) - D_GAIN * qvel

    # 🔹 **균형 유지 토크 추가 (발목 + 고관절)**
    torque += ankle_torque + hip_torque

    # 토크 크기 제한 (과도한 힘 방지)
    torque = np.clip(torque[:action_dim], -1, 1)  # (23,) → (17,)

    return torque, target_qpos

# 그래프 설정
fig, ax = plt.subplots()

# 초기 목표 자세 설정
target_qpos = obs[:nq].copy()[1:]  # (23,)

# 시뮬레이션 루프
for _ in range(1000):  # 1000 스텝 동안 실행
    action, target_qpos = balance_controller(obs, target_qpos)  # 균형 유지 컨트롤 적용
    obs, reward, terminated, truncated, info = env.step(action)  # 환경 업데이트
    
    if terminated or truncated:
        obs, info = env.reset()  # 에피소드 종료 시 초기화
        target_qpos = obs[:nq].copy()[1:]  # 목표 자세도 초기화

    # 렌더링 (이미지 시각화)
    img = env.render()
    plt.imshow(img)
    plt.pause(0.01)  # 그래프 업데이트
    plt.clf()  # 기존 그림 삭제

env.close()

# Isaac Sim 6 / Isaac Lab 3 后端

这里保存 TinyMal、原创串联式轮腿机器人的原生 Isaac Lab 任务，以及训练/评估脚本和 ROS 2 Nav2 入口；NVIDIA 的大型仓库与二进制始终外置，不复制进 ActuateX。

## 精确版本

| 组件 | 锁定版本 | 说明 |
|---|---|---|
| Isaac Sim | `6.0.1 GA` / `v6.0.1` | 正式版，不是 nightly；Linux ZIP MD5 `65e2c2e83e2461ce0f33b0732d0ee4a3` |
| Isaac Lab | `v3.0.0-beta2.patch1` / `ffff603e...` | 官方明确适配 Sim 6.0.1，但 Lab 本身仍是 beta |
| Python | `3.12` | 使用 Sim 自带解释器，不混入 Ubuntu 22.04 的 ROS Humble Python 3.10 |
| PyTorch | `2.10.0+cu128` | 与该 Lab 标签一致 |
| RSL-RL | `rsl-rl-lib 5.0.1` | 使用新的 actor/critic model 配置 |

机器上原有的 Isaac Sim 5.1 与旧 Isaac Lab 不会被覆盖。`patches/0001-isaac-sim-5.1-offline-compat.patch` 只用于复现实验，**禁止应用到 6.0.1**。完整锁文件见 [`upstream.json`](upstream.json)。

逐项 API、资产、传感器、验证门槛和训练计划见
[`Isaac Sim 6.0.1 迁移报告`](../../docs/ISAAC_SIM_6_MIGRATION.zh-CN.md)。

2026-07-19 的真实 runtime、训练、双种子复评、双向 sim2sim 和 RTX/ROS writer 验收已经完成。最终路径、参数、哈希与证据索引见本地 `artifacts/isaac_sim_6/manifest.json`。

## 安装

从 [NVIDIA 官方下载页](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/download.html) 获取 `isaac-sim-standalone-6.0.1-linux-x86_64.zip`，校验后解压到一个新目录：

```bash
md5sum isaac-sim-standalone-6.0.1-linux-x86_64.zip
mkdir -p /path/to/isaac-sim-6.0.1
unzip isaac-sim-standalone-6.0.1-linux-x86_64.zip -d /path/to/isaac-sim-6.0.1
/path/to/isaac-sim-6.0.1/post_install.sh
```

准备精确锁定的 Isaac Lab，并创建 `_isaac_sim` 链接：

```bash
git clone https://github.com/isaac-sim/IsaacLab.git /path/to/IsaacLab-3
git -C /path/to/IsaacLab-3 checkout v3.0.0-beta2.patch1

python scripts/install_isaac_lab_compat.py \
  --isaac-lab-root /path/to/IsaacLab-3 \
  --isaac-sim-root /path/to/isaac-sim-6.0.1 \
  --link-runtime --verify-python

cd /path/to/IsaacLab-3
./isaaclab.sh -i 'rl[rsl-rl]'
```

安装器遇到已有 `_isaac_sim` 时会停止，不会替你删除或覆盖旧运行时。

## 分层验证

先做最小启动，不要直接投入长训练：

```bash
export ACTUATEX_ROOT=/path/to/ActuateX
export ISAAC_LAB_ROOT=/path/to/IsaacLab-3

"$ISAAC_LAB_ROOT/isaaclab.sh" -p \
  "$ACTUATEX_ROOT/backends/isaac_lab/scripts/train_tinymal.py" \
  --task Isaac-Velocity-Native-Forward-TinyMal-v0 \
  --num_envs 4 --max_iterations 1 --seed 1
```

通过后再把环境数量从 `4` 提高。本机纯物理鲁棒训练实测选择 `8192`：约 15.3–15.7 万 steps/s、峰值显存 15092 MiB（92.2%）。训练脚本会按环境数自动扩展 PhysX found/lost-pair buffer；`--gpu_found_lost_pairs_capacity` 只用于显式覆盖。RTX 相机/录像必须单独测显存上限。

## 训练

```bash
"$ISAAC_LAB_ROOT/isaaclab.sh" -p \
  "$ACTUATEX_ROOT/backends/isaac_lab/scripts/train_tinymal.py" \
  --task Isaac-Velocity-Native-Robust-TinyMal-v0 \
  --num_envs 8192 --max_iterations 60 --seed 29 \
  --learning_rate 1e-5 --schedule fixed \
  --entropy_coef 2e-4 --init_noise_std 0.05 \
  --init_checkpoint /path/to/general_model375.pt --init_critic
```

训练日志默认写到 `artifacts/isaac_lab/logs/rsl_rl/`；可用 `ACTUATEX_ARTIFACTS` 指向其他磁盘。

本轮通用策略选中 `model50`：两种子平均主轴 RMSE 从 `0.09544` 降到 `0.09445`（`1.04%`），重置数不变。台阶策略选中 Sim 6 微调 `model10`：256 次随机试验中无重置登顶 `212/256`，高于热启动模型的 `206/256`。

## 串联式轮腿机器人

本后端还注册了三个 6-DOF 串联式双轮腿任务：

- `Isaac-Velocity-Flat-SerialWheelLegged-v0`
- `Isaac-Velocity-Flat-SerialWheelLegged-Play-v0`
- `Isaac-Velocity-Robust-SerialWheelLegged-v0`

机器人 URDF 为本仓库使用基础几何体原创构建；四个腿关节使用位置动作，两个轮关节使用速度动作。策略为 28 维观测、6 维动作、50 Hz。鲁棒任务对每个环境独立采样 0–20 ms actuator delay，并扩大摩擦、质量、质心、PD 增益和外力随机化。

```bash
"$ISAAC_LAB_ROOT/isaaclab.sh" -p \
  "$ACTUATEX_ROOT/backends/isaac_lab/scripts/train_tinymal.py" \
  --task Isaac-Velocity-Flat-SerialWheelLegged-v0 \
  --num_envs 8192 --max_iterations 500 --seed 23

"$ISAAC_LAB_ROOT/isaaclab.sh" -p \
  "$ACTUATEX_ROOT/backends/isaac_lab/scripts/evaluate_wheel_legged.py" \
  --delayed --mode holdout --num_envs 1024 --seed 109 \
  --ckpt /path/to/stage1_model_499.pt /path/to/robust_model_199.pt \
  --out artifacts/isaac_sim_6/evaluation/wheel_legged_holdout.json
```

正式两阶段训练共采样 137,625,600 条 transition。最终 `model_199` 在 1024 环境未见强扰动中为 7 falls、RMSE 0.09817；第一阶段模型为 23 falls、RMSE 0.11662。最终 actor 已导出为 TorchScript，逐值一致性误差为 0。开源参考筛选、精确参数、完整命令、所有不利结果和下一步边界见 [`串联式轮腿完整报告`](../../docs/WHEEL_LEGGED_RL.zh-CN.md)。

## 楼梯评估与录像

```bash
"$ISAAC_LAB_ROOT/isaaclab.sh" -p \
  "$ACTUATEX_ROOT/backends/isaac_lab/scripts/evaluate_tinymal_stairs.py" \
  --ckpt artifacts/checkpoints/isaac_lab/model.pt \
  --num_envs 64 --video \
  --video_dir artifacts/videos/isaac_lab_stairs
```

`--robust` 会启用摩擦、质量、质心、PD 增益、执行器延迟、观测噪声和外推扰动。展示条件和鲁棒条件必须分别报告。

多 checkpoint 比较会为每个候选重建同一种子环境，保证随机流完全一致。最终 PPT 成片位于 `artifacts/isaac_sim_6/videos/TinyMal_IsaacSim_6_FinalPolicy_Stairs_PPT.mp4`。

## ROS 2 Nav2、雷达与相机

Sim 6 使用 `isaacsim.sensors.experimental.rtx`。导航脚本已迁移到新的 `Lidar/LidarSensor` 与 `RtxCamera/CameraSensor`，并使用相机 prim 的 `tick_rate` 控制帧率。

没有标定文件时相机不会创建。先从实机导出标准 ROS `camera_info` YAML，再运行：

```bash
"$ISAAC_LAB_ROOT/isaaclab.sh" -p \
  "$ACTUATEX_ROOT/backends/isaac_lab/scripts/run_tinymal_nav2.py" \
  --actor artifacts/checkpoints/isaac_lab/actor.pt \
  --camera_calibration /path/to/real_front_camera.yaml \
  --camera_mount_xyz 0.16 0.0 0.08 \
  --camera_mount_rpy_deg 0.0 0.0 0.0 \
  --camera_rate 30
```

示例文件 `navigation/ros2/actuatex_navigation/config/front_camera_info.example.yaml` 只用于检查格式，不能当作任何真实型号的标定结果。可选 `--camera_depth_topic` 发布的是理想 RTX 深度；若目标硬件是结构光、ToF 或双目，必须进一步使用相应的深度传感器模型。

Mid-360 不再使用旋转雷达近似。默认 `--mid360_stride 1` 会装载 Livox 官方
800,000 点非重复扫描表，按 40 个 10 Hz emitter state 逐点保留角度与
`fireTimeNs`。由于 Sim 6 RTX Hydra 对单传感器属性有 5 MiB 上限，20,000
发射点/帧被无损拆为 4 个同步 prim，再按绝对逐点时间合并、排序并发布为
与 `livox_ros_driver2` 一致的 26-byte `PointXYZRTLT`。独立验证为 40/40
状态、20,000/20,000 发射点覆盖、4/4 分片和四条硬件 line；机器狗完整密度
联调为 49 帧/5 秒、250 控制步零终止。

完整密度的独立复现命令会同时生成机器可读证据：

```bash
/path/to/isaac-sim-6.0.1/python.sh \
  backends/isaac_lab/scripts/validate_mid360_rtx.py \
  --stride 1 --frames 180 \
  --out artifacts/isaac_sim_6/evaluation/mid360_exact_profile_validation.json
```

RGB、`CameraInfo`、可选 depth writers 和 Mid-360 PointCloud2 已在实际仿真
步进中通过。这里仍未声称完成外部 ROS 订阅回环、Livox UDP 包级驱动、实机
强度/噪声标定或真实相机误差对齐；`CustomMsg` 仅在系统安装
`livox_ros_driver2` 消息包时启用。

## Isaac Gym 策略迁移

```bash
"$ISAAC_LAB_ROOT/isaaclab.sh" -p \
  "$ACTUATEX_ROOT/backends/isaac_lab/scripts/play_old_policy.py" \
  --ckpt artifacts/checkpoints/isaac_gym/model.pt \
  --suite --num_envs 64 \
  --out artifacts/isaac_lab/gym_transfer.json
```

48 维观测、关节顺序、动作缩放、力矩裁剪和控制周期会显式对齐。Sim 6 的 URDF Importer 已移除 cylinder-to-capsule 自动替换，因此默认使用仓库内经过审计的 capsule-compatible USD；显式设置 `ACTUATEX_TINYMAL_URDF` 才会切到 6.0 原生 cylinder 导入做 A/B。

## English note

This backend targets Isaac Sim 6.0.1 GA and Isaac Lab `v3.0.0-beta2.patch1`. Runtime startup, 8192-environment training, fair checkpoint evaluation, bidirectional sim2sim, and RTX/ROS writer rollouts were completed on 2026-07-19. NVIDIA runtimes remain external; real sensor fidelity and hardware deployment are still separate acceptance stages.

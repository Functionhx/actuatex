# 串联式轮腿机器人：开源调研、Isaac Sim 6 训练与鲁棒性验收

> 实测日期：2026-07-19
> 运行后端：Isaac Sim 6.0.1 GA、Isaac Lab 3.0.0-beta2.patch1、PhysX 5
> 训练硬件：NVIDIA GeForce RTX 4070 Ti SUPER 16 GB
> 最终策略：`model_199.pt`，已导出为独立 TorchScript

## 1. 结论先行

ActuateX 已经完成一套可训练、可评估、可导出的**串联式两轮腿机器人**基线，不是只搭了一个能加载的模型：

- 机器人为左右对称的开放串联腿，每侧 `hip → knee → wheel`，共 6 个主动自由度；
- 资产由本仓库使用 URDF 基础几何体原创构建，没有复制第三方 CAD 或 mesh；
- 控制策略使用 28 维观测、6 维动作和 50 Hz PPO 控制；
- 第一阶段完成 8192 环境、500 轮平地训练，采样 98,304,000 条 transition；
- 第二阶段加入 0–20 ms 执行器延迟与更宽域随机化，继续训练 200 轮，采样 39,321,600 条 transition；
- 在 1024 环境的未见强扰动 holdout 中，最终策略相对第一阶段模型将跌倒数从 23 降到 7，减少 **69.6%**；综合主轴 RMSE 从 0.11662 降到 0.09817，改善 **15.8%**；
- 在带 0–20 ms 延迟的干净环境中，两者均为 0 跌倒，最终策略的 RMSE 由 0.07101 降至 0.04467，改善 **37.1%**；
- 最终 actor 已导出为 `28-512-256-128-6` 的 TorchScript，和 eager PyTorch 的逐值比较最大误差为 **0.0**。

这是一条已经闭环的教学基线。它不等于真实机器部署已经完成；电机辨识、通信抖动实测、轮胎模型、热限制和硬件急停仍属于 sim-to-real 阶段。

## 2. 没有从零盲造：先筛选三个参考项目

调研时固定了上游 commit，避免以后因上游更新导致描述漂移。

| 项目 | 固定版本 | 可借鉴内容 | 为什么不直接搬过来 |
|---|---|---|---|
| [Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym) | `c354431e` | 与本任务最接近的 6 动作开放串联结构；端到端与 VMC 两条路线；Isaac Gym 训练经验 | 基于旧 Isaac Gym/PhysX 4，不是 Isaac Sim 6；其机器人资产有独立许可链 |
| [Isaac-RL-Two-wheel-Legged-Bot / Flamingo](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot) | `d922cce9` | 当前 Isaac Lab 项目组织、腿位置动作与轮速动作拆分、导出推理流程 | Flamingo 本体自由度与机械参数不同；上游根目录 `LICENCE` 为 MIT，本项目仍只参考公开设计思想，不复制代码/资产 |
| [robot_lab](https://github.com/fan-ziqi/robot_lab) | `500399ed` | 现代 Isaac Lab 扩展组织、PPO 任务注册、Go2W 等轮足任务的域随机化思路 | Go2W 是轮式四足，不是两轮串联腿；直接迁移会改变本实验的控制问题 |

最终采用的原则是：

1. 用 Wheel-Legged-Gym 确认“6 自由度串联轮腿 + PPO”这条路线已有成功先例；
2. 用 Flamingo 与 robot_lab 校验当前 Isaac Lab 的任务拆分方式；
3. 自己建立最小、透明、无外部 mesh 依赖的机器人，避免把机械差异和授权问题一起带进课程仓库；
4. 使用 Isaac Sim 6.0.1 实测 API，而不是机械翻译旧 Isaac Gym 配置。

参考仓库只用于调研，不作为 ActuateX 的运行时依赖，也没有被打包进发布物。

## 3. 机器人与控制契约

### 3.1 机械结构

原创 URDF 位于：

- `robots/wheel_legged/urdf/actuatex_serial_wheel_legged.urdf`
- `robots/wheel_legged/ASSET_NOTICE.md`

主要参数：底盘质量 8.8 kg、上/下腿长约 0.15/0.25 m、单轮半径 0.0675 m、单轮质量 1.22 kg。默认腿姿为髋关节 `0.35 rad`、膝关节 `-0.70 rad`，初始底盘高度为 `0.50 m`。

策略关节顺序是控制契约的一部分：

```text
left_hip, left_knee, right_hip, right_knee, left_wheel, right_wheel
```

四个腿关节由位置目标控制，两个轮关节由速度目标控制。腿位置动作缩放为 `0.45 rad`，轮速动作缩放为 `20 rad/s`，策略输出统一裁剪到 `[-1, 1]`。

### 3.2 28 维观测

| 观测项 | 维度 | 说明 |
|---|---:|---|
| base linear velocity | 3 | 机体系底盘线速度 |
| base angular velocity | 3 | 机体系底盘角速度 |
| projected gravity | 3 | 姿态的重力投影 |
| velocity command | 3 | `vx, vy, yaw_rate`；本任务 `vy=0` |
| leg joint position | 4 | 只放有限位的腿关节相对角度 |
| all joint velocity | 6 | 四个腿关节与两个轮关节，乘 `0.05` |
| previous action | 6 | 上一控制周期动作 |
| **合计** | **28** | 50 Hz 更新 |

轮角度是连续无界量，放进观测会产生随行驶距离增长的无意义状态，因此明确排除；轮速度仍保留。

### 3.3 奖励与终止

环境包含 18 个奖励项，核心分为四组：

- 任务：线速度跟踪、角速度跟踪、命令方向前进量；
- 生存：存活奖励、跌倒终止惩罚；
- 稳定：水平姿态、目标高度、垂向速度、横滚/俯仰角速度、横向滑移；
- 硬件友好：左右腿对称、默认腿姿偏差、轮速运动学先验、力矩、加速度、动作变化、腿碰撞与关节限位。

底盘碰地、倾斜超过 `0.90 rad` 或底盘低于 `0.25 m` 会终止该环境。每个 episode 为 20 s；独立评估把 episode 延长至 40 s，确保 22 s 的测试序列不会被时间截断。

## 4. Isaac Sim 6 迁移中真正遇到的问题

Isaac Sim 6 的 URDF 转换器会把刚体 link 生成在嵌套 prim 中。当前固定的 Isaac Lab 版本在执行 `activate_contact_sensors()` 时沿用旧的层级假设，只给根刚体加上 `PhysxContactReportAPI`，结果腿和轮的 contact sensor 无法解析完整 body 集合。

本仓库没有修改 Isaac Lab 安装目录，而是在进程内安装了一个幂等兼容层：

- `backends/isaac_lab/tinymal_lab/sim6_compat.py`

它遍历 articulation 下所有刚体后代，并对每个刚体调用上游 contact sensor 激活逻辑。修复后，`base_link`、四个腿 link 和两个 wheel link 都能进入 contact view。这一改动是 Sim 6 任务从“资产能显示”走到“奖励与终止真的有接触数据”的关键。

## 5. 两阶段 PPO 训练

### 5.1 第一阶段：平地基线

配置：

- 8192 个并行环境；
- 每环境每轮 24 步；
- 500 轮，共 `8192 × 24 × 500 = 98,304,000` 条 transition；
- actor/critic 均为 `512-256-128` ELU MLP；
- PPO 学习率 `3e-4`、adaptive schedule、entropy `0.005`、初始标准差 `0.3`；
- 中等范围摩擦、底盘质量、质心、PD 增益与周期推力随机化。

正式训练用时 376.08 s。最后一个训练窗口达到 100% episode success，平均 episode 长度 995.6/1000，`vx` 误差 0.0918、yaw 误差 0.1252。

```bash
ISAAC_LAB_ROOT=/path/to/IsaacLab-3

PYTHONPATH=$PWD/backends/isaac_lab \
"$ISAAC_LAB_ROOT/isaaclab.sh" -p \
  backends/isaac_lab/scripts/train_tinymal.py \
  --task Isaac-Velocity-Flat-SerialWheelLegged-v0 \
  --num_envs 8192 --max_iterations 500 --seed 23 \
  --experiment_name wheel_legged_flat_isaacsim6 \
  --run_name formal_seed23_stage1 --save_interval 50 \
  --init_checkpoint \
  artifacts/isaac_lab/logs/rsl_rl/wheel_legged_capacity/\
2026-07-19_19-31-27_env8192_seed11/model_19.pt
```

### 5.2 第二阶段：延迟与宽域随机化

鲁棒阶段从第一阶段 `model_499.pt` 热启动 actor 和 critic，加入：

- 每个环境独立采样 0–4 个 physics tick 的动作延迟，即 0–20 ms；
- 静摩擦 `0.45–1.40`、动摩擦 `0.40–1.30`；
- 底盘质量 `0.85–1.15×`，质心最大偏移约 25 mm；
- 执行器刚度/阻尼 `0.85–1.15×`；
- 每 5–8 s 施加一次外力速度扰动；
- 更保守的固定学习率 `1e-4`、entropy `0.001`、初始标准差 `0.12`。

200 轮共生成 39,321,600 条 transition，用时 164.83 s。两阶段合计 137,625,600 条 transition，正式训练墙钟时间约 9.0 分钟。

```bash
PYTHONPATH=$PWD/backends/isaac_lab \
"$ISAAC_LAB_ROOT/isaaclab.sh" -p \
  backends/isaac_lab/scripts/train_tinymal.py \
  --task Isaac-Velocity-Robust-SerialWheelLegged-v0 \
  --num_envs 8192 --max_iterations 200 --seed 31 \
  --experiment_name wheel_legged_robust_isaacsim6 \
  --run_name delayed20ms_seed31 --save_interval 25 \
  --learning_rate 1e-4 --schedule fixed \
  --entropy_coef 1e-3 --init_noise_std 0.12 \
  --gpu_found_lost_pairs_capacity 16777216 \
  --init_checkpoint \
  artifacts/isaac_lab/logs/rsl_rl/wheel_legged_flat_isaacsim6/\
2026-07-19_19-32-52_formal_seed23_stage1/model_499.pt \
  --init_critic
```

### 5.3 为什么没有强行把 16 GB 显存“吃满”

8192 环境的容量探针约使用 4996 MiB 显存，但 GPU 利用率已约 85%，吞吐约 26.4 万 physics/control samples/s。这个任务只有 8 个刚体和 6 个关节，状态张量小，瓶颈更接近 PhysX 计算与同步，而不是显存容量。

继续增加环境只为了让显存数字接近 100%，不保证提高吞吐，还会增大 broad-phase pair buffer、启动时间和 OOM 风险。因此本任务把“每秒有效样本数、更新稳定性、是否 OOM”作为容量标准，而不是把显存占用率作为目标。

## 6. 独立评估，而不是用训练 reward 自证

每个 checkpoint 都在一个新建、同种子的环境里测试，使用 deterministic actor。命令序列固定为：

```text
stand 2 s
forward 0.5 m/s 4 s
forward 1.0 m/s 4 s
backward -0.5 m/s 4 s
yaw 0.8 rad/s 4 s
arc: vx 0.7 m/s + yaw 0.6 rad/s 4 s
```

排名先比较跌倒数，再比较所有被命令主轴的 RMSE；弧线段同时计入 `vx` 与 yaw，避免只看一条轴。

| 测试 | 第一阶段 `model_499` | 最终 `model_199` | 结果 |
|---|---:|---:|---|
| clean + 0–20 ms delay，256 env | 0 falls；RMSE 0.07101 | 0 falls；RMSE 0.04467 | RMSE 改善 37.1% |
| train randomization + delay，256 env | 1 fall；RMSE 0.09860 | 2 falls；RMSE 0.08025 | 跟踪更好，但单种子跌倒多 1 次 |
| unseen holdout + delay，256 env | 14 falls；RMSE 0.12177 | 3 falls；RMSE 0.09892 | 跌倒减少 78.6%，RMSE 改善 18.8% |
| unseen holdout + delay，1024 env | 23 falls；RMSE 0.11662 | 7 falls；RMSE 0.09817 | 跌倒减少 69.6%，RMSE 改善 15.8% |

大样本 holdout 使用训练范围之外的组合：静摩擦 `0.40–1.45`、动摩擦 `0.35–1.35`、质量 `0.82–1.18×`、质心最大偏移 30 mm、PD 增益 `0.80–1.20×`，并把推力间隔缩短到 4–6 s、强度提高到 `x ±0.70 m/s`、`y ±0.50 m/s`、yaw `±0.50 rad/s`。

这里没有隐藏不利结果：在训练随机化的 256 环境单种子测试中，第一阶段模型确实少跌倒 1 次。但在更强的 256 与 1024 环境 holdout 中，最终模型都明显更稳，同时 clean 性能也更好，因此最终选择 `model_199.pt`，而不是按某一张表挑对自己最有利的 checkpoint。

结果原始 JSON：

- `artifacts/isaac_sim_6/evaluation/wheel_legged_delayed_clean_seed113.json`
- `artifacts/isaac_sim_6/evaluation/wheel_legged_delayed_trainrand_seed103.json`
- `artifacts/isaac_sim_6/evaluation/wheel_legged_delayed_holdout_seed107.json`
- `artifacts/isaac_sim_6/evaluation/wheel_legged_delayed_holdout_1024_seed109.json`

复现大样本 A/B：

```bash
PYTHONPATH=$PWD/backends/isaac_lab \
"$ISAAC_LAB_ROOT/isaaclab.sh" -p \
  backends/isaac_lab/scripts/evaluate_wheel_legged.py \
  --delayed --mode holdout --num_envs 1024 --seed 109 \
  --ckpt \
  artifacts/isaac_lab/logs/rsl_rl/wheel_legged_flat_isaacsim6/\
2026-07-19_19-32-52_formal_seed23_stage1/model_499.pt \
  artifacts/isaac_lab/logs/rsl_rl/wheel_legged_robust_isaacsim6/\
2026-07-19_19-44-28_delayed20ms_seed31/model_199.pt \
  --out artifacts/isaac_sim_6/evaluation/\
wheel_legged_delayed_holdout_1024_seed109.json
```

## 7. 导出与演示

最终部署产物：

- `artifacts/isaac_sim_6/checkpoints/serial_wheel_legged_robust_sim6.jit.pt`
- `artifacts/isaac_sim_6/checkpoints/serial_wheel_legged_robust_sim6.state_dict.pt`
- `artifacts/isaac_sim_6/checkpoints/serial_wheel_legged_robust_sim6.export.json`

导出命令：

```bash
python backends/isaac_lab/scripts/export_wheel_legged_actor.py \
  --checkpoint artifacts/isaac_lab/logs/rsl_rl/\
wheel_legged_robust_isaacsim6/\
2026-07-19_19-44-28_delayed20ms_seed31/model_199.pt \
  --out_dir artifacts/isaac_sim_6/checkpoints \
  --name serial_wheel_legged_robust_sim6
```

演示视频由同一个独立评估器生成，不是手工遥控：

```bash
PYTHONPATH=$PWD/backends/isaac_lab \
"$ISAAC_LAB_ROOT/isaaclab.sh" -p \
  backends/isaac_lab/scripts/evaluate_wheel_legged.py \
  --video --delayed --mode clean --seed 127 \
  --ckpt artifacts/isaac_lab/logs/rsl_rl/wheel_legged_robust_isaacsim6/\
2026-07-19_19-44-28_delayed20ms_seed31/model_199.pt \
  --video_dir artifacts/isaac_sim_6/videos/wheel_legged_ppt \
  --video_prefix ActuateX_SerialWheelLegged_IsaacSim6_PPT
```

成片为 `artifacts/isaac_sim_6/videos/wheel_legged_ppt/ActuateX_SerialWheelLegged_IsaacSim6_PPT-step-0.mp4`：H.264、1280×720、50 FPS、17.5 s、875 帧。视频模式使用位置互相抵消的前进/倒车、左右转和正反弧线，使固定机位下机器人始终留在画面中；数值验收仍使用前述 22 s 长距离六段协议。

## 8. 代码地图

| 文件 | 作用 |
|---|---|
| `robots/wheel_legged/urdf/actuatex_serial_wheel_legged.urdf` | 原创 6-DOF 串联轮腿资产 |
| `backends/isaac_lab/tinymal_lab/wheel_legged_cfg.py` | 普通与随机延迟 actuator 配置 |
| `backends/isaac_lab/tinymal_lab/wheel_legged_env_cfg.py` | 观测、动作、奖励、终止、随机化、场景 |
| `backends/isaac_lab/tinymal_lab/wheel_legged_mdp.py` | 轮速先验、对称性、进度等自定义 MDP 项 |
| `backends/isaac_lab/tinymal_lab/sim6_compat.py` | Sim 6 嵌套刚体 contact sensor 兼容层 |
| `backends/isaac_lab/tinymal_lab/agents/rsl_rl_wheel_legged_ppo_cfg.py` | PPO 网络和优化参数 |
| `backends/isaac_lab/scripts/evaluate_wheel_legged.py` | 多 checkpoint、公平种子、三种评估与录制 |
| `backends/isaac_lab/scripts/export_wheel_legged_actor.py` | TorchScript/state-dict 导出与一致性校验 |
| `robots/wheel_legged/mjcf/actuatex_serial_wheel_legged.xml` | 等质量、惯量与关节契约的 MuJoCo twin |
| `backends/mujoco/wheel_legged_sim2sim.py` | Isaac Sim 策略在 MuJoCo 的评估与录制 |
| `backends/mujoco/sweep_wheel_legged_sim2sim.py` | 指令、延迟、质量和摩擦边界扫描 |

## 9. 下一步，而不是夸大现在的边界

当前完成的是平地速度控制与动力学鲁棒性基线。后续优先级建议为：

1. 加入坡面、台阶、窄梁和离散障碍课程；
2. 按 Wheel-Legged-Gym 的思路增加 VMC/显式高度与腿长中间层，做端到端 PPO 对照；
3. ~~在 MuJoCo 重建同一 6-DOF 资产，做 Isaac Sim → MuJoCo sim2sim；~~ **已完成**：原始迁移 23 次跌倒；加入部署命令斜率限制后 0 跌倒、RMSE 0.05549，完整证据见 [轮腿 sim2sim 报告](./WHEEL_LEGGED_SIM2SIM.zh-CN.md)；
4. 用真实电机阶跃响应辨识延迟、PD、摩擦与饱和参数；
5. 增加电流、温升、轮胎滑移和急停边界，再进入实机低速悬空测试。

其中第 3 项不能只把 URDF 导入 MuJoCo 就宣布成功：必须逐项对齐关节顺序、惯量、碰撞、接触摩擦、50 Hz 动作保持、PD/轮速执行器和观测坐标系，再使用本文同一六段命令协议比较。

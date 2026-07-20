# 倒立摆控制方法全景基准

ActuateX 不把“某一次没倒”当成算法优劣。所有控制器必须使用相同模型、初态、执行器限幅、控制频率、终止条件和随机种子，并分别报告一阶、二阶、三阶结果。

## 为什么不能把所有方法塞进同一张表

当前基准是“直立附近平衡”：初始摆角在直立点附近，目标是保持 10 秒。能量控制起摆、能量整形和切换控制的核心问题却是“从下垂点摆起并捕获直立点”；TVLQR、iLQR、DDP 和轨迹优化通常需要一条时变参考轨迹；卡尔曼滤波、EKF、互补滤波和观测器则是状态估计器，不是独立控制律。

因此项目拆成四条互不混淆的赛道：

1. **Balance**：直立附近平衡，当前 1/2/3 阶 10 秒基准。
2. **Swing-up**：从下垂或大角度初态起摆、捕获并稳定。
3. **Tracking**：跟踪小车位置或预先生成的时变轨迹。
4. **Robust/Sim2Sim**：质量、阻尼、延迟、噪声和物理引擎变化下的鲁棒性。

## 方法矩阵

| 方法 | 类型 | 首要赛道 | 当前状态 | 实现说明 |
|---|---|---|---|---|
| PID | 反馈控制 | Balance | 双后端已实现 | 一阶，带积分限幅，LQR 辅助整定 |
| 串级 PID | 反馈控制 | Balance | 双后端已实现 | 外环小车位置、内环摆角 |
| 极点配置 | 线性状态反馈 | Balance | 双后端已实现 | 各后端原生离散模型，闭环极点显式记录 |
| LQR | 最优线性控制 | Balance | 已实现 | MuJoCo 与 PhysX 分别辨识并设计 |
| LQG | 控制 + 估计 | Balance/Noise | 双后端已实现 | LQR + 稳态卡尔曼滤波，仅测位置/关节角 |
| 能量控制起摆 | 非线性控制 | Swing-up | 已实现 | 单摆能量泵；纯方法会撞轨，负结果保留 |
| 切换控制 | 混合控制 | Swing-up | 已实现 | 能量整形与 LQR 捕获切换，带进入/退出迟滞 |
| MPC | 在线优化 | Tracking/Balance | 双后端已实现首版 | 有限时域线性 MPC；当前仅输入饱和，约束 QP 待补 |
| NMPC | 非线性在线优化 | Swing-up/Tracking | 计划 | CasADi 多重打靶，使用完整非线性动力学 |
| 鲁棒 MPC | 鲁棒在线优化 | Robust | 计划 | 管束/场景 MPC，显式覆盖质量和延迟不确定性 |
| 滑模控制 | 鲁棒非线性控制 | Robust | 双后端已实现 | 离散到达律 + `tanh` 边界层，报告控制抖振 |
| 反馈线性化 | 非线性控制 | Tracking | 双后端已实现 | 一阶小车加速度输入输出线性化 |
| 部分反馈线性化 | 欠驱动控制 | Swing-up | 双后端已实现 | 一阶摆角加速度线性化 + 小车外环 |
| 能量整形 | 非线性控制 | Swing-up | 已实现 | 加入有限轨道的小车位置与速度整形 |
| 自适应控制 | 参数自适应 | Robust | 计划 | 在线质量/质心估计，不使用真值泄漏 |
| H∞ 鲁棒控制 | 鲁棒线性控制 | Robust | 双后端已实现 | 游戏 Riccati 方程；另测质量、延迟与推力矩阵 |
| 轨迹优化 | 离线最优控制 | Swing-up/Tracking | 已实现 | CasADi 多重打靶 + RK4 + IPOPT，显式轨道/力约束 |
| iLQR | 二阶近似优化 | Swing-up/Tracking | 已实现 | 能量控制热启动、动作限幅、正则化和线搜索 |
| DDP | 二阶轨迹优化 | Swing-up/Tracking | 计划 | 与 iLQR 使用相同初始化和预算 |
| TVLQR | 时变线性反馈 | Tracking | 已实现 | 分别围绕多重打靶和 iLQR 轨迹反向求 Riccati |
| 强化学习（PPO） | 数据驱动控制 | 全部 | 已实现 | MuJoCo 与 Isaac Lab 原生训练、双向 Sim2Sim |
| 强化学习（SAC） | 离策略最大熵 RL | Balance/Robust | 已实现首版 | 两后端共用 actor/critic/replay；一阶原生 Isaac/MuJoCo 为 77.93%/64.45% |
| 强化学习（TD3） | 离策略确定性 RL | Balance/Robust | 已实现首版 | 两后端共用 twin-Q 与延迟更新；一阶原生 Isaac/MuJoCo 为 16.50%/20.80% |
| 残差强化学习 | 混合控制 | Robust | 下一阶段 | `u = u_model + u_RL`，残差单独限幅 |
| 卡尔曼滤波 | 状态估计 | Noise | 双后端已实现 | LQG 中的线性稳态 KF |
| EKF | 非线性状态估计 | Noise/Swing-up | 双后端已实现 | 单摆 RK4 非线性预测 + 数值雅可比 |
| 高增益观测器 | 状态估计 | Noise | 双后端已实现 | 独立快速极点组，并保留噪声放大后的失败率 |
| 状态观测器 | 状态估计 | Noise | 双后端已实现 | Luenberger 极点与控制极点分开配置 |
| 互补滤波 | 状态估计 | Noise | 双后端已实现 | 编码器差分速度与模型预测速度融合 |

此外，项目已经实现 **PhysX 非线性动力学上的 GPU 并行 CEM-LQR**：它不改变可解释的线性反馈结构，而是在执行器饱和条件下搜索更大的吸引域，再将教师蒸馏给 PPO 使用的统一 MLP。

## 当前可运行实验

```bash
# MuJoCo：首批传统控制套件，默认每种方法 1024 条 episode
RSL_RL_ROOT=/path/to/rsl_rl-1.0.2 \
python backends/mujoco/evaluate_classical_control_suite.py \
  --orders 1 2 3 --episodes 1024 --num_threads 16

# MuJoCo：从下垂点起摆，要求进入直立区并连续稳定 5 秒
python backends/mujoco/evaluate_swingup_suite.py \
  --episodes 1024 --duration_s 15 --stable_duration_s 5

# 质量、阻尼、执行器、两帧延迟、噪声、双向推力与组合扰动
RSL_RL_ROOT=/path/to/rsl_rl-1.0.2 \
python backends/mujoco/evaluate_robust_control_suite.py \
  --orders 1 2 3 --episodes 256 \
  --methods lqr h_infinity sliding_mode mujoco_ppo

# 非线性多重打靶 / iLQR 轨迹与 TVLQR 闭环跟踪
python backends/mujoco/evaluate_trajectory_control_suite.py \
  --horizon 120 --episodes 1024 --num_threads 16

# Isaac Sim / PhysX：三阶饱和 LQR 的 GPU 并行 CEM 优化
PYTHONPATH=$PWD/backends/isaac_lab /path/to/isaaclab.sh -p \
  backends/isaac_lab/scripts/tune_inverted_pendulum_lqr.py \
  --task Isaac-InvertedPendulum-3-Direct-v0 \
  --base_checkpoint artifacts/checkpoints/inverted_pendulum/isaac_lqr_seed_order_3.pt \
  --population 128 --episodes_per_candidate 256 --generations 24 \
  --validation_episodes_per_candidate 1024 --device cuda:0

# Isaac Sim / PhysX：39 格 GPU 传统控制矩阵
PYTHONPATH=$PWD/backends/isaac_lab /path/to/isaaclab.sh -p \
  backends/isaac_lab/scripts/evaluate_classical_control_suite.py \
  --orders 1 2 3 --episodes 1024 --device cuda:0

# 两后端共用的 SAC/TD3；checkpoint 可直接做双向 Sim2Sim
python backends/mujoco/train_inverted_pendulum_off_policy.py \
  --algorithm sac --order 1 --total-transitions 2000000 --device cuda:0

PYTHONPATH=$PWD/backends/isaac_lab /path/to/isaaclab.sh -p \
  backends/isaac_lab/scripts/train_inverted_pendulum_off_policy.py \
  --algorithm td3 --task Isaac-InvertedPendulum-1-Direct-v0 \
  --total_transitions 5000000 --device cuda:0
```

## 公平性规则

- 执行器统一为小车最大 `±20 N`，控制频率统一为 `60 Hz`。
- 直立平衡统一为 `10 s`；越过轨道边界或任意杆绝对角超过 `π/2` 即失败。
- 不用训练成功率代替独立评估；正式表至少使用 1024 条 episode。
- 调参集与最终验证集分离，固定随机种子写入 JSON。
- 估计器只能读取声明的测量量；不得把仿真真值偷偷传给“输出反馈”算法。
- Sim2Sim 不重新排列观测或手工修正动作；同一 checkpoint 原样运行。
- 同时报成功率、平均/中位存活时间、角度 RMSE、小车 RMSE 和动作幅值。
- 失败结果不删除，并区分“算法失效”“参数未收敛”和“任务不适用”。

## 下一轮实现顺序

1. 引入残差 PPO：CEM-LQR/能量控制为基座，策略只学习受限残差。
2. 将已经完成的多重打靶、iLQR 与 TVLQR 扩展成在线 NMPC，并补完整 DDP 二阶动力学项。
3. 在已经完成的扰动矩阵上加入管束/场景鲁棒 MPC 与自适应控制。
4. 将估计器矩阵扩展到起摆和参数失配，而不只在直立小角度测试。

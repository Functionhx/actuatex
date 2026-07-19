# 1–3 阶倒立摆：传统控制、强化学习与双向 Sim2Sim 基准

> 实验日期：2026-07-19
> 物理后端：MuJoCo 3.10.0、Isaac Sim 6.0.1 GA / PhysX
> 任务框架：Isaac Lab 3.0.0-beta2.patch1
> 正式评估：学习策略与 Sim2Sim 使用 4096 回合；传统控制套件使用 1024 回合

## 结论先行

ActuateX 已经完成一套可运行、可复核的 1、2、3 阶串联倒立摆基准，而不是只生成三个外观模型：

- MuJoCo 和 Isaac Sim 各有原生动力学环境、训练入口、评估入口和原生策略；
- 两个后端共用 14 维观测、1 维动作、控制频率、初态、限幅和终止条件；
- 传统控制已在两端同步：GPU 版 PID、LQR、MPC、LQG、H∞、滑模、反馈线性化及五类估计器与 NumPy 参考逐步对齐，并完成 1024 个配对初态评估；
- 一阶和二阶 PPO 在两个原生后端都接近或达到 100%，双向迁移仍保持 95.92%–100%；
- 共用实现的 SAC/TD3 已在两个后端完成首轮正式训练和双向迁移；当前最佳离策略结果是 Isaac-SAC 原生 77.93%、迁移 MuJoCo 77.20%，仍明显低于 PPO/LQR；
- 三阶满量程明显更难：MuJoCo 原生 PPO 为 40.87%，Isaac 原生 PPO 为 49.51%；
- 在三阶 PhysX 中，GPU 并行 CEM 直接优化饱和 LQR 后，教师控制器达到 68.65%；蒸馏 MLP 为 52.69%，再经 PPO 微调为 49.51%，因此教师仍是当前最优，PPO 微调没有带来提升；
- 三阶跨引擎迁移几乎失效，说明低阶通过不等于多摆动力学已经对齐。

这组结果的价值恰恰在于同时保留成功与失败：一、二阶证明软件接口和基本动力学约定正确；三阶暴露非线性吸引域、饱和控制和引擎模型差异。

## “用 Isaac Sim 做倒立摆”具体指什么

Isaac Sim 是仿真运行时，负责 USD 场景、PhysX 动力学、渲染和传感器；Isaac Lab 是构建训练任务的 Python 框架，负责并行环境、reset、观测、奖励、终止条件以及 RL 算法适配。

因此本项目的 Isaac 实现是：

```text
Isaac Sim 6.0.1 GA（运行时与 PhysX）
└── Isaac Lab DirectRLEnv（任务与并行环境）
    └── RSL-RL PPO（训练器）
```

三项正式任务分别是：

- `Isaac-InvertedPendulum-1-Direct-v0`
- `Isaac-InvertedPendulum-2-Direct-v0`
- `Isaac-InvertedPendulum-3-Direct-v0`

任务注册位于 [`tinymal_lab/__init__.py`](../backends/isaac_lab/tinymal_lab/__init__.py)，核心环境位于 [`inverted_pendulum_env.py`](../backends/isaac_lab/tinymal_lab/inverted_pendulum_env.py)。

## 统一实验契约

后端无关契约定义在 [`contract.py`](../tasks/inverted_pendulum/contract.py)。两个引擎不能各自选择“更容易”的设置。

| 项目 | 统一设置 |
|---|---:|
| 仿真步长 | 1/120 s |
| 策略降采样 | 2 |
| 控制频率 | 60 Hz |
| 单回合时长 | 10 s |
| 动作 | 小车水平力，1 维 |
| 力限幅 | ±20 N |
| 轨道范围 | ±2.4 m |
| 小车终止边界 | ±2.2 m |
| 摆杆终止角 | 任一杆绝对角超过 π/2 |
| 初始角范围 | 一阶 ±0.25、二阶 ±0.16、三阶 ±0.10 rad |
| 小车/单杆质量 | 1.0 / 0.20 kg |
| 单杆长度 | 0.60 m |

统一 14 维观测布局为：

```text
[cart_x, cart_v,
 sin(theta_1), cos(theta_1), omega_1,
 sin(theta_2), cos(theta_2), omega_2,
 sin(theta_3), cos(theta_3), omega_3,
 pole_1_mask, pole_2_mask, pole_3_mask]
```

低阶任务将未使用的摆杆槽置零，并用 mask 标记阶数。这样同一 MLP 结构既能按 1→2→3 阶热启动，也能不改网络地进行双向 Sim2Sim。

## 物理审计：先证明模型真的“倒立”

早期 Isaac 结果出现过不可信的 100%：模型导入后实际处于重力稳定的下垂构型，而不是不稳定的直立构型。该结果已作废，没有进入正式表格。

修复和复核包括：

1. 修正 Sim 6 URDF 导入所需的串联杆 Z 向几何与关节偏移，使导入后的世界坐标几何朝上；
2. 将 Isaac 铰链轴统一为 `0 -1 0`，复核正角、正角速度和控制力的符号；
3. 移除接触固定轨道的地面平面，避免固定基座与地面接触向 PhysX 注入非预期冲量；
4. 提高 PhysX 位置与速度求解迭代次数；
5. 在直立点做有限差分辨识，检查开环离散系统确实含有模大于 1 的不稳定极点；
6. 对照 MuJoCo 线性化增益、短时状态响应和动作方向，再开始训练。

资产由 [`generate_inverted_pendulum_assets.py`](../scripts/generate_inverted_pendulum_assets.py) 生成，避免人工维护三套逐渐漂移的 URDF/MJCF。

## 已实现的控制与训练方法

### 传统控制

[`classical_control.py`](../tasks/inverted_pendulum/classical_control.py) 当前提供：

- PID 与串级 PID；
- 离散极点配置；
- 离散 LQR；
- 有限时域线性 MPC；
- 稳态卡尔曼滤波与 LQG；
- 带 ±20 N 饱和的全状态反馈。

所有方法由 [`evaluate_classical_control_suite.py`](../backends/mujoco/evaluate_classical_control_suite.py) 在相同种子和初态上统一评估。

### PPO 与课程学习

MuJoCo 与 Isaac 使用相同 MLP 结构。Isaac 训练入口 [`train_tinymal.py`](../backends/isaac_lab/scripts/train_tinymal.py) 新增了：

- 初始角范围缩放；
- 单进程多阶段角度课程；
- critic 单独预热；
- actor/critic checkpoint 热启动；
- 可控学习率、初始探索标准差和最终 checkpoint 路径。

三阶曾尝试 `0.25 → 0.50 → 0.75 → 1.00` 的朴素课程。满量程成功率先升后降，最终候选仅约 15%–22%，所以停止了无效长跑，并保留日志作为负结果。正式三阶策略改为“PhysX CEM-LQR 教师蒸馏 → critic 预热 → 小学习率 PPO”。

### GPU 并行 CEM-LQR

[`tune_inverted_pendulum_lqr.py`](../backends/isaac_lab/scripts/tune_inverted_pendulum_lqr.py) 不把控制器变成黑盒：它保留 `u = -Kx` 的可解释结构，只在 PhysX 完整非线性动力学和动作饱和下，用 Cross-Entropy Method 搜索各增益的乘性尺度。

正式配置为：

- 128 个候选；
- 每候选 256 个相同分布初态，即每代 32768 个并行环境；
- 24 代搜索，16 个精英；
- 独立 1024 初态归档验证；
- 最后将最佳教师蒸馏到 PPO 兼容 MLP。

最佳三阶 PhysX 增益为：

```text
[-0.7067, -54.3671, 235.0339, -506.1034,
 -2.2663, -15.4870, -13.2638, -45.8408]
```

## GPU 利用率与吞吐

RTX 4070 Ti 的 12 GB/16 GB 级显存占用并不自动等于低利用率。实际瓶颈主要是 PhysX 计算和 kernel 调度，因此优先增大并行环境，直到吞吐不再增长，而不是机械追求显存 100%。

| 任务 | 并行环境 | 实测吞吐 | 典型 GPU 状态 |
|---|---:|---:|---:|
| 一阶 PPO | 65,536 | 约 1.34 M step/s | 高占用 |
| 二阶 PPO | 49,152 | 约 1.02–1.18 M step/s | 约 86% |
| 三阶 PPO | 40,960 | 约 0.91–1.02 M step/s | 约 90% |
| 三阶 CEM | 32,768 | 每代 128×256 同步评估 | 约 88% |

继续堆环境会增加内存、reset 和同步开销，不一定增加样本吞吐；上述容量来自实测，而非静态估算。

## 正式结果

### 原生 PPO：4096 回合满量程评估

| 原生后端 | 阶数 | 成功率 | 平均存活 | 绝对摆角 RMSE | 小车 RMSE |
|---|---:|---:|---:|---:|---:|
| MuJoCo | 1 | **100.00%** | 10.000 s | 0.0206 rad | 0.2347 m |
| MuJoCo | 2 | **99.98%** | 9.998 s | 0.0448 rad | 0.6995 m |
| MuJoCo | 3 | **40.87%** | 4.568 s | 0.3549 rad | 0.5906 m |
| Isaac / PhysX | 1 | **100.00%** | 9.983 s | 0.0205 rad | 0.1900 m |
| Isaac / PhysX | 2 | **100.00%** | 9.983 s | 0.0355 rad | 0.9790 m |
| Isaac / PhysX | 3 | **49.51%** | 5.323 s | 0.1494 rad | 0.5480 m |

Isaac 的最大成功时长显示为 9.983 s，是 60 Hz 计步与 timeout 记录边界造成的一帧差异，不代表提前失败。

### 一阶离策略 RL：SAC 与 TD3 首轮正式评估

SAC 与 TD3 使用同一套后端无关 PyTorch 实现、14 维观测和 1 维动作。Isaac 每种算法采集 500 万 transition，MuJoCo 每种算法采集 200 万 transition；评估均为独立的 4096 回合。下表中的箭头表示 checkpoint 原样迁移，不做动作修正或再训练。

| 训练后端 / 算法 | 原生成功率 | Sim2Sim 成功率 | 原生平均存活 | 当前判断 |
|---|---:|---:|---:|---|
| Isaac SAC | **77.93%** | Isaac→MuJoCo **77.20%** | 9.536 s | 部分收敛；角度稳但小车漂移 |
| Isaac TD3 | 16.50% | Isaac→MuJoCo 21.12% | 7.129 s | 部分收敛；训练后段仍波动 |
| MuJoCo SAC | **64.45%** | MuJoCo→Isaac **65.50%** | 8.811 s | 部分收敛；跨引擎下降很小 |
| MuJoCo TD3 | 20.80% | MuJoCo→Isaac 17.16% | 7.555 s | 部分收敛；位置漂移明显 |

MuJoCo 首轮曾得到 SAC/TD3 均为 0% 的异常结果。审计发现环境复用 `obs_buf`，而原训练循环在 `env.step()` 之后才写 replay，使 `s_t` 被原地覆盖成 `s_{t+1}`。修复为 step 前显式克隆状态后，在完全相同的样本预算下，SAC/TD3 分别恢复到 64.45%/20.80%；错误产物以 `pre_alias_fix` 后缀保留。这说明离策略 RL 的 replay 完整性必须作为独立测试项，不能把采样管线错误误判为算法失败。

同一任务上的成熟基线中，MuJoCo/Isaac PPO 与一阶 LQR、极点配置、长时域 MPC 均为 100%。因此当前数据不支持“换成更先进的 RL 名字就一定更强”：PPO 已收敛，SAC/TD3 仍对奖励尺度、采样/更新比和超参数敏感。Isaac-SAC 的摆角 RMSE 只有 0.0133 rad，但小车位置 RMSE 达 0.855 m；其策略以很小动作维持摆角，却没有充分抑制慢速位置漂移。下一轮将增加位置代价、分阶段初态与模型控制残差基座，而不是只延长失败训练。

### MuJoCo 传统控制：1024 回合

下表报告成功率；`MPC-H` 中 H 是离散预测步数。

| 阶数 | 极点配置 | LQR | MPC-40 | MPC-120 | MPC-240 | LQG | PID | 串级 PID |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100.00% | 100.00% | 40.63% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| 2 | 100.00% | 100.00% | 0.00% | 78.91% | 100.00% | 81.05% | — | — |
| 3 | 19.73% | **42.48%** | 0.00% | 19.82% | 41.80% | 15.53% | — | — |

这张表揭示两个重要事实：

- 预测时域不是装饰参数。短时域 MPC 看不到小车缓慢漂出轨道，H=40 在二、三阶完全失败；H=240 才逐渐接近无限时域 LQR。
- 三阶满量程已超出局部线性控制器的大部分吸引域。LQR 的 42.48% 与 MuJoCo PPO 的 40.87% 接近，不能宣称 RL 已全面超过传统控制。

LQG 只读取带噪声的小车位置和相对关节角，测量噪声标准差为 0.002；它没有偷用速度真值。

### Isaac / PhysX 传统控制：1024 个跨后端配对初态

Isaac 端不是把 NumPy 控制器逐帧搬回 CPU，而是使用经过逐动作一致性测试的 GPU 张量实现。初态采用与 MuJoCo 完全相同的随机种子和逐环境抽样顺序；状态反馈使用 PhysX 原生有限差分辨识得到的离散模型。下表为成功率：

| 阶数 | 极点配置 | LQR | H∞ | 滑模 | MPC-40 | MPC-120 | MPC-240 | LQG | 互补滤波 | CEM-LQR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100.00% | 100.00% | 100.00% | 100.00% | 59.08% | 100.00% | 100.00% | 100.00% | 100.00% | — |
| 2 | 100.00% | 100.00% | 100.00% | 45.90% | 0.59% | 98.34% | 100.00% | 86.62% | 98.24% | — |
| 3 | 12.70% | 54.00% | 36.62% | 0.00% | 0.00% | 31.05% | 53.61% | 28.42% | 33.01% | **66.99%** |

一阶还正式评估了状态观测器、高增益观测器、EKF、反馈线性化、部分反馈线性化、PID 与串级 PID，均为 100%。二阶状态观测器/高增益观测器分别为 64.75%/86.52%。完整 39 格结果保存在 `classical_control_suite_isaac.json`；生成入口为 [`evaluate_classical_control_suite.py`](../backends/isaac_lab/scripts/evaluate_classical_control_suite.py)。

配对结果说明“仿真器更先进”不自动等于控制器更强或更弱：一、二阶结构结论在两个引擎一致；三阶普通 LQR 在 PhysX/MuJoCo 分别为 54.00%/42.48%，但 PhysX 优化的 CEM-LQR 原样迁移到 MuJoCo 仅 0.78%。局部可控性可以一致，饱和后的大范围吸引域仍高度依赖接触之外的积分器、关节约束和惯量细节。

### 三阶 PhysX 教师、蒸馏与 PPO

| 控制器 | 评估后端 | 回合 | 成功率 | 平均存活 |
|---|---|---:|---:|---:|
| CEM 优化的直接饱和 LQR 教师 | Isaac | 1024 | **68.65%** | 7.072 s |
| 教师蒸馏 MLP | Isaac | 4096 | **52.69%** | 5.621 s |
| 蒸馏后 PPO 微调 | Isaac | 4096 | **49.51%** | 5.323 s |

蒸馏误差虽低至 MSE `6.81e-5`，但闭环系统会累积细小动作误差，因此成功率仍下降约 16 个百分点。PPO 微调又下降 3.18 个百分点。这是当前明确的改进方向：残差策略应在教师之上学习受限修正，而不是让完整 actor 漂离教师。

### 双向 Sim2Sim：同一 checkpoint 原样迁移

| 策略来源 | 评估后端 | 一阶 | 二阶 | 三阶 |
|---|---|---:|---:|---:|
| MuJoCo PPO | Isaac / PhysX | **100.00%** | **99.73%** | **0.00%** |
| Isaac PPO | MuJoCo | **100.00%** | **95.92%** | **4.08%** |

迁移时没有重新训练、重排观测、反转动作或人工选择有利初态。三阶结果说明两个引擎在多体串联耦合、关节惯量、约束稳定化和饱和后的闭环轨迹上仍有显著差异。PhysX CEM 教师进入 MuJoCo 也只有 0.78%，进一步证明它优化的是 PhysX 非线性吸引域，而不是跨引擎鲁棒性。

### 一阶下垂起摆：1024 个配对初态

起摆是独立赛道：初始角为 `π ± 0.04 rad`，最大运行 15 秒；成功要求进入 `|θ| < 0.30 rad` 的直立捕获区，并连续保持 5 秒。三种方法使用完全相同的 1024 个初态。

| 控制方法 | 进入直立区 | 连续稳定 5 s | 撞轨率 | 平均首次到达 |
|---|---:|---:|---:|---:|
| 纯能量泵 | 0.00% | 0.00% | 100.00% | — |
| 能量 + 小车位置/速度整形 | **100.00%** | 0.00% | 0.00% | 1.633 s |
| 能量整形 ↔ LQR 迟滞切换 | **100.00%** | **100.00%** | 0.00% | 1.633 s |

这不是三行“换名字”的控制器。纯能量控制只关心摆杆能量，有限轨道上会把小车推出边界；加入小车状态整形后能够起摆但不会自动停在不稳定平衡点；只有再切换到局部 LQR，才能完成“起摆 + 捕获 + 稳定”的完整任务。实现见 [`swingup_control.py`](../tasks/inverted_pendulum/swingup_control.py)，正式结果由 [`evaluate_swingup_suite.py`](../backends/mujoco/evaluate_swingup_suite.py) 生成。

### 非线性轨迹优化、iLQR 与 TVLQR：1024 个配对初态

CasADi 多重打靶在完整非线性 RK4 模型上求解 120 步（2 秒）轨迹，显式约束 `|x|≤2.2 m` 与 `|u|≤20 N`。IPOPT 用 46 次迭代求解成功，轨迹最大位移 1.088 m。iLQR 使用能量控制轨迹热启动，带动作裁剪、正则化和线搜索；100 次迭代后的目标值为 3.882，接近多重打靶的 3.867。

| 方法 | 进入直立区 | 连续稳定 5 s | 平均首次到达 |
|---|---:|---:|---:|
| 多重打靶轨迹开环回放 | 0.00% | 0.00% | — |
| 多重打靶轨迹 + TVLQR | **100.00%** | **100.00%** | **0.771 s** |
| iLQR 轨迹开环回放 | 0.00% | 0.00% | — |
| iLQR 轨迹 + TVLQR | **100.00%** | **100.00%** | 0.792 s |
| 能量整形 ↔ LQR | **100.00%** | **100.00%** | 1.632 s |

开环轨迹在随机初态下完全失败，而同一轨迹加 TVLQR 后全部成功，说明优化器提供的是名义轨迹，反馈才提供局部鲁棒性。两类优化轨迹比能量法约快 0.84 秒到达，但平均动作更大。实现位于 [`trajectory_optimization.py`](../tasks/inverted_pendulum/trajectory_optimization.py)。

可直接用于 PPT 的 1280×720、60 FPS、H.264 演示视频生成在 `artifacts/mujoco/videos/ActuateX_InvertedPendulum_Swingup_TVLQR_PPT.mp4`；视频包含优化轨迹/终端 LQR 阶段、时间、控制力、小车位置和摆角叠加信息。录制入口是 [`record_inverted_pendulum_swingup.py`](../backends/mujoco/record_inverted_pendulum_swingup.py)。

### 仅位置测量的估计器：1% 噪声，1024 回合

控制器统一为 LQR，估计器只读取带噪声的小车位置和相对关节角；速度真值不可见。

| 阶数 | LQG/KF | Luenberger | 高增益观测器 | 互补滤波 | EKF |
|---:|---:|---:|---:|---:|---:|
| 1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| 2 | 70.90% | 58.98% | 79.88% | **96.09%** | — |
| 3 | 0.00% | 0.00% | 0.00% | 0.00% | — |

三阶结果不能解释成“卡尔曼滤波不好”，而是当前局部模型、噪声水平和三阶狭窄吸引域的组合使输出反馈闭环失效。高增益观测器在二阶更好，却在三阶最快失败，体现了加快收敛与放大噪声之间的典型权衡。

### H∞、滑模与反馈线性化：名义 1024 回合

| 方法 | 一阶 | 二阶 | 三阶 |
|---|---:|---:|---:|
| H∞ 状态反馈 | 100.00% | 95.21% | 41.31% |
| 离散滑模 | 100.00% | 41.70% | 13.48% |
| 小车输入输出反馈线性化 | 100.00% | — | — |
| 摆角部分反馈线性化 + 小车外环 | 100.00% | — | — |

完整静态反馈线性化不适用于这个 4 状态、1 输入的欠驱动系统，因此项目明确实现并命名为两个输入输出/部分反馈线性化版本，而没有伪造一个“全状态完全线性化”。

### 九种扰动下传统控制与 PPO：每格 256 个配对初态

扰动包括轻/重摆杆、三倍关节阻尼、20% 执行器衰减、两帧动作延迟、传感噪声、±5 N 双向推力和组合失配。下表是九种场景成功率的宏平均。

| 阶数 | LQR | MuJoCo PPO | H∞ | 滑模 |
|---:|---:|---:|---:|---:|
| 1 | 100.00% | 100.00% | 100.00% | 100.00% |
| 2 | **95.27%** | 94.57% | 78.95% | 33.38% |
| 3 | **32.47%** | 29.56% | 26.78% | 3.21% |

二阶组合失配下 LQR/PPO/H∞ 分别为 92.97%/91.02%/56.25%；三阶为 13.28%/7.03%/0%。两帧延迟使三阶四种方法全部降到 0%，也是下一轮鲁棒 MPC、延迟观测和残差 RL 必须共同解决的核心问题。

H∞ 在这里并没有“因为名字鲁棒就获胜”：当前游戏 Riccati 方程针对显式状态扰动通道设计，没有把未知纯时延纳入广义对象。这个负结果是正确的控制理论边界，而不是实现时应当隐藏的数据。

## 如何复现

### MuJoCo 原生 PPO

```bash
RSL_RL_ROOT=/path/to/rsl_rl-1.0.2 \
python backends/mujoco/train_inverted_pendulum.py \
  --order 3 --num_envs 4096 --max_iterations 1500

RSL_RL_ROOT=/path/to/rsl_rl-1.0.2 \
python backends/mujoco/evaluate_inverted_pendulum.py \
  --order 3 \
  --checkpoint artifacts/checkpoints/inverted_pendulum/mujoco_order_3.pt \
  --policy-source mujoco_ppo --episodes 4096
```

### Isaac 原生 PPO

```bash
export PYTHONPATH="$PWD/backends/isaac_lab${PYTHONPATH:+:$PYTHONPATH}"

/path/to/IsaacLab/isaaclab.sh -p \
  backends/isaac_lab/scripts/train_tinymal.py \
  --task Isaac-InvertedPendulum-3-Direct-v0 \
  --num_envs 40960 --max_iterations 20 \
  --init_checkpoint \
    artifacts/checkpoints/inverted_pendulum/isaac_cem_lqr_seed_order_3.pt \
  --critic_warmup_iterations 15 --learning_rate 1e-7 \
  --init_noise_std 0.002 \
  --output_checkpoint \
    artifacts/checkpoints/inverted_pendulum/isaac_lab_order_3.pt
```

### 传统控制套件

```bash
python backends/mujoco/evaluate_classical_control_suite.py \
  --orders 1 2 3 --episodes 1024 --num_threads 16 \
  --output \
    artifacts/inverted_pendulum/evaluation/classical_control_suite_mujoco.json

# 独立的下垂起摆赛道
python backends/mujoco/evaluate_swingup_suite.py \
  --episodes 1024 --num_threads 16 \
  --duration_s 15 --stable_duration_s 5 \
  --out artifacts/inverted_pendulum/evaluation/swingup_suite_mujoco.json

# 非线性轨迹优化、iLQR 与 TVLQR
python backends/mujoco/evaluate_trajectory_control_suite.py \
  --horizon 120 --episodes 1024 --num_threads 16

# 模型失配、延迟、噪声与外力矩阵（同时包含 PPO）
RSL_RL_ROOT=/path/to/rsl_rl-1.0.2 \
python backends/mujoco/evaluate_robust_control_suite.py \
  --orders 1 2 3 --episodes 256 \
  --methods lqr h_infinity sliding_mode mujoco_ppo
```

正式原始 JSON 位于 `artifacts/inverted_pendulum/evaluation/`，checkpoint 位于 `artifacts/checkpoints/inverted_pendulum/`。二进制 checkpoint 和批量运行产物默认不纳入 Git；代码、契约和报告本身可以独立审查。

## 下一阶段：把完整方法清单逐项变成实验

所有方法的适用赛道、实现状态和公平性规则见 [倒立摆控制方法全景基准](./INVERTED_PENDULUM_CONTROL_METHODS.zh-CN.md)。接下来的落地顺序是：

1. **Swing-up**：能量控制、能量整形、带迟滞的 LQR 捕获切换；
2. **Hybrid RL**：`u = u_model + u_RL` 的残差强化学习，限制残差幅值；
3. **Trajectory**：轨迹优化、iLQR、DDP、TVLQR，以及完整非线性 NMPC；
4. **Robust**：鲁棒 MPC、滑模、自适应与 H∞，统一质量、阻尼、延迟、外力扰动；
5. **Estimation**：KF、EKF、Luenberger、高增益观测器和互补滤波，与控制器形成二维组合实验。

起摆、平衡、跟踪和状态估计不会被强行塞进同一张排行榜。例如，能量控制从下垂点起摆才有意义；卡尔曼滤波必须和控制器组合评估；TVLQR 必须围绕一条时变参考轨迹。项目会给每种方法它应有的赛道，同时保持相同动力学与评估预算。

## 当前判断

一、二阶已经形成可靠的传统控制、RL 和双向 Sim2Sim 教学基线。三阶还不能称为“解决”：当前 PhysX 最佳教师为 68.65%，最佳统一 MLP 为 52.69%，跨引擎迁移接近失败。

因此下一步不是盲目延长 PPO，而是结合模型控制的强先验与 GPU 并行搜索：先扩大可解释教师的非线性吸引域，再用残差 RL、域随机化和跨引擎联合验证学习真正有用的补偿。

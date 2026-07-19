# Isaac Sim 6.0.1 GA 迁移、训练与验收报告

> 完成日期：2026-07-19（Asia/Shanghai）
> 硬件：NVIDIA GeForce RTX 4070 Ti SUPER 16 GB，驱动 `580.159.03`
> 结论口径：仿真链路与 MID360 官方逐点扫描表复现已验收；相机/MID360 实机统计标定、Livox UDP 包级仿真和真机部署尚未验收。

## 1. 结论先行

TinyMal 已完成从旧 Isaac Lab / Isaac Sim 5.1 到 **Isaac Sim 6.0.1 GA + Isaac Lab `v3.0.0-beta2.patch1`** 的迁移、真实 runtime 启动、原生微调、公平复评、双向 sim2sim、RTX LiDAR、标定相机 ROS writer、Nav2 接口和楼梯视频录制。

本轮最终保留两套策略，而不是强行用一个网络包办所有任务：

| 策略 | 用途 | 最终选择 | 主要结果 |
|---|---|---|---|
| 通用鲁棒策略 | 站立、前后/横向/转向命令与随机动力学 | Sim 6 微调 `model50` | 两个种子平均主轴 RMSE `0.09445`，较热启动模型降低 `1.04%`，重置数没有恶化 |
| 台阶鲁棒策略 | 5 × 20 mm 台阶、随机动力学、延迟、噪声和推扰 | Sim 6 微调 `model10` | 256 次随机试验中无重置登顶 `212/256 = 82.81%`，首次登顶 `236/256 = 92.19%` |

PPT 成片为 [`TinyMal_IsaacSim_6_FinalPolicy_Stairs_PPT.mp4`](../artifacts/isaac_sim_6/videos/TinyMal_IsaacSim_6_FinalPolicy_Stairs_PPT.mp4)：H.264、`1280×720`、50 fps、4.0 秒。它来自最终 `model10` 的无随机展示 rollout；原始 5 秒录像和对应验收 JSON 均保留。

需要诚实说明三点：

1. 两个最终策略都是从迁移前的强策略热启动后在 Sim 6 原生微调，不是从随机权重重新训练；
2. 通用策略在 Isaac Sim 6 内部有小幅、可复现的提升，但直接放进 MuJoCo 仍存在横移/转向跌倒和“存活但不跟踪命令”；
3. MID360 已复现官方非重复角度/时间序列和 ROS 点布局，但没有实机噪声、强度、丢点与网络包标定，不能写成完成了全链路传感器数字孪生；示例相机标定同样不能冒充真实相机。

所有最终路径、哈希、参数和证据索引集中在 [`manifest.json`](../artifacts/isaac_sim_6/manifest.json)。

## 2. Isaac Sim、Isaac Gym 和 Isaac Lab 不是同一个东西

| 名称 | 定位 | 本项目中的作用 |
|---|---|---|
| Isaac Gym | 较早的 GPU 并行物理仿真与 RL API | 保留历史原生训练策略和跨引擎基线 |
| Isaac Sim | 基于 Omniverse/Kit、USD、PhysX 和 RTX 的完整机器人仿真器 | 当前 6.0.1 主仿真环境，负责物理、渲染、传感器和 ROS 2 |
| Isaac Lab | 构建在 Isaac Sim 上的机器人学习任务框架 | 定义 TinyMal 的 MDP、随机化、PPO、评估和训练入口 |

“Isaac Lab 更先进”主要意味着框架、传感器、场景组织和维护状态更先进，不代表旧权重换个运行时就自动更稳。策略表现仍由观测、动作、关节顺序、控制周期、执行器、碰撞几何、求解器和训练分布共同决定。升级任何一项而未对齐，其影响都可能大于算法改进。

因此项目没有删除 Isaac Gym 或 MuJoCo 版本，而是让三套后端各自训练、各自验收，再做双向 sim2sim。结果证明这种做法是必要的：Sim 6 内部更好的 checkpoint 并没有自动成为最好的 MuJoCo checkpoint。

## 3. 精确软件与二进制身份

| 组件 | 实际版本 | 验证状态 |
|---|---|---|
| Isaac Sim | `6.0.1 GA`，官方标签 `v6.0.1` | 官方 Linux ZIP 已完整下载并通过发布 MD5 |
| Linux ZIP | `isaac-sim-standalone-6.0.1-linux-x86_64.zip` | MD5 `65e2c2e83e2461ce0f33b0732d0ee4a3` |
| Isaac Lab | `v3.0.0-beta2.patch1`，commit `ffff603eafc6b74264a5261cc0183d6a65390d78` | 官方匹配 Sim 6.0.1；Lab 标签本身仍是 beta |
| Python | `3.12` | 使用 Sim 自带解释器 |
| PyTorch / CUDA wheel | `2.10.0+cu128` | CUDA 张量运算通过 |
| torchvision | `0.25.0+cu128` | 导入通过 |
| RSL-RL | `rsl-rl-lib 5.0.1` | actor/critic 新配置与训练通过 |

官方 ZIP 解压后的 `VERSION` 文件写作 `6.0.1-rc.7+release.42383.32955d8d.gl`。这是 GA 压缩包内部保留的构建标识；本报告根据“官方文件名 + 官方发布 MD5”确认二进制身份，不把内部 `rc.7` 字符串误判为另一个下载版本。

新旧运行时安装在彼此独立的目录，迁移过程没有覆盖旧环境。用户已同意 EULA；启动时设置 `PRIVACY_CONSENT=false`，并关闭匿名遥测、开放端点、性能、个性化和使用数据开关。

## 4. 迁移中真正修改了什么

### 4.1 安装与启动链

- 安装器锁定 Sim 6.0.1 与 Lab 的精确 tag/commit；
- `_isaac_sim` 只链接到新 runtime，遇到已有链接时拒绝覆盖；
- 修复 Lab 安装命令在 Sim 6 包元数据下的兼容判断；
- `isaaclab.sh` 增加 standalone Sim 6 的环境回退；
- MoviePy/Pillow 组合已修复，H.264 录像实际编码通过；
- Lab 安装器回归测试 `10/10` 通过。

### 4.2 Isaac Lab 3 API 与 RSL-RL 5

| 迁移点 | 处理方式 |
|---|---|
| WXYZ/XYZW 四元数差异 | 初始姿态、ROS 坐标和相机安装姿态显式转换 |
| Warp/ProxyArray | 运行时读取统一转为 Torch 视图 |
| 状态写入 API | 使用 Lab 3 的索引与写回语义 |
| PhysX 配置位置 | 使用新的 `SimulationCfg.physics=PhysxCfg(...)` |
| RSL-RL 网络 schema | 独立 actor、critic 和 Gaussian distribution 配置 |
| 非对称 critic | actor 读取 48 维本体观测；critic 读取 policy + 21 维 privileged，共 69 维 |
| checkpoint 热启动 | actor 严格对齐；只有形状兼容的 69 维 critic 才加载 |
| checkpoint 公平比较 | 每个 checkpoint 重建环境并复位同一种子的全部随机流 |

最后一项尤其重要。旧评估器在同一个环境实例中顺序评估多个 checkpoint，后面的模型会看到已经推进过的随机数流。修复后，所有候选都面对相同摩擦、质量、质心、PD、延迟、噪声和推扰序列；报告只使用修复后的 `*_reseeded.json`。

### 4.3 碰撞几何没有为了“能启动”而缩水

旧 Gym/Lab 实验会把 URDF 中四个 hip cylinder 替换为 capsule。Sim 6 删除了这个 importer 开关；直接重新导入会静默改变物理实验。

默认资产因此改为 [`capsule_compat/tinymal.usd`](../robots/tinymal/usd/capsule_compat/tinymal.usd)，审计结果为 4 个 capsule、0 个 cylinder，半径和高度均为 `0.03 m`，且没有绝对外部引用。只有显式设置 `ACTUATEX_TINYMAL_URDF` 才切到 Sim 6 原生 cylinder，作为独立 A/B 组。

## 5. 运行时分层验收

| 验收项 | 结果 |
|---|---|
| 官方 ZIP MD5 | PASS |
| RTX 4070 Ti SUPER + 驱动 `580.159.03` headless Vulkan 启动 | PASS |
| Warp/CUDA，SM 8.9 | PASS |
| PyTorch CUDA matmul | PASS |
| TinyMal `4 env × 1 iteration` 冒烟训练 | PASS |
| 48 维 actor 观测、69 维 critic 输入、12 维动作 | PASS |
| 经过审计的 capsule USD | PASS |
| 重复创建/关闭同种子评估环境 | PASS |
| RTX LiDAR → ROS `LaserScan` writer rollout | PASS |
| RGB + `CameraInfo` + 可选理想 depth writer rollout | PASS |
| 导航/相机/MID360 纯 Python 测试 | `26 passed` |
| H.264 楼梯录像 | PASS |

ROS 传感器验收的准确表述是：writer 已正确附着，仿真实际推进并没有触发终止。它不是对外部 ROS 机器的完整订阅回环测试，也不是传感器与实物误差的标定结果。

## 6. 显存和吞吐：为什么选择 8192 环境

PhysX 的 found/lost pair 需求随并行环境数量近似二次增长。实测 4096 环境所需容量超过 3700 万，8192 环境超过 1.51 亿。训练脚本现在按环境数自动把 `gpu_found_lost_pairs_capacity` 向上取到安全的 2 次幂，也允许手工覆盖。

| 环境数 | pair 容量 | 峰值显存 | 稳态吞吐 | 结论 |
|---:|---:|---:|---:|---|
| 4096 | `2^26` | 6860 MiB | 11.5–12.0 万 steps/s | 稳定，但 GPU 未充分利用 |
| 8192 | `2^28` | 15092 MiB（92.2%） | 15.3–15.7 万 steps/s | 稳定、无容量告警，正式训练采用 |

8192 环境压测中，活动期 GPU 利用率均值约 `86.1%`、峰值 `100%`，功耗峰值约 101 W，温度峰值 58°C。这里追求的是最高稳定吞吐和约 8% 的瞬时显存余量，不是把显存读数长期顶到 100%。RTX 相机/雷达训练需要单独定容量，不能照搬纯物理 headless 的 8192。

机器可读压测记录见 [`capacity_benchmark.json`](../artifacts/isaac_sim_6/capacity_benchmark.json)。

## 7. Sim 6 原生微调

### 7.1 共同训练设计

- 算法：PPO，5 epochs，4 mini-batches，clip `0.2`，`γ=0.99`，`λ=0.95`；
- 网络：actor `48-512-256-128-12` ELU；critic `69-512-256-128-1`；
- 每个环境每轮 24 步；8192 个并行环境；
- mirror loss 系数 `0.05`；
- 随机化：摩擦、基座质量/质心、PD 增益、20–60 ms 执行器延迟、观测噪声、4–7 秒随机推扰；
- 低学习率固定调度，避免强策略在迁移后被大步更新破坏；
- 选择 checkpoint 时，原生压力测试优先于训练 reward。

### 7.2 通用鲁棒策略

| 参数 | 值 |
|---|---|
| 热启动 | Sim 5.1 general robust `model375` actor + 兼容 critic |
| seed | 29 |
| 学习率 | `1e-5` fixed |
| entropy | `2e-4` |
| 初始动作标准差 | `0.05` |
| 训练长度 | 60 iterations，11,796,480 samples |
| 训练本体耗时 | 113.16 s |
| 选中 checkpoint | `model50` |

同一种子重新播随机流后的结果：

| 评估 | 热启动 model375 | Sim 6 model50 | 相对变化 | 重置 |
|---|---:|---:|---:|---:|
| seed 17，256 env，六段命令 | `0.09455` RMSE | `0.09371` RMSE | 降低 `0.89%` | 均为 1 |
| seed 71，256 env，六段命令 | `0.09632` RMSE | `0.09518` RMSE | 降低 `1.18%` | 均为 1 |
| 两种子均值 | `0.09544` RMSE | `0.09445` RMSE | 降低 `1.04%` | 无恶化 |

这是小幅但在留出种子上同方向的提升，不写成夸张的 3%–5%。完整证据：

- [`seed17 checkpoint sweep`](../artifacts/isaac_sim_6/evaluation/sim6_finetune_sweep_seed17_256env_reseeded.json)
- [`seed71 holdout`](../artifacts/isaac_sim_6/evaluation/sim6_model50_holdout_seed71_256env_reseeded.json)

### 7.3 台阶鲁棒策略

| 参数 | 值 |
|---|---|
| 热启动 | Sim 5.1 stairs robust `model399` actor + 兼容 critic |
| seed | 47 |
| 学习率 | `5e-6` fixed |
| entropy | `1e-4` |
| 初始动作标准差 | `0.04` |
| 实际训练 | 40 iterations，7,864,320 samples，58.23 s |
| 候选 | `model0/10/20/30/39` |
| 选中 checkpoint | `model10` |

预先固定的排序规则是：无重置登顶数 → 首次登顶数 → 总登顶数 → 重置数 → 登顶时间。两个随机种子合并结果如下：

| 策略 | 无重置登顶 | 首次登顶 | 总登顶 | 重置 | 平均登顶时间 |
|---|---:|---:|---:|---:|---:|
| 热启动 model399 | `206/256`（80.47%） | `236/256` | `238/256` | 48 | 4.021 s |
| Sim 6 model10 | **`212/256`（82.81%）** | `236/256` | `237/256` | **43** | **3.966 s** |

新模型少 1 次“包括重试后的总成功”，但多 6 次干净成功、少 5 次重置，且首次成功数相同，因此按规则选择 `model10`。无随机展示评估为首次登顶 `64/64`，其中 `60/64` 在完整 8 秒 rollout 内没有后续重置；最终录像所用单环境 5 秒 rollout 为登顶成功、0 重置。

证据：

- [`seed43 sweep`](../artifacts/isaac_sim_6/evaluation/sim6_stairs_finetune_sweep_seed43_128env_reseeded.json)
- [`seed31 holdout`](../artifacts/isaac_sim_6/evaluation/sim6_stairs_old399_vs_refine10_holdout_seed31_128env_reseeded.json)
- [`64-env nominal`](../artifacts/isaac_sim_6/evaluation/sim6_stairs_refine10_nominal_seed5_64env.json)
- [`recorded rollout`](../artifacts/isaac_sim_6/evaluation/sim6_stairs_refine10_video_seed5.json)

### 7.4 为什么发布双模型

把通用 `model50` 直接放进同一台阶压力测试，只得到 `6/128 = 4.69%` 登顶；台阶专家基线为 `119/128 = 92.97%`。任务分布差异远大于通用策略微调带来的 1% 级收益。当前最稳妥的工程方案是“通用运动策略 + 台阶专家”，以后再研究地形感知 gating、蒸馏或 mixture-of-experts，而不是现在用一个退化的折中模型覆盖两者。

## 8. 双向 sim2sim：通过不等于迁移成功

### 8.1 其他引擎策略进入 Isaac Sim 6

| 来源策略 | 目标 | 六段命令总重置 | 结论 |
|---|---|---:|---|
| Isaac Gym `model12` | Sim 6 / PhysX 5 | 43 | 能运行，但接触/执行器差异明显 |
| MuJoCo fine-tuned `model10` | Sim 6 / PhysX 5 | 31 | 比 Gym 来源少，但仍未达到原生策略标准 |

证据分别为 [`isaac_gym_model12_to_isaac_sim6_suite.json`](../artifacts/isaac_sim_6/sim2sim/isaac_gym_model12_to_isaac_sim6_suite.json) 和 [`mujoco_model10_to_isaac_sim6_suite.json`](../artifacts/isaac_sim_6/sim2sim/mujoco_model10_to_isaac_sim6_suite.json)。

### 8.2 Isaac Sim 6 策略进入 MuJoCo

通用 `model50` 的 30 秒反向测试结果：

- 站立、前进和后退名义上存活 30 秒；
- 两个前进段的 `vx RMSE` 约为命令本身，说明机器人几乎没有按命令前进；
- 横移在 17.12 秒跌倒；
- 转向在 8.02 秒跌倒。

因此“30 秒没倒”不能单独作为 sim2sim 成功指标，必须同时检查命令跟踪。Sim 6 微调改善了原生 PhysX 指标，却没有解决 MuJoCo 的执行器、接触和坐标语义差异。当前跨引擎最可靠的仍是 MuJoCo 专用 fine-tuned `model10`：历史正式验收为六段 30 秒 `0/6` 跌倒、主轴 RMSE `0.07957`、动力学组合 `16/16` 不跌倒，并通过 MuJoCo 台阶测试。

这也是保留不同后端原生模型的理由。后续应先做接触参数、armature、摩擦、PD/延迟和命令缩放的系统辨识，再考虑共享策略或在线适应。

## 9. ROS 2、Nav2、RTX LiDAR 与相机

导航入口 [`run_tinymal_nav2.py`](../backends/isaac_lab/scripts/run_tinymal_nav2.py) 已完成以下迁移：

- 选择 Sim 随附且与 Python 3.12 兼容的 Humble/Jazzy ROS 库，剥离 `/opt/ros` 的 Python 3.10 混装路径；
- 提前启用 `isaacsim.sensors.experimental.rtx`、`isaacsim.ros2.bridge`、`isaacsim.ros2.nodes` 和镜头畸变 schema；
- 使用官方 SyntheticData RenderVar API 解析 RGB、CameraInfo 和 depth writer；
- 让全局相机正确跟随 instanceable 机器人并组合 frame transform；
- MID360 PointCloud2/`LaserScan` 与相机 writers 均通过实际仿真步进。

相机入口要求 ROS `camera_info` YAML，并校验分辨率、K/P/D、畸变模型和 skew，再映射到 OpenCV lens schema。示例 [`front_camera_info.example.yaml`](../navigation/ros2/actuatex_navigation/config/front_camera_info.example.yaml) 只用于 API 测试，不能代表实机。

要接近真实相机，还需测量并建模：真实内外参、滚动/全局快门、曝光与响应曲线、镜头畸变、景深/光圈、噪声、动态模糊、帧延迟/抖动、深度孔洞与多径、安装振动和时间同步。

MID360 已从通用旋转 LiDAR 替换为官方逐点模型：固定使用 Livox 发布的
800,000 行、4 秒非重复扫描表，切成 40 个 10 Hz 状态，每帧保留 20,000
个角度、`fireTimeNs` 和 4 条硬件 line。Sim 6 RTX Hydra 的单传感器属性
上限为 5 MiB，完整表约 15.287 MiB，因此实现把每帧交错拆为 4 个 5,000
点 RTX prim，并在同一 writer tick 中按绝对时间合并排序；这不是降采样。
`channelId` 使用 RTX 内部 detector 语义，Livox line 保存在 `bank` 并从
`emitterId` 恢复，避免把四条物理 line 错当成四个 RTX detector 后无回波。

独立六面标定场验证了 40/40 emitter state、20,000/20,000 发射点属性覆盖、
4/4 同步分片、四条 line 和有序逐点时间；带完整外观壳体的机器狗联调在 5
秒内得到 49 个物理帧、250 个控制步零终止且无 Vulkan 重置。ROS 端按
`livox_ros_driver2` 的 26-byte `PointXYZRTLT` 发布；安装官方消息包时还可发
`CustomMsg`。有效回波数取决于场景几何和材质，不能拿它冒充发射点总数。

完整密度验收命令：

```bash
/path/to/isaac-sim-6.0.1/python.sh \
  backends/isaac_lab/scripts/validate_mid360_rtx.py \
  --stride 1 --frames 180 \
  --out artifacts/isaac_sim_6/evaluation/mid360_exact_profile_validation.json
```

尚未完成的是实机统计校准：强度响应、距离/角噪声、材质相关丢点、运动畸变
参数、时间同步抖动和 Livox UDP 包级驱动。Isaac Sim 是主环境；Gazebo 插件
路线应作为同输入交叉验证，优先保证迁移准确性，而不是为了 ROS 2 接入方便
而降低模型要求。

## 10. 原创串联式轮腿机器人

在四足链路之外，本轮新增了一台左右各 `hip → knee → wheel` 的 6-DOF 开放串联轮腿机器人。URDF 使用基础几何体原创构建，没有复制调研项目的 CAD、mesh 或源码；Wheel-Legged-Gym、Flamingo 和 robot_lab 仅用于验证路线与当前 Isaac Lab 组织方式。

任务使用 28 维观测、4 个腿位置动作和 2 个轮速度动作。第一阶段在 8192 环境训练 500 轮，第二阶段加入每环境独立的 0–20 ms actuator delay 与更宽摩擦、质量、质心、增益和推力随机化，再训练 200 轮。两阶段共采样 137,625,600 条 transition，正式训练约 9.0 分钟。

独立 1024 环境未见强扰动 A/B 中，第一阶段 `model_499` 为 23 falls、RMSE 0.11662，最终 `model_199` 为 7 falls、RMSE 0.09817，跌倒减少 69.6%。带延迟的 clean 测试中两者均 0 跌倒，最终模型 RMSE 改善 37.1%。因此最终选择来自大样本 holdout，而不是训练 reward 或单段视频。

完整开源调研、机械/MDP 设计、训练命令、不利结果和边界见 [`WHEEL_LEGGED_RL.zh-CN.md`](WHEEL_LEGGED_RL.zh-CN.md)。轮腿 MuJoCo sim2sim、VMC 对照与复杂地形课程仍是下一阶段，不在本轮完成范围内。

## 11. 最终产物

### 11.1 可部署策略

每套策略都保留三种形态：

- 完整 RSL-RL checkpoint：可续训；
- 纯 actor `state_dict`：便于审计和自定义加载；
- TorchScript actor：TinyMal 为 48→12，串联式轮腿为 28→6，适合脱离训练器做确定性推理集成。

目录：[`artifacts/isaac_sim_6/checkpoints/`](../artifacts/isaac_sim_6/checkpoints/)。导出过程已比较 eager 与 TorchScript 的零输入输出，要求逐元素相等。

### 11.2 录像

- PPT 成片：[`TinyMal_IsaacSim_6_FinalPolicy_Stairs_PPT.mp4`](../artifacts/isaac_sim_6/videos/TinyMal_IsaacSim_6_FinalPolicy_Stairs_PPT.mp4)
- 原始 5 秒录像：`artifacts/isaac_sim_6/videos/stairs_refine10_raw/`
- 轮腿 PPT 成片：`artifacts/isaac_sim_6/videos/wheel_legged_ppt/ActuateX_SerialWheelLegged_IsaacSim6_PPT-step-0.mp4`

成片在完整登顶后于 3.3 秒结束运动并定格 0.7 秒，避免静态机位下机器人走出画面；没有剪掉失败或重置，因为对应单环境 rollout 本身就是 0 重置。

### 11.3 机器可读证据

- 总清单：[`manifest.json`](../artifacts/isaac_sim_6/manifest.json)
- 显存压测：[`capacity_benchmark.json`](../artifacts/isaac_sim_6/capacity_benchmark.json)
- 公平评估：`artifacts/isaac_sim_6/evaluation/`
- 双向 sim2sim：`artifacts/isaac_sim_6/sim2sim/`

`artifacts/` 默认不进入 Git，发布 checkpoint 或课程包时应作为单独 Release/ZIP，并连同清单一起分发。

## 12. 可复现命令

环境变量：

```bash
export ACTUATEX_ROOT=/path/to/ActuateX
export ISAAC_LAB_ROOT=/path/to/IsaacLab-3
export PRIVACY_CONSENT=false
export OMNI_KIT_ACCEPT_EULA=YES
```

最小训练烟测：

```bash
"$ISAAC_LAB_ROOT/isaaclab.sh" -p \
  "$ACTUATEX_ROOT/backends/isaac_lab/scripts/train_tinymal.py" \
  --task Isaac-Velocity-Native-Forward-TinyMal-v0 \
  --num_envs 4 --max_iterations 1 --seed 1
```

鲁棒训练时可直接使用自动 PhysX 容量：

```bash
"$ISAAC_LAB_ROOT/isaaclab.sh" -p \
  "$ACTUATEX_ROOT/backends/isaac_lab/scripts/train_tinymal.py" \
  --task Isaac-Velocity-Native-Robust-TinyMal-v0 \
  --num_envs 8192 --max_iterations 60 --seed 29 \
  --learning_rate 1e-5 --schedule fixed \
  --entropy_coef 2e-4 --init_noise_std 0.05 \
  --init_checkpoint /path/to/general_model375.pt --init_critic
```

公平评估多个 checkpoint：

```bash
"$ISAAC_LAB_ROOT/isaaclab.sh" -p \
  "$ACTUATEX_ROOT/backends/isaac_lab/scripts/evaluate_tinymal_robustness.py" \
  --ckpt model375.pt model50.pt \
  --num_envs 256 --seed 71 \
  --out artifacts/isaac_sim_6/evaluation/holdout.json
```

## 13. 完成边界与下一步

截至本报告日期，**Isaac Sim 6.0.1 的仿真迁移、训练、复评、传感器接口和展示视频已经完成**。下面这些不应被写成已经完成：

1. 使用真实相机标定与真实图像统计完成 photometric/depth 对齐；
2. 用实机数据对齐 MID360 的强度、噪声、丢点、运动畸变和 UDP 包级时序；
3. 在 Gazebo MID360 插件与 Isaac Sim RTX 模型之间做同轨迹点云误差对照；
4. 缩小 Isaac Sim 6 → MuJoCo 的横移、转向和命令跟踪差距；
5. 加入历史观测、特权教师/学生、在线适应或地形感知专家切换；
6. 完成真机执行器标定、延迟、急停、跌倒保护和受控场地验收。
7. 完成串联式轮腿的 MuJoCo sim2sim、VMC 对照与复杂地形课程。

所以本阶段可以判定为“仿真实验圆满收尾”，但不能判定为“真实机器人部署完成”。

## 14. 官方资料

- [Isaac Sim 6.0.1 Release Notes](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/release_notes.html)
- [Isaac Sim 6.0.1 Download](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/download.html)
- [Isaac Sim 6.0 Migration Guide](https://docs.isaacsim.omniverse.nvidia.com/latest/migration_guides/isaac_sim_6_0/index.html)
- [Isaac Sim v6.0.1 Release](https://github.com/isaac-sim/IsaacSim/releases/tag/v6.0.1)
- [Isaac Lab v3.0.0-beta2.patch1 Release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2.patch1)

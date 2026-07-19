# ActuateX Sentinel 阶段一：社区机械基线与双后端全动力学

> 状态：M0 动力学与训练管线已闭合，正式长时训练尚未开始
>
> 后端：Isaac Sim 6.0.1 GA + Isaac Lab 3.0.0-beta2.patch1 / MuJoCo 3.10
>
> 更新：2026-07-20

## 1. 这一阶段真正完成了什么

ActuateX 现在有一台原创的 11-DOF `ActuateX Sentinel` 轮腿机器人，并且不是只放了一份能显示的模型：

- MuJoCo MJCF 与 Isaac Sim/Isaac Lab URDF 使用相同质量、惯量、关节限制、armature 和执行器顺序；
- 6 个底盘动作控制左右髋、膝和轮，另外 5 个关节表示 yaw/pitch 云台、双摩擦轮和拨弹机构；
- 两个后端共享 28 维观测、6 维动作、500 Hz 物理频率和 50 Hz 策略频率；
- 自研显式执行器同时实现电压、电流、反电动势、转矩—转速包络、传动效率、再生功率、温升与热降额；
- 2026 规则中的 100 W 底盘限制、60 J 缓冲、10 Hz 裁判检测和耗尽后 5 s 断电被固定到可审计配置；
- PPO、SAC、TD3 在 MuJoCo 和 Isaac Lab 都已有训练入口；
- SAC/TD3 使用同一 replay-sample-ratio 调度器，不再让并行环境数隐式改变数据复用强度；
- 同一策略加载器支持旧 RSL-RL PPO、新 RSL-RL PPO、SAC 和 TD3，可做双向 sim2sim；
- 发射状态机、热量、弹量、速度门控、后坐冲量和带二次阻力的 RK4 弹道已有后端无关实现。

这里的“完成”是指模型、算法入口和验证闭环已经形成，不表示已经得到比赛级策略。当前保存的本地 smoke checkpoint 只训练了极少步，只用于证明前向、反向传播和跨引擎推理没有断线，不能用于比较 PPO、SAC、TD3 的最终优劣。

## 2. RoboMaster 社区的机械设计给了我们什么启发

本仓库没有复制或再分发任何战队 CAD。很多社区资料明确限制商用或再分发，因此我们只把公开文章当作需求和失效模式调研，仓库资产仍由基础几何原创构建。

| 社区一手资料 | 机械信息 | 对 ActuateX 的影响 |
|---|---|---|
| [武汉工程大学 RM2024 轮腿平衡步兵](https://bbs.robomaster.com/article/9657) | 以小型化、低高度活动范围和地形通过为目标；公开描述采用 Go1 轮毂电机、200 mm 轮、18.4 kg 检录质量、约 30 cm 跳跃，并使用“抬升—撞阶—检测倾角—快速缩腿”的高容错登阶动作 | 我们当前 17.015 kg 质量在同一量级，但轮径为 150 mm；下一阶段应把 150/200 mm 做成可切换变体，并把撞阶缩腿作为独立课程，不只训练腾空跳跃 |
| [香港科技大学 ENTERPRIZE RM2024 轮腿结构](https://bbs.robomaster.com/article/20424) | 强调双自由度轮腿的弹跳、台阶能力和低矮空间布局，并公开技术报告、加工图和整车 CAD | 证明“腿—云台—供弹—高度包络”必须联合设计；但其许可不自动兼容 MIT，因此不导入模型 |
| [上海交通大学交龙 RM2025 串联腿](https://bbs.robomaster.com/article/803691) | 技术资料覆盖需求、构型选择、连杆参数和迭代；实机展示包含飞坡、倒飞坡、二级台阶和翻倒起身 | 支持本项目先做串联腿：树状拓扑适合 PhysX articulation，也便于训练跳跃和自救；后续需加入二级台阶与翻倒起身硬验收 |
| [同济大学 SuperPower RM2025 开链四连杆](https://bbs.robomaster.com/article/769697) | 作者如实记录该构型未达到预期，原因包括机械试错过久以及构型需要更强的控制算法 | 不把“结构更新颖”误当“效果必然更好”；每个构型都必须经过同一地形、能耗、跌倒和维护指标验证 |
| [东莞理工学院关节轴系复盘](https://bbs.robomaster.com/article/54960) | “塞打螺丝 + 法兰轴承”方案在强跳跃下可能松动，过紧又会卡滞，轮组悬臂较大时还可能外八；自研碗组提高轴向强度和拆装性，但机加工成本更高 | 域随机化不能只随机质量和摩擦，还要加入关节间隙、轴向偏载、左右轮外倾/前束和摩擦漂移；这些是当前模型尚缺的机械误差层 |
| [香港大学 HerKules RM2026 质心偏移建模](https://bbs.robomaster.com/article/1448924) | 在既有轮腿模型上显式考虑腿部与机体质心偏移 | 当前对称惯量只能作为名义模型；需要按电池、云台、弹仓和线束位置建立非对称质心分布与 holdout |
| [中国石油大学（北京）SPR RM2026 控制开源](https://bbs.robomaster.com/article/1884251) | 公布 2.2 m/s、280 mm 跳跃、蹭台阶和自救等实机功能，并说明使用二阶倒立摆、滤波和高速转向补偿 | 为下一阶段验收提供功能量级，但不是可直接照抄的参数；ActuateX 仍需用自身尺寸、功率和电机数据重新辨识 |

社区结论不是“大家都用同一套五连杆”。真实路线包含闭链五连杆、开链四连杆、串联腿、球轮和共轴麦轮。ActuateX 第一版选择串联 `hip-knee-wheel`，是为了先把全动力学、学习和 sim2sim 做扎实；后续可把闭链五连杆作为对照资产，而不是未经验证地替换主模型。

## 3. 当前机械模型到底是什么

| 项目 | 当前值 | 性质 |
|---|---:|---|
| 总质量 | 17.015 kg | MuJoCo/URDF 严格一致；原创名义值 |
| 底盘尺寸 | 0.48 × 0.38 × 0.16 m | 原创基础几何，不对应某支战队 CAD |
| 左右髋间距 | 0.44 m | 两后端共享 |
| 大腿/小腿长度 | 0.15 / 0.25 m | 串联双关节 |
| 轮径 | 0.15 m | 第一版；计划增加 0.20 m 社区量级变体 |
| 底盘自由度 | 6 | 左右髋、膝、轮 |
| 整机可控自由度 | 11 | 再加云台 yaw/pitch、双摩擦轮、拨弹盘 |
| 物理/策略频率 | 500 / 50 Hz | 每个动作执行 10 个物理子步 |
| MuJoCo 传感器 | 44 | 基座、关节、枪口、MID360 位姿和四面装甲接触 |

质量和惯量虽然已经跨后端一致，但仍是从基础几何计算的名义量，不是从某台实机 CAD 或摆线实验辨识所得。它适合算法开发和接口验证，不应称为某个战队机器人的数字孪生。

## 4. Isaac Sim 能仿真到什么程度

简短结论：**现在已经达到控制算法级的全刚体动力学，可以向参数辨识后的控制级数字孪生推进；还不是结构强度、轴承寿命或轮胎材料级数字孪生。**

NVIDIA 的 PhysX articulation 采用约化坐标，树状关节机构具有较小关节误差，并能处理较大的质量比；它还支持关节 armature、库仑/黏性摩擦、mimic joint、固定/空间 tendon 和 GPU 张量批量状态交换。官方也明确说明：闭环 articulation 需要断开一个关节并用普通约束闭环，求解更困难，可能需要更小步长。因此当前串联腿能够直接高效训练，而闭链五连杆必须单独做稳定性与 sim2sim 验收，不能只靠 URDF 导入成功来判断。[PhysX articulation 说明](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/articulations.html)

| 层级 | Isaac Sim 6 能力 | ActuateX 当前状态 | 可信边界 |
|---|---|---|---|
| 刚体与关节 | 质量、惯量、关节限制、armature、碰撞、自碰撞、摩擦和恢复系数 | 已实现并与 MuJoCo 对齐 | 参数准确时可用于平衡、跳跃、撞阶和翻倒；不能替代 FEA |
| 执行器 | Isaac Lab 支持显式物理或神经网络执行器；PhysX 也有静态转矩—转速包络 | 已实现 24 V DC 电机、电流、反电动势、传动、热降额和命令延迟 | 当前电机常数为 ActuateX 名义值；真机前必须测功机辨识。官方静态包络本身也不覆盖温度和高频动态，[Actuator API](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.actuators.html) |
| 功率与裁判 | 可在 GPU 控制回路中维护批量状态 | 已实现 100 W、60 J、10 Hz 与 5 s 断电逻辑 | 是规则/控制仿真，不是 ESC 开关、电池化学或超级电容电路仿真 |
| 轮地接触 | 刚体接触、库仑摩擦、恢复和接触力 | 可训练平地、斜坡、台阶、碰撞与打滑随机化 | 不自动包含轮胎形变、胎纹、橡胶迟滞、灰尘和地毯纤维；需实验拟合或自定义模型 |
| 机械误差 | 可用关节摩擦、偏置、柔顺约束和随机化近似 | 尚未加入间隙、外八、轴承预紧、螺钉松动和线束拖曳 | 能做统计近似，不能从渲染模型自动得到制造公差或疲劳寿命 |
| 闭链五连杆 | 可用断环普通 joint、mimic 或 tendon 表达 | 当前未实现，主模型为串联腿 | 可以研究，但比树状串联腿更敏感；必须降低步长并和 MuJoCo/实机对照 |
| IMU/力/接触 | 官方支持关节力、effort、IMU、接触和 joint-state 传感器 | 关节状态和接触已进入训练；完整传感发布待接 | 物理真值需再叠加噪声、偏置、采样、时间戳和传输延迟，[传感器列表](https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/index.html) |
| RTX 相机与 LiDAR | 可做材质、光照、相机和 RTX ray tracing | 机器人已有相机/MID360 安装位；完整 Sentinel 感知联调待完成 | 能生成高质量合成数据，但域差仍需真实数据标定 |
| 发射与命中 | 可做弹丸刚体、碰撞、接触、渲染和后坐力 | 状态机、解析弹道和后坐冲量已完成；Isaac L1 刚体弹丸尚未接入任务 | 可以做到比赛场景级事件仿真，不模拟摩擦轮橡胶微观接触和弹丸材料破坏 |

因此，应把“能不能仿真”拆成三档：

1. **已经可靠**：树状刚体轮腿、云台、关节/轮接触、显式电机、功率热状态、并行 RL 和双向策略回放；
2. **可以做到但要数据**：实机质量/惯量、转矩曲线、摩擦、轮胎滑移、延迟、质心漂移和传感器噪声；
3. **不应交给 Isaac Sim 代替**：结构应力、螺栓疲劳、轴承寿命、齿面接触、材料破坏与电池电化学。这些应由 CAD/FEA、电机台架和实物试验提供参数，再回填仿真。

## 5. 双后端实现

```text
tasks/robomaster/
  contract.py          关节、时序、电机和 2026 规则唯一真源
  locomotion.py        28-D 观测、6-D 动作与 PD 目标
  powertrain.py        NumPy 电机/功率/热模型
  torch_powertrain.py  与 NumPy 数值对齐的 GPU 批量实现
  launcher.py          安全状态机、热量、后坐与弹道
  policy.py            PPO/SAC/TD3 统一推理加载器

robots/robomaster/
  urdf/actuatex_sentinel.urdf
  mjcf/actuatex_sentinel.xml

backends/isaac_lab/
  tinymal_lab/sentinel_*.py
  scripts/train_sentinel_off_policy.py
  scripts/evaluate_sentinel.py

backends/mujoco/
  sentinel_env.py
  train_sentinel_ppo.py
  train_sentinel_off_policy.py
  evaluate_sentinel.py
```

两套后端不假设 URDF 声明顺序等于策略顺序，而是始终按关节名映射。Isaac Sim 实际 articulation 顺序与 URDF/策略顺序不同，运行时已经验证映射为正确的髋、膝和轮动作 ID。

## 6. 如何复现阶段检查

### 6.1 纯 Python / MuJoCo 回归

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
pytest -q backends/mujoco/tests/test_robomaster_dynamics.py

python backends/mujoco/train_sentinel_ppo.py \
  --num-envs 8 --num-threads 2 --max-iterations 1

python backends/mujoco/train_sentinel_off_policy.py \
  --algorithm sac --num-envs 8 --num-threads 2 \
  --total-transitions 1024 --learning-starts 128 --batch-size 64 \
  --replay-sample-ratio 4
```

### 6.2 Isaac Sim 6 / Isaac Lab

```bash
/path/to/isaac-sim-6.0.1/python.sh \
  backends/isaac_lab/scripts/smoke_sentinel.py \
  --num_envs 4 --steps 250 --device cuda:0

/path/to/isaac-sim-6.0.1/python.sh \
  backends/isaac_lab/scripts/train_tinymal.py \
  --task Isaac-Velocity-Flat-Sentinel-v0 \
  --num_envs 4 --max_iterations 1 --device cuda:0 \
  --output_checkpoint artifacts/checkpoints/sentinel/isaac_lab_ppo.pt

/path/to/isaac-sim-6.0.1/python.sh \
  backends/isaac_lab/scripts/train_sentinel_off_policy.py \
  --algorithm td3 --num_envs 16 --total_transitions 1024 \
  --learning_starts 128 --batch_size 64 --replay_sample_ratio 4 \
  --device cuda:0
```

### 6.3 双向 sim2sim

```bash
python backends/mujoco/evaluate_sentinel.py \
  --checkpoint /path/to/isaac_checkpoint.pt \
  --num-envs 64 --num-threads 16

/path/to/isaac-sim-6.0.1/python.sh \
  backends/isaac_lab/scripts/evaluate_sentinel.py \
  --checkpoint /path/to/mujoco_checkpoint.pt \
  --num_envs 256 --mode clean --device cuda:0
```

评估器固定六段命令，统计速度/yaw RMSE、直立误差、跌倒、最低机身高度、峰值功率、最低缓冲能量、最高电机温度和裁判断电样本。`clean` 用确定性复位；`train_randomization` 和 `holdout` 使用不同的动力学范围与 0–20 ms 延迟。

### 6.4 SAC/TD3 的公平更新预算

离策略算法不能直接用“每次向量环境调用更新几次”做跨后端预算，因为一次 Isaac 调用可能产生数千条 transition，而一次 MuJoCo 调用只产生几十条。ActuateX 统一记录：

```text
replay sample ratio = gradient updates × batch size / 新进入训练阶段的 transitions
```

默认值为 `4.0`。调度器保留小数更新额度，因此即使 MuJoCo 单次采集量小于 `batch_size / ratio`，也会跨环境步精确累计，不会因为整数取整而少训练。metrics 同时保存目标值、实际值、batch size、有效训练 transitions 和每条 transition 的梯度更新次数。

旧参数 `--updates-per-step`（MuJoCo）/`--updates_per_step`（Isaac Lab）仍可显式覆盖，便于复现实验；但它会随并行环境数改变实际 replay ratio，脚本会打印警告，不用于正式 PPO/SAC/TD3 对比。

## 7. 已通过的证据与不能越界的结论

| 检查 | 本机结果 | 能证明什么 |
|---|---:|---|
| 单元/数值回归 | 19 passed | 官方常量、资产结构、电机限制、NumPy/Torch 一致性、裁判状态、发射与环境有限值成立 |
| MuJoCo 资产编译 | `nq=18, nv=17, nu=11, sensors=44` | MJCF 完整且执行器顺序未漂移 |
| Isaac 刚体 smoke | 4 env × 250 steps | Sim 6 能加载并推进自研显式执行器，无非有限状态 |
| Isaac PPO smoke | 4 env × 1 iteration | RSL-RL 前向、采样和反向更新闭合 |
| Isaac SAC / TD3 ratio smoke | 各 1024 env、131072 transitions、384 GPU updates、batch 1024、ratio 4.000 | PhysX 采样、GPU replay 与两种 off-policy 反向更新闭合，预算一致 |
| MuJoCo PPO smoke | 8 env × 1 iteration | MuJoCo PPO 与 GPU 网络更新闭合 |
| MuJoCo SAC / TD3 ratio smoke | 各 32 env、4096 transitions、56 GPU updates、batch 256、ratio 4.000 | 小并行的小数更新额度与共享 off-policy 实现闭合，预算和 Isaac 定义一致 |
| 双向推理 | MuJoCo SAC → Isaac、Isaac SAC → MuJoCo 均通过 | 观测、动作、网络与执行器接口可迁移 |

极短 smoke 运行时间不足以评价平衡和跟踪能力，因此其零跌倒或 RMSE 不进入正式成绩。下一阶段必须使用独立种子完成长时训练、完整 22 s 六段命令、复杂地形、持续扰动和跨引擎测试后，才发布“哪种算法更好”的结论。

## 8. 下一阶段的机械与训练优先级

1. PPO 使用已测得的 8192 环境甜点；SAC/TD3 独立扫描并行数与 replay ratio，按相同 transition 和 replay-sample 预算启动正式长训；
2. 增加 150/200 mm 轮径和不同连杆参数变体，用统一功率、台阶、跌倒和能耗指标筛选；
3. 加入社区暴露出的关节间隙、摩擦漂移、轮外倾/前束、质心偏移、左右不对称和撞击后参数漂移；
4. 建立“抬升撞阶—倾角检测—缩腿”、跳一级/二级台阶、飞坡/倒飞坡和翻倒起身课程；
5. 完成刚体弹丸、装甲命中、云台反作用和射击后坐对平衡的耦合测试；
6. 只在获得明确授权和许可兼容时接入外部 CAD；否则提供用户本地转换器、尺寸表和哈希，不把受限模型放进 MIT 仓库；
7. 真机前建立称重/悬挂摆惯量、电机测功、轮地滑移、延迟和温升台架，形成参数辨识数据集。

阶段一的价值不是“外观已经像比赛机器人”，而是已经有一条能持续提高真实性的主干：社区需求定义机械问题，实测数据校准共享契约，Isaac Sim 批量训练与复杂感知，MuJoCo 独立复核，最后用 sim2sim 和实机 holdout 拒绝只在单一模拟器里成立的策略。

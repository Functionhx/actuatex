# 串联轮腿 Isaac Sim 6 → MuJoCo sim2sim 实验报告

> 实测日期：2026-07-19
> 源后端：Isaac Sim 6.0.1 GA、Isaac Lab 3.0.0-beta2.patch1、PhysX 5
> 目标后端：MuJoCo 3.10.0
> 策略：Isaac Sim 6 鲁棒阶段 `model_199.pt` 导出的 TorchScript

## 1. 结论

ActuateX 已完成串联式两轮腿策略从 Isaac Sim 6 到 MuJoCo 的可复现迁移。结果不是“导入资产后看起来能动”，而是对齐了模型、观测、动作、执行器和控制时序，再按同一组命令量化：

- 原始零调参迁移在 22 s 连续命令中记录到 **23 次跌倒重置**，主轴 RMSE 为 **0.11125**；主要失败发生在 `+1.0 → -0.5 m/s` 突然换向；
- 不修改神经网络权重，只在策略外加入真实系统同样需要的速度命令加速度限制后，跌倒变为 **0**，主轴 RMSE 降至 **0.05549**；
- 源 PhysX 参考在 256 个环境中为 **0 falls、RMSE 0.04468**；MuJoCo 稳定迁移的 RMSE 高约 **24.2%**；
- MuJoCo 中 `8 m/s²` 的线速度指令斜率仍能完成测试，`9 m/s²` 出现 22 次跌倒，说明连续换向边界位于二者之间；最终选择 `6 m/s²`，保留约 25% 的斜率裕量；
- 基础质量 `0.85×–1.15×`、5 ms 延迟以及大部分摩擦扫描保持 0 跌倒；10 ms 延迟开始失稳；摩擦系数 `0.60` 出现显著非单调失稳，说明轮地接触仍需标定。

稳定迁移不等于两个模拟器等价。它证明当前策略可以在明确的命令整形器和已记录的接触设定下跨引擎工作，也明确暴露了延迟与接触模型边界。

## 2. 五层对齐

### 2.1 机械模型

MuJoCo 使用显式 MJCF，而不是直接把 URDF 丢给转换器后接受隐式默认值：

- 7 个机器人刚体，总质量 `12.28 kg`；
- 四个腿关节与两个轮关节的轴、范围、质量和主惯量逐项匹配 URDF；
- 策略动作顺序固定为 `left_hip, left_knee, right_hip, right_knee, left_wheel, right_wheel`；
- `nq=13`、`nv=12`、`nu=6`；
- 物理步长 `0.005 s`，MuJoCo `implicitfast`，接触使用 6 维椭圆摩擦锥。

MJCF SHA-256：`79018739db102c0aff0226afdbba7305fd9c371f6dd6de023b9ed1a41a5bdd50`。

### 2.2 策略接口

TorchScript actor 的结构为 `28 → 512 → 256 → 128 → 6`，文件 SHA-256 为：

```text
0b7003dc8f4ac0aa445d862f447d82a86706ba3ec8592e291b06fc3269743338
```

28 维观测严格保持源任务语义：机体系线速度、角速度、重力投影、原始速度命令、四个腿关节相对位置、六个关节速度和上一动作。轮角度仍被排除，避免无界旋转位置进入网络。

### 2.3 执行器

每个 `5 ms` physics tick 重新计算一次与 Isaac Lab 相同的混合 PD：

- 腿：位置目标，`kp=40`、`kd=1`、`±30 Nm`；
- 轮：速度目标，`kd=0.5`、`±12 Nm`；
- 腿 action scale `0.45 rad`，轮 action scale `20 rad/s`；
- armature 为腿 `0.01`、轮 `0.02`。

目标动作每 4 个 physics tick 更新一次，因此策略频率仍为 50 Hz。可选延迟按 5 ms 整数 tick 进入 FIFO，不把“延迟”错误实现成只延迟网络调用。

### 2.4 坐标系

MuJoCo 的 `wxyz` 根四元数显式转为 body-to-world 旋转矩阵，机体系速度和重力投影由同一变换计算。单元测试固定观测维度、重力方向、关节顺序、动作缩放和力矩裁剪，防止代码重构后发生静默漂移。

### 2.5 测试协议

数值测试连续执行以下 22 s 序列：

```text
stand 2 s → forward 0.5 m/s 4 s → forward 1.0 m/s 4 s
→ backward -0.5 m/s 4 s → yaw 0.8 rad/s 4 s
→ arc (vx 0.7 m/s, yaw 0.6 rad/s) 4 s
```

跌倒后重置并继续当前段，所以“falls”是跌倒重置次数，不是六个命令中失败段的数量。RMSE 去掉每段开头最多 1 s 的过渡区，再按被命令主轴取平均。

源参考是 256 个 PhysX 环境的统计，目标是一个确定性 MuJoCo 连续 rollout；两者可用于同协议趋势比较，但不能把跌倒计数当成相同样本量的统计检验。

## 3. 原始迁移与部署包装器

| 设置 | 网络权重 | 线/角命令斜率 | 跌倒重置 | 主轴 RMSE |
|---|---|---:|---:|---:|
| Isaac Sim 6 源参考，256 env | `model_199` | 原始测试命令 | 0 | 0.04468 |
| MuJoCo 原始零调参 | 同一 TorchScript | 无限制 | 23 | 0.11125 |
| MuJoCo 部署设置 | 同一 TorchScript | `6 / 4` 每秒² | **0** | **0.05549** |

命令斜率限制不是重新训练、模型融合或隐藏的物理调参。它是位于操作者/导航器和低层策略之间的部署包装器：限制参考速度在一个控制周期内的变化量。真实底盘同样不会把速度参考视为无穷大加速度的理想阶跃。

这项处理也揭示了训练改进方向：源任务应对命令跃迁率做随机化或直接观察目标加速度，而不应只在部署端依赖固定滤波。

## 4. 鲁棒性与失败边界

所有下表目标测试都使用 `6 m/s²` 线速度和 `4 rad/s²` yaw 命令斜率。

| 变量 | 设置 | 跌倒重置 | 主轴 RMSE | 判定 |
|---|---:|---:|---:|---|
| 命令边界 | `8 m/s²` | 0 | 0.05589 | 通过 |
| 命令边界 | `9 m/s²` | 22 | 0.11142 | 失败 |
| 额外延迟 | 5 ms | 0 | 0.05541 | 通过 |
| 额外延迟 | 10 ms | 3 | 0.08385 | 失败起点 |
| 额外延迟 | 15 ms | 13 | 0.06932 | 失败 |
| 额外延迟 | 20 ms | 6 | 0.38552 | 失败 |
| 底盘质量 | `0.85×` | 0 | 0.04569 | 通过 |
| 底盘质量 | `1.15×` | 0 | 0.06979 | 通过 |
| 摩擦 | 0.45 | 0 | 0.06334 | 通过 |
| 摩擦 | 0.60 | 89 | 0.39921 | 失败 |
| 摩擦 | 0.80 | 0 | 0.04017 | 通过 |
| 摩擦 | 1.20 | 0 | 0.07658 | 通过 |
| 摩擦 | 1.40 | 0 | 0.10059 | 通过 |

摩擦结果不是单调的材料规律。两轮动态平衡、圆柱接触、6 维接触约束和离散控制形成混合闭环，局部参数区间可能触发接触模式切换或共振。`0.60` 的失败应被看作必须继续检查的模型敏感区，而不是删掉的“离群值”。后续要用真实轮胎的滑移曲线和电机阶跃响应校准，而不是继续凭一个标量摩擦系数找最好看的结果。

## 5. 复现

先运行控制契约和模型回归测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  pytest -q backends/mujoco/tests/test_wheel_legged_contract.py
```

运行原始迁移：

```bash
python backends/mujoco/wheel_legged_sim2sim.py \
  --out artifacts/mujoco/sim2sim/wheel_legged/raw_nominal.json \
  --tracking artifacts/mujoco/sim2sim/wheel_legged/raw_nominal.csv
```

运行部署设置：

```bash
python backends/mujoco/wheel_legged_sim2sim.py \
  --linear-command-slew 6 --yaw-command-slew 4 \
  --out artifacts/mujoco/sim2sim/wheel_legged/deployed.json \
  --tracking artifacts/mujoco/sim2sim/wheel_legged/deployed.csv
```

一键重跑规范扫描：

```bash
python backends/mujoco/sweep_wheel_legged_sim2sim.py
```

生成演示视频。演示包含更激进的 `+1 → -1 m/s` 连续换向，所以使用 `4 / 3` 的展示斜率，不替代上面的数值协议：

```bash
MUJOCO_GL=egl python backends/mujoco/wheel_legged_sim2sim.py \
  --sequence showcase \
  --linear-command-slew 4 --yaw-command-slew 3 \
  --camera-distance 1.4 --camera-elevation -12 \
  --video artifacts/mujoco/videos/\
ActuateX_SerialWheelLegged_MuJoCo_Sim2Sim.mp4
```

成片规格为 H.264、1280×720、50 FPS、17.5 s、875 帧，实测全程 0 跌倒。

## 6. 代码与证据

| 路径 | 作用 |
|---|---|
| `robots/wheel_legged/mjcf/actuatex_serial_wheel_legged.xml` | 与 URDF 逐项对齐的 MuJoCo twin |
| `backends/mujoco/wheel_legged_contract.py` | 跨框架控制与观测契约 |
| `backends/mujoco/wheel_legged_sim2sim.py` | rollout、扰动、指标、CSV 和视频 |
| `backends/mujoco/sweep_wheel_legged_sim2sim.py` | 规范参数扫描与聚合 JSON |
| `backends/mujoco/tests/test_wheel_legged_contract.py` | 资产、惯量、接口和动力学回归测试 |
| `artifacts/.../canonical/sweep_summary.json` | 本机完整扫描索引，不进入 Git |
| `artifacts/mujoco/videos/ActuateX_SerialWheelLegged_MuJoCo_Sim2Sim.mp4` | 本机 MuJoCo 演示成片，不进入 Git |

## 7. 下一阶段

1. 在源任务中随机化命令跃迁率，并与部署斜率限制做消融；
2. 以实测电机/轮胎数据辨识延迟、扭矩带宽、静摩擦和滑移曲线；
3. 对摩擦 `0.50–0.70` 的不稳定带做求解器、接触维数和控制频率扫描；
4. 在 MuJoCo 原生训练轮腿策略，并做 MuJoCo → Isaac Sim 反向迁移；
5. 加入坡道、台阶和离散障碍，避免把平地稳定误称为完整机动能力。

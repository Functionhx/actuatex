# ActuateX Sentinel：RoboMaster 风格轮腿机器人路线图

> 状态：M0 动力学与双后端训练管线已实现
> 规则快照：RoboMaster 2026 University Championship Rule Manual V1.4.2，2026-04-30
> 原则：仿真优先、接口可审计、赛季参数配置化、真实机构低能量且规则合规

阶段一的社区机械调研、当前 17.015 kg 模型、Isaac Sim 保真边界、验证命令和下一步参数辨识计划见 [Sentinel 阶段一报告](./ROBOMASTER_SENTINEL_PHASE1.zh-CN.md)。报告明确区分“训练/迁移管线已打通”与“高质量策略尚未完成长训”。

## 1. 我们要做什么

ActuateX 不复制某一台现成步兵车，而是把已有能力组合成一台原创的 **ActuateX Sentinel 轮腿哨兵**：

```text
串联轮腿底盘
  + yaw/pitch 云台
  + 全局快门相机 / 深度相机
  + MID360
  + ROS 2 Nav2 / 状态估计
  + 自动瞄准与弹道预测
  + 安全的仿真发射机构 / 装甲命中与裁判状态
  + 多机器人导航、对抗和强化学习
```

它的研究价值不只是“能发射”：轮腿底盘会主动改变相机和发射轴姿态，导航、平衡、瞄准、射击窗口和能量管理彼此耦合，因此可以形成比普通全向轮底盘更有特点的控制问题。

项目不把真实高速发射机构当作开源入门步骤。第一阶段只做低风险仿真；实机阶段必须使用当季官方弹丸、能量/射速限制、硬件急停、独立使能、物理保险和封闭测试区。

## 2. Isaac Sim、Isaac Lab、MuJoCo 和 Gazebo 的分工

| 层 | 主角色 | 为什么 |
|---|---|---|
| **Isaac Sim 6** | 主要数字孪生、RTX 相机/MID360、比赛场景、弹丸刚体、命中事件、视频和合成数据 | 适合复杂传感器、USD 场景、PhysX 碰撞和高质量展示 |
| **Isaac Lab** | 在 Isaac Sim 中批量训练底盘机动、目标跟踪、射击时机、策略课程和域随机化 | 它是学习框架，不是另一套物理引擎；低成本观测可开数千环境，RTX 感知训练另开小批量作业 |
| **MuJoCo** | 轮腿底盘、云台和执行器控制的快速回归；简化弹道；跨引擎鲁棒性检查 | 模型透明、调试快，适合控制消融；不作为高保真相机和完整赛场的主后端 |
| **Gazebo / Ignition** | ROS 2 兼容、历史 RoboMaster 场地和社区插件适配 | 便于复用已有 ROS/Gazebo 生态；不因为迁移方便而降低 Isaac Sim 主实现精度 |

因此“用 Isaac Sim 做哨兵”在本项目中的准确含义是：世界、碰撞、传感器、弹丸和渲染运行在 Isaac Sim；需要大规模强化学习的任务通过 Isaac Lab 管理；ROS 2 只是接口，不决定物理后端。

## 3. 模块边界

### 3.1 轮腿底盘

沿用当前 6-DOF `hip-knee-wheel` 资产和 28 维低层策略。上层只发送连续速度与机身高度/姿态参考，不能直接绕过低层平衡器写轮力矩。先完成：

1. 原地稳定和低速移动；
2. 云台运动时的质心/反作用扰动；
3. 发射事件的可配置脉冲扰动；
4. 坡面、减速带、狭窄通道和碰撞恢复。

### 3.2 两轴云台

云台使用 yaw/pitch 两个关节，包含角度、速度、加速度和电机带宽限制。接口分为三层：

- `target_direction`：感知/预测给出的世界系瞄准方向；
- `gimbal_setpoint`：解算后的关节角/角速度；
- `gimbal_state`：编码器状态、饱和、跟踪误差和是否进入允许窗口。

底盘和云台同时运动时，目标方向必须经时间同步的 `map → odom → base_link → gimbal → camera/launcher` 变换，不能用最新姿态拼接旧图像。

### 3.3 视觉、MID360 与定位

- 相机：内参、畸变、曝光、运动模糊、滚动/全局快门、噪声和传输延迟均可配置；
- MID360：复用现有非重复扫描表、逐点时间和运动畸变管线；
- 状态估计：轮速、IMU、LiDAR/视觉里程计融合；
- Nav2：负责全局/局部导航基线，强化学习只替换经过独立验收的局部行为或高层决策，不把安全避障完全埋进不可解释策略。

### 3.4 安全仿真发射机构

同一接口提供三个保真等级，避免在数千环境训练时为每颗弹丸都付出完整刚体成本：

| 等级 | 模型 | 用途 |
|---|---|---|
| L0 | 射击事件 + 解析弹道/射线 + 概率命中 | 大规模策略训练、战术和奖励消融 |
| L1 | 有质量、半径、初速度、自旋、重力、阻力和碰撞的刚体弹丸 | 弹道、遮挡、跳弹排除、装甲命中和视频 |
| L2 | 摩擦轮动态、供弹器状态、发射门控、速度离散、堵转/空发故障 | 机构控制和软硬件在环，不用于默认大规模训练 |

发射链采用状态机：

```text
DISARMED → ARMED → READY → FEEDING → PROJECTILE_EXIT → COOLDOWN
     └──────── 任一异常 / 越界 / 急停 ────────→ SAFE_STOP
```

任何发射事件都必须同时满足：仿真/硬件总使能、裁判允许、云台误差窗口、友军/禁射区检查、弹丸库存、热/速率限制和超时看门狗。参数从按赛季版本固定的配置文件加载，不把弹丸或限制写死在控制代码中。

### 3.5 装甲与裁判系统

装甲板是带 ID 和所属队伍的碰撞/传感区域。命中消息至少包含：

```text
timestamp, shooter_id, target_id, armor_id,
projectile_id, impact_position, impact_velocity, valid_hit
```

裁判状态机管理生命值、弹量、冷却、禁射区、复活/补给和比赛阶段。策略只读取受限的比赛状态，避免直接访问仿真真值形成无法上机的信息泄漏。

## 4. ROS 2 接口草案

| 接口 | 类型/方向 | 说明 |
|---|---|---|
| `/sentinel/cmd_vel` | `geometry_msgs/Twist` → 底盘 | 经过斜率、可行域和急停限制的速度命令 |
| `/sentinel/gimbal/target` | stamped target → 云台 | 带时间戳和参考坐标系的目标方向 |
| `/sentinel/launcher/arm` | service/action | 独立使能；默认 false |
| `/sentinel/launcher/fire` | action | 单次受控请求，不接受无限阻塞触发 |
| `/sentinel/launcher/state` | state topic | 状态机、库存、冷却、故障和使能 |
| `/sentinel/hit` | event topic | 命中事件；可接裁判仿真 |
| `/camera/*`, `/livox/lidar` | sensor topics | 与真机话题和时间语义对齐 |
| `/tf`, `/tf_static` | transforms | 统一底盘、云台、相机、雷达和发射轴 |

所有外部命令都经过一个 `safety_supervisor`；仿真插件和实机驱动共享消息语义，但实现、权限与参数文件严格分离。

## 5. 强化学习任务

不一开始就把导航、瞄准、发射和多机对抗塞进一个巨型端到端策略。采用可归因的课程：

1. **GimbalTrack**：固定底盘，跟踪移动目标，指标为角误差、稳定时间和饱和率；
2. **BalanceWhileAim**：轮腿原地/移动平衡，同时补偿云台和可配置脉冲扰动；
3. **NavigateAndObserve**：到达观察位、保持视线、避障并减少机身剧烈姿态；
4. **FireWindow**：仅在预测命中概率、安全约束和资源状态均满足时选择离散射击时机；
5. **OneVsOne**：部分可观测对抗，先用脚本对手，再做 self-play；
6. **TeamPolicy**：多机器人任务分配、通信受限、延迟/丢包与对手随机化。

每一级先建立非学习基线：PID 云台、弹道解算、Nav2、规则射击窗口和行为树。RL 必须在统一 holdout 中超过基线，才获得替换权限。

## 6. 里程碑与硬门槛

| 里程碑 | 产物 | 通过条件 |
|---|---|---|
| M0 · 资产 | 轮腿 + 云台 + 相机 + MID360 + 装甲 USD/URDF | 质量/惯量/TF 审计；静态无穿透 |
| M1 · 云台 | 独立控制器和跟踪测试 | 阶跃、正弦、底盘转动下误差达标且不饱和 |
| M2 · 发射仿真 | L0/L1、命中事件和裁判状态 | 固定种子弹道可复现；非法请求 100% 被拒绝 |
| M3 · 感知 | 检测、跟踪、PnP/状态估计 | 独立录像集和仿真域随机化集验收 |
| M4 · 导航 | Nav2 基线 + 轮腿适配 | 场地路线、动态障碍、定位丢失恢复 |
| M5 · 自主哨兵 | 行为树/RL 分层策略 | 多种子、多对手、受限观测，无真值泄漏 |
| M6 · Gazebo | 相同 ROS 接口适配 | 关键消息、TF、控制频率和回放一致 |
| M7 · 实机 | 低能量、规则合规原型 | 独立安全评审、物理保险、急停和封闭区测试 |

## 7. 代码落点

计划使用以下目录，避免把比赛逻辑塞进现有 TinyMal 环境：

```text
robots/robomaster/
  urdf/               # 原创模块化描述
  usd/                # Isaac Sim 组合资产
  mjcf/               # 底盘/云台控制 twin
backends/isaac_lab/actuatex_lab/sentinel/
  scene_cfg.py
  gimbal_env_cfg.py
  fire_window_env_cfg.py
  referee.py
backends/mujoco/sentinel/
  gimbal_control.py
  ballistic_l0.py
navigation/robomaster/
  config/
  launch/
interfaces/actuatex_msgs/
```

所有第三方场地图、机器人 mesh 和代码先做许可证审计；来源不清或限制再分发的资产只保留下载脚本与哈希，不进入仓库。

## 8. 2026 参考基线

- 当季规则以官方 [RoboMaster 2026 University Championship Rule Manual V1.4.2](https://bbs-web-static.robomaster.com/1f59fe9c9d154752a5e456e1f7139d4c1777519819115/RoboMaster%202026%20University%20Championship%20Rule%20Manual%20V1.4.2%EF%BC%8820260430%EF%BC%89.pdf) 为准；仓库只记录版本，不声称替代官方安全与参赛审查。
- [RoboMaster OSS RMUC21 Ignition Gazebo 仿真环境](https://github.com/robomaster-oss/rmuc21_ignition_simulator) 可作为 Gazebo 场景和 ROS 2 接口的历史参考，但不能直接代表 2026 赛制。
- [robomaster_ros](https://github.com/jeguzzi/robomaster_ros) 已提供 DJI S1/EP 的 ROS 2 驱动、描述、云台和 blaster 接口，可用于消息与真机适配思路；ActuateX 不把它当作原创轮腿平台的底层驱动。
- 2026 COD 战队公开了[哨兵导航、决策、自瞄与下位机资料](https://bbs.robomaster.com/article/1882897?source=1)，适合作为行为树、Nav2 和自动瞄准工程拆分的案例。
- [The Cambridge RoboMaster](https://arxiv.org/abs/2405.02198) 展示了基于定制 S1、ROS 2、板载自治和多智能体强化学习的研究平台，可作为后期多机器人 benchmark 参考。

这些项目提供架构经验，不意味着其资产或代码自动适用 MIT。引入前仍需逐项核对许可证、赛季版本、坐标和硬件差异。

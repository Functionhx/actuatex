# ActuateX 串联轮腿教学机器人资产说明

`urdf/actuatex_serial_wheel_legged.urdf` 是本仓库使用基础几何体原创构建的教学资产，不包含第三方 CAD、STL 或纹理。

两侧各由“髋关节—膝关节—轮关节”组成的串联拓扑，以及代表性参数（上/下连杆长约 0.15/0.25 m、轮半径 0.0675 m、主体和轮质量量级）参考了 [Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym) 的公开实现。该参考项目采用 BSD-3-Clause 许可证，检索时所用版本为提交 `c354431e5633`。本资产重新命名、重新建模并重新计算惯量，避免复制其网格文件。

本资产随 ActuateX 以 MIT 许可证发布。参数仅用于仿真教学，制造实体前必须重新完成结构强度、电机选型、碰撞包络和惯量标定。

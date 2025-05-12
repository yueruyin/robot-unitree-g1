# 宇树科技 G1 机器人控制学习研究

## 项目概述

本项目是基于宇树科技（Unitree）G1人形机器人的控制系统实现，使用MuJoCo物理引擎进行仿真。该系统支持G1机器人的全身控制，包括双足行走、蹲起、手臂与手指的精确控制等功能。

![01](resources%2Fg1%2Fimages%2F01_mujoco_g1_53dof.png)
![02](resources%2Fg1%2Fimages%2F02_mujoco_g1_53dof.png)

## 系统特点

- **53自由度全身控制**：支持G1机器人的全部53个自由度联合控制
- **双足步态控制**：实现稳定行走、转向和位置控制
- **手臂运动学控制**：支持逆运动学控制机械臂到达目标位置
- **手指精细控制**：支持手指的抓取和释放动作
- **基于MuJoCo的物理仿真**：提供高精度的物理特性模拟

## 项目结构

```
.
├── deploy/             # 部署和运行脚本
│   ├── config/         # 配置文件
│   └── deploy_mujoco53.py  # 主部署脚本
├── resources/          # 资源文件
│   └── g1/             # G1机器人模型和场景定义
│       ├── meshes/     # 3D模型网格文件
│       ├── g1_53dof.xml  # 机器人MuJoCo模型定义
│       └── scene53.xml   # 仿真场景定义
└── pre_train/          # 预训练模型
    └── g1/             # G1机器人预训练模型
```

## 主要功能

### 1. 双足行走控制
- 支持前进/后退、侧向移动和转向控制
- 通过命令行交互实时调整运动参数
- 支持目标位置寻路功能

### 2. 手臂控制
- 基于阻尼最小二乘法的逆运动学求解
- 支持左右手臂独立控制
- PD控制器实现关节位置精确控制

### 3. 手指控制
- 支持手指抓取和释放动作
- 可以实现对物体的交互操作

### 4. 蹲起控制
- 可调节高度的蹲起动作
- 平稳的过渡动作保证稳定性

## 使用方法

### 环境要求
- Python 3.8+
- MuJoCo 2.3.0+
- PyTorch 1.9+
- NumPy, YAML

### 运行仿真

```bash
python deploy/deploy_mujoco53.py
```

### 交互控制指令

在运行仿真后，可以使用以下命令行指令进行交互控制：

- `left_pos x y z`: 设置左臂目标位置，例如: `left_pos 1.2 -1.5 1.0`
- `right_pos x y z`: 设置右臂目标位置，例如: `right_pos 1.2 -1.5 1.0`
- `left_toggle`: 切换左手抓取/释放状态
- `right_toggle`: 切换右手抓取/释放状态
- `left_wrist_roll angle`: 设置左手腕roll角度，例如: `left_wrist_roll 1.57`
- `right_wrist_roll angle`: 设置右手腕roll角度，例如: `right_wrist_roll 1.57`
- `squat`: 开始/停止下蹲动作
- `walk`: 开始/停止走路模式
- `goto x y`: 设置行走目标位置，例如: `goto 2.0 3.0`
- `help`: 显示帮助信息

## 配置文件

`deploy/config/g1_53.yaml` 包含了关节控制参数、默认姿态和控制增益等配置：

- `kps/kds`: PD控制器的比例/微分增益
- `default_angles`: 默认关节角度
- `simulation_dt`: 仿真时间步
- `control_decimation`: 控制抽取率
- `height_cmd`: 默认站立高度

## 贡献与开发

如需扩展或修改模型，可关注以下文件：

- `resources/g1/g1_53dof.xml`: 机器人模型定义
- `deploy/deploy_mujoco53.py`: 控制器逻辑实现

## 许可证

本项目遵循 LICENSE 文件中指定的许可条款。

## 感谢🙏
- [legged\_gym](https://github.com/leggedrobotics/legged_gym): 构建训练与运行代码的基础。
- [rsl\_rl](https://github.com/leggedrobotics/rsl_rl.git): 强化学习算法实现。
- [mujoco](https://github.com/google-deepmind/mujoco.git): 提供强大仿真功能。
- [unitree](https://github.com/unitreerobotics/unitree_rl_gym): 提供g1-mjcf和代码示例等。

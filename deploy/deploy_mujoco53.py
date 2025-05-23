import time
import threading
import queue
import collections
import numpy as np
import mujoco
import mujoco.viewer
import yaml
import argparse
import torch
import math  # 为方向计算添加
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import os

# 导入RRT算法
from deploy.rrt.RRT_star import RRTStar3D, Node3D

dir_root = os.path.dirname(os.path.abspath(__file__)).replace("deploy", '')

# 全局变量，用于走路控制
cmd = [0, 0, 0]  # 行走命令 [前进速度, 侧向速度, 转向速度]
xy = [0, 0]  # 目标位置 [x, y]
switch = False  # 是否激活行走模式

# 实际应用中可以从场景中提取障碍物信息 障碍物格式：(x, y, z, 长, 宽, 高)
obstacles = [(1.00, -1.62, 0.87, 0.08, 0.04, 0.12)]


# obstacles = []

@dataclass
class ArmConfig:
    """机械臂配置参数数据类"""
    # 控制器增益
    kps: np.ndarray = None
    kds: np.ndarray = None

    # 默认关节角度
    default_angles: np.ndarray = None

    # 关节限位
    joint_limits: Dict[str, List[float]] = field(default_factory=dict)

    # 关节力矩限制
    joint_torque_limits: Dict[str, float] = field(default_factory=dict)

    # 工作空间限制 - 根据mujoco模型更新
    workspace_limits: Dict[str, List[float]] = field(default_factory=lambda: {
        'x': [-2, 5],  # 根据模型实际可达范围调整
        'y': [-2, 5],  # 根据模型实际可达范围调整
        'z': [-2, 5]  # 根据模型实际可达范围调整
    })

    # 控制频率参数
    simulation_dt: float = 0.001
    control_decimation: int = 10

    # 阻尼最小二乘IK参数
    damping: float = 0.1
    max_dq: float = 0.1

    # 右手默认角度和握物体的角度
    hand_default_angles: np.ndarray = None
    hand_grasp_angles: np.ndarray = None

    # 右手控制参数
    hand_kps: np.ndarray = None
    hand_kds: np.ndarray = None

    # 命令缩放
    cmd_scale: np.ndarray = None

    # 路径相关参数
    motion_path: str = None
    policy_path: str = None

    # 观察向量缩放参数
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    action_scale: float = 0.25

    # 初始命令
    cmd_init: np.ndarray = None

    # 高度相关参数
    height_cmd: float = 0.80  # 默认高度
    min_height: float = 0.30  # 最小高度
    max_squat_depth: float = 0.3  # 最大下蹲深度

    # 历史长度
    obs_history_len: int = 6

    @classmethod
    def from_yaml(cls, config_path: str) -> 'ArmConfig':
        """从YAML配置文件加载配置"""
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        instance = cls()
        instance.kps = np.array(config["kps"], dtype=np.float32)
        instance.kds = np.array(config["kds"], dtype=np.float32)
        instance.default_angles = np.array(config["default_angles"], dtype=np.float32)
        instance.simulation_dt = config["simulation_dt"]
        instance.control_decimation = config["control_decimation"]
        instance.policy_path = config["policy_path"]

        # 加载命令缩放参数
        if "cmd_scale" in config:
            instance.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        # 加载角度缩放参数
        if "ang_vel_scale" in config:
            instance.ang_vel_scale = config["ang_vel_scale"]
        if "dof_pos_scale" in config:
            instance.dof_pos_scale = config["dof_pos_scale"]
        if "dof_vel_scale" in config:
            instance.dof_vel_scale = config["dof_vel_scale"]
        if "action_scale" in config:
            instance.action_scale = config["action_scale"]

        # 加载高度相关参数
        if "height_cmd" in config:
            instance.height_cmd = config["height_cmd"]
        if "min_height" in config:
            instance.min_height = config["min_height"]
        if "max_squat_depth" in config:
            instance.max_squat_depth = config["max_squat_depth"]

        # 加载命令和观察历史相关参数
        if "cmd_init" in config:
            instance.cmd_init = np.array(config["cmd_init"], dtype=np.float32)
        if "obs_history_len" in config:
            instance.obs_history_len = config["obs_history_len"]

        # 添加关节限位 - 根据mujoco模型更新
        instance.joint_limits = {
            # 右臂关节限位
            'right_shoulder_pitch': [-3.0892, 2.6704],  # 肩部俯仰
            'right_shoulder_roll': [-2.2515, 1.5882],  # 肩部横滚
            'right_shoulder_yaw': [-2.618, 2.618],  # 肩部偏航
            'right_elbow': [-1.0472, 2.0944],  # 肘部
            'right_wrist_roll': [-1.97222, 1.97222],  # 腕部横滚
            'right_wrist_pitch': [-1.61443, 1.61443],  # 腕部俯仰
            'right_wrist_yaw': [-1.61443, 1.61443],  # 腕部偏航

            # 左手关节限位
            'L_thumb_proximal_yaw': [-0.1, 1.3],
            'L_thumb_proximal_pitch': [0, 0.5],
            'L_thumb_intermediate': [0, 0.8],
            'L_thumb_distal': [0, 1.2],
            'L_index_proximal': [0, 1.7],
            'L_index_intermediate': [0, 1.7],
            'L_middle_proximal': [0, 1.7],
            'L_middle_intermediate': [0, 1.7],
            'L_ring_proximal': [0, 1.7],
            'L_ring_intermediate': [0, 1.7],
            'L_pinky_proximal': [0, 1.7],
            'L_pinky_intermediate': [0, 1.7],

            # 右手关节限位
            'R_thumb_proximal_yaw': [-0.1, 1.3],
            'R_thumb_proximal_pitch': [0, 0.5],
            'R_thumb_intermediate': [0, 0.8],
            'R_thumb_distal': [0, 1.2],
            'R_index_proximal': [0, 1.7],
            'R_index_intermediate': [0, 1.7],
            'R_middle_proximal': [0, 1.7],
            'R_middle_intermediate': [0, 1.7],
            'R_ring_proximal': [0, 1.7],
            'R_ring_intermediate': [0, 1.7],
            'R_pinky_proximal': [0, 1.7],
            'R_pinky_intermediate': [0, 1.7]
        }

        # 添加关节力矩限制 - 根据mujoco模型更新
        instance.joint_torque_limits = {
            # 右臂关节力矩限制
            'right_shoulder_pitch': 25,
            'right_shoulder_roll': 25,
            'right_shoulder_yaw': 25,
            'right_elbow': 25,
            'right_wrist_roll': 25,
            'right_wrist_pitch': 5,
            'right_wrist_yaw': 5,

            # 左手关节力矩限制
            'L_thumb_proximal_yaw': 1,
            'L_thumb_proximal_pitch': 1,
            'L_thumb_intermediate': 1,
            'L_thumb_distal': 1,
            'L_index_proximal': 1,
            'L_index_intermediate': 1,
            'L_middle_proximal': 1,
            'L_middle_intermediate': 1,
            'L_ring_proximal': 1,
            'L_ring_intermediate': 1,
            'L_pinky_proximal': 1,
            'L_pinky_intermediate': 1,

            # 右手关节力矩限制
            'R_thumb_proximal_yaw': 1,
            'R_thumb_proximal_pitch': 1,
            'R_thumb_intermediate': 1,
            'R_thumb_distal': 1,
            'R_index_proximal': 1,
            'R_index_intermediate': 1,
            'R_middle_proximal': 1,
            'R_middle_intermediate': 1,
            'R_ring_proximal': 1,
            'R_ring_intermediate': 1,
            'R_pinky_proximal': 1,
            'R_pinky_intermediate': 1
        }

        # 设置手指默认角度（完全展开状态）
        instance.hand_default_angles = np.zeros(12)  # 12个手指关节

        # 设置手部PD控制增益
        instance.hand_kps = instance.kps[41:53]  # 手部关节的kp
        instance.hand_kds = instance.kds[41:53]  # 手部关节的kd

        return instance


class ArmBaseController:
    """机械臂控制器基类"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: ArmConfig, arm_side: str):
        """
        初始化机械臂控制器基类

        参数:
            model: MuJoCo 模型
            data: MuJoCo 数据
            config: 机械臂配置
            arm_side: 机械臂类型，"left" 或 "right"
        """
        self.model = model
        self.data = data
        self.config = config
        self.arm_side = arm_side

        # 定义关节索引 - 根据arm_side设置正确的索引
        # 0-5: 左腿6个关节
        # 6-11: 右腿6个关节
        # 12-14: 腰部关节
        # 15-21: 左臂7个关节
        # 22-28: 左手7个关节
        # 29-35: 右臂7个关节
        # 36-42: 右手7个关节
        if arm_side == "left":
            self.arm_joint_indices = list(range(15, 22))  # 左臂关节
            self.hand_joint_indices = list(range(22, 34))  # 左手关节
        else:  # arm_side == "right"
            self.arm_joint_indices = list(range(34, 41))  # 右臂关节
            self.hand_joint_indices = list(range(41, 53))  # 右手关节

        # 打印关节索引信息
        print(f"{arm_side}臂关节索引: {self.arm_joint_indices}")

        # 初始化目标位置和命令队列
        self.target_position_queue = queue.Queue()
        self.cmd_queue = queue.Queue()
        self.timer = None
        self.running = False
        self.move_speed = 0.006
        self.last_target_pos = np.zeros(3)
        self.has_user_input = False
        # 添加单独控制特定关节的标志和目标值 # 格式: {关节索引: 目标角度}
        self.single_joint_control = {}
        # 添加手腕roll关节平滑过渡的定时器
        self.wrist_roll_timer = None

        # 添加RRT路径规划相关变量
        self.rrt_path = []  # 存储RRT规划的路径
        self.current_path_index = 0  # 当前执行到的路径点索引
        self.path_executing = False  # 是否正在执行路径

        # 添加路径可视化相关变量
        self.path_sites = []  # 存储路径点的site引用
        self.sites_added = False  # 标记是否已添加site到模型

        self.arm_joint_names = [
            f'{arm_side}_shoulder_pitch_joint',
            f'{arm_side}_shoulder_roll_joint',
            f'{arm_side}_shoulder_yaw_joint',
            f'{arm_side}_elbow_joint',
            f'{arm_side}_wrist_roll_joint',
            f'{arm_side}_wrist_pitch_joint',
            f'{arm_side}_wrist_yaw_joint'
        ]

        # 获取机械臂关节在模型中的索引
        self.arm_joint_ids = self._get_arm_joint_ids()

        # 获取末端执行器在模型中的索引
        self.end_effector_id = model.body(f'{arm_side}_wrist_yaw_link').id

        # 初始化目标关节位置
        if arm_side == "left":
            # 左臂关节位置从索引15开始
            self.dof_offset = 15
        else:
            # 右臂关节位置从索引34开始
            self.dof_offset = 34

        self.target_dof_pos = config.default_angles.copy()

    def _get_arm_joint_ids(self) -> List[int]:
        """获取机械臂关节在模型中的索引"""
        joint_ids = []
        for name in self.arm_joint_names:
            try:
                joint_id = self.model.joint(name).id
                joint_ids.append(joint_id)
            except KeyError:
                print(f"警告: 没有此关节ID {name}")
                joint_ids.append(None)

        # 确认索引对应关系
        print(f"{self.arm_side}臂关节ID和DOF索引对应关系:")
        for i, joint_id in enumerate(joint_ids):
            if joint_id is not None:
                dof_adr = self.model.jnt_dofadr[joint_id] if joint_id < len(self.model.jnt_dofadr) else None
                qpos_adr = self.model.jnt_qposadr[joint_id] if joint_id < len(self.model.jnt_qposadr) else None
                print(f"  关节 {self.arm_joint_names[i]}: ID={joint_id}, DOF索引={dof_adr}, QPOS索引={qpos_adr}")

        return joint_ids

    def set_target_position(self, position: np.ndarray) -> bool:
        """
        设置末端执行器的目标位置，使用RRT*算法进行路径规划

        参数:
            position: 目标位置坐标 [x, y, z]

        返回:
            bool: 位置是否有效
        """
        if not self._check_position_limits(position):
            return False

        # 获取当前末端位置作为起点
        current_pos = self.data.xpos[self.end_effector_id].copy()

        # 设置工作空间边界
        limits = self.config.workspace_limits
        bounds = (
            limits['x'][0], limits['x'][1],  # x边界
            limits['y'][0], limits['y'][1],  # y边界
            limits['z'][0], limits['z'][1]  # z边界
        )

        # 执行RRT*路径规划
        try:
            rrt = RRTStar3D(
                start=tuple(current_pos),
                goal=tuple(position),
                obstacles=obstacles,
                bounds=bounds,
                step_size=0.005,  # 较小的步长以获得更平滑的路径
                max_iter=3000,
                search_radius=0.01
            )

            path = rrt.plan()

            if path:
                print(f"RRT规划成功，路径点数量: {len(path)}")
                # 清空现有路径和队列
                self.rrt_path = path
                self.current_path_index = 0
                self.path_executing = True

                # 清空现有队列，只保留第一个路径点
                while not self.target_position_queue.empty():
                    try:
                        self.target_position_queue.get_nowait()
                    except queue.Empty:
                        break

                # 放入第一个路径点
                if len(path) > 0:
                    first_point = np.array(path[0])
                    self.target_position_queue.put(first_point)
                    self.last_target_pos = position.copy()  # 最终目标位置
                    self.has_user_input = True

                    # 可视化路径
                    self.visualize_path(path)

                    return True
            else:
                print("RRT规划失败，使用直线路径")
                self.clear_path_visualization()
                # 规划失败则使用直线路径
                self.target_position_queue.put(position)
                self.last_target_pos = position.copy()
                self.has_user_input = True
                return True

        except Exception as e:
            print(f"RRT路径规划出错: {e}")
            # 规划出错则使用直线路径
            self.target_position_queue.put(position)
            self.last_target_pos = position.copy()
            self.has_user_input = True
            return True

    def visualize_path(self, path):
        """
        使用site对象可视化RRT路径
        
        参数:
            path: RRT规划的路径点列表
        """
        # 清除之前的可视化路径点
        self.clear_path_visualization()

        # 创建新的site进行路径可视化
        self.path_sites = []

        # 使用viewer.user_scn添加自定义geoms
        for i, point in enumerate(path):
            if i % 5 != 0 and i != len(path) - 1:
                continue
            # 添加一个标记为路径点的site
            site_name = f"{self.arm_side}_path_site_{i}"
            site_rgb = [0.0, 0.8, 0.2] if i < len(path) - 1 else [1.0, 0.0, 0.0]  # 终点为红色，其他点为绿色
            site_size = 0.005 if i < len(path) - 1 else 0.015  # 终点大一点

            # 将site信息保存，以便在渲染时添加到场景中
            self.path_sites.append({
                'name': site_name,
                'pos': point,
                'size': site_size,
                'rgba': site_rgb + [0.7]  # 添加alpha通道
            })

        # 标记需要添加sites到模型
        self.sites_added = True

    def clear_path_visualization(self):
        """清除路径可视化"""
        self.path_sites = []
        self.sites_added = False

    def add_path_sites_to_scene(self, scn):
        """
        将路径点添加到场景中进行可视化
        
        参数:
            scn: mjvScene对象
        """
        if not self.sites_added or not self.path_sites:
            return

        # 为每个路径点添加一个geom到场景
        for i, site_info in enumerate(self.path_sites):
            if scn.ngeom < scn.maxgeom:  # 检查是否还有空间添加geom
                g = scn.geoms[scn.ngeom]
                # 设置为球体
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([site_info['size'], 0, 0]),
                    np.array(site_info['pos']),
                    np.eye(3).flatten(),
                    np.array(site_info['rgba'])
                )
                scn.ngeom += 1

        # 为相邻路径点之间添加连接线
        for i in range(len(self.path_sites) - 1):
            if scn.ngeom < scn.maxgeom:  # 检查是否还有空间添加geom
                start_pos = np.array(self.path_sites[i]['pos'])
                end_pos = np.array(self.path_sites[i + 1]['pos'])

                # 添加连接线
                g = scn.geoms[scn.ngeom]
                # 初始化为胶囊体形状
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.array([0.003, 0, 0]),  # 线的半径
                    np.zeros(3),  # 初始位置将被connector函数更新
                    np.eye(3).flatten(),  # 旋转矩阵(默认)
                    np.array([0.0, 0.8, 0.8, 0.7])  # 青色，半透明
                )

                # 使用正确的参数格式调用connector函数
                mujoco.mjv_connector(
                    g,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    0.003,  # 宽度
                    start_pos,  # 起点
                    end_pos  # 终点
                )

                scn.ngeom += 1

    def _check_position_limits(self, pos: np.ndarray) -> bool:
        """
        检查目标位置是否在机器人工作空间范围内

        参数:
            pos: 目标位置坐标 [x, y, z]
        返回:
            bool: 位置是否有效
        """
        limits = self.config.workspace_limits
        return (limits['x'][0] <= pos[0] <= limits['x'][1] and
                limits['y'][0] <= pos[1] <= limits['y'][1] and
                limits['z'][0] <= pos[2] <= limits['z'][1])

    def pd_control(self, target_q, q, kp, target_dq, dq, kd) -> np.ndarray:
        """
        PD控制器计算关节力矩

        参数:
            target_q: 目标关节角度
            q: 当前关节角度
            kp: 位置增益
            target_dq: 目标关节速度
            dq: 当前关节速度
            kd: 速度增益
        返回:
            tau: 计算得到的关节力矩
        """
        return (target_q - q) * kp + (target_dq - dq) * kd

    def damped_ls_ik(self, j_eef: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """
        使用阻尼最小二乘法求解逆运动学

        参数:
            j_eef: 末端执行器的雅可比矩阵
            dx: 期望的末端执行器位移
        返回:
            dq: 关节角度增量
        """
        # 提取机械臂关节的雅可比矩阵
        # 获取机械臂的自由度数量
        n_arm_dof = len(self.arm_joint_ids)

        # 找到机械臂关节在整个模型的自由度索引中的位置
        qvel_indices = []
        for joint_id in self.arm_joint_ids:
            qvel_indices.append(self.model.jnt_dofadr[joint_id])

        # 提取机械臂关节对应的雅可比矩阵列
        j_arm = np.zeros((3, n_arm_dof))
        for i, idx in enumerate(qvel_indices):
            j_arm[:, i] = j_eef[:, idx]

        # 计算阻尼最小二乘解
        j_arm_T = j_arm.T

        # 增加阻尼因子以提高稳定性
        lmbda = np.eye(3) * (self.config.damping ** 2)

        # 添加关节权重矩阵，使运动更自然
        # 肩部关节权重更大，手腕关节权重更小
        w = np.diag([1.0, 1.0, 1.0, 0.8, 0.6, 0.6, 0.4])

        try:
            # 计算加权的阻尼最小二乘解
            temp = j_arm @ w @ j_arm_T + lmbda
            dq = w @ j_arm_T @ np.linalg.solve(temp, dx)

            # 限制关节速度
            max_dq = self.config.max_dq
            norm = np.linalg.norm(dq)
            if norm > max_dq:
                dq *= max_dq / norm

            # 为小运动添加扰动以避免停滞
            if norm < 1e-5:
                dq += np.random.normal(0, 0.001, dq.shape)

            return dq

        except np.linalg.LinAlgError:
            print("警告：逆运动学计算出现奇异性")
            return np.zeros(n_arm_dof)

    def apply_joint_limits(self) -> None:
        """应用关节角度限制"""
        for i, joint_name in enumerate(self.arm_joint_names):
            name = joint_name.replace('_joint', '')  # 从关节名中提取基础名称
            if name in self.config.joint_limits:
                limits = self.config.joint_limits[name]
                # 计算对应的DOF索引
                dof_idx = self.dof_offset + i
                if dof_idx < len(self.target_dof_pos):
                    self.target_dof_pos[dof_idx] = np.clip(
                        self.target_dof_pos[dof_idx], limits[0], limits[1]
                    )

    # 添加设置腕关节roll角度的方法
    def set_wrist_roll_angle(self, angle: float) -> bool:
        """
        设置手腕roll关节角度，使用平滑过渡

        参数:
            angle: 目标角度（弧度）
        返回:
            是否成功设置
        """
        # 查找wrist_roll关节在关节列表中的索引
        wrist_roll_idx = None
        for i, name in enumerate(self.arm_joint_names):
            if name.endswith('wrist_roll_joint'):
                wrist_roll_idx = i
                break

        # 获取关节限位
        joint_name = self.arm_joint_names[wrist_roll_idx].replace('_joint', '')
        if joint_name in self.config.joint_limits:
            limits = self.config.joint_limits[joint_name]
            # 限制角度在有效范围内
            angle = np.clip(angle, limits[0], limits[1])

        # 设置目标角度
        dof_idx = self.dof_offset + wrist_roll_idx
        if dof_idx < len(self.target_dof_pos):
            # 获取当前角度
            current_angle = self.target_dof_pos[dof_idx]

            # 计算角度差值
            angle_diff = angle - current_angle

            # 使用较小的步长进行平滑过渡（减小步长可以让变化更慢）
            # 如果角度差值很大，使用更小的步长
            if abs(angle_diff) > self.move_speed:
                # 按步长方向移动
                new_angle = current_angle + np.sign(angle_diff) * self.move_speed
                # 将该关节添加到单独控制列表中，但使用平滑过渡的目标
                self.single_joint_control[dof_idx] = new_angle

                # 使用定时器继续平滑过渡到目标角度
                if hasattr(self, 'wrist_roll_timer') and self.wrist_roll_timer:
                    self.wrist_roll_timer.cancel()

                # 创建定时器继续执行平滑过渡
                self.wrist_roll_timer = threading.Timer(0.02, lambda: self.set_wrist_roll_angle(angle))
                self.wrist_roll_timer.daemon = True
                self.wrist_roll_timer.start()
            else:
                # 直接设置到目标角度
                self.target_dof_pos[dof_idx] = angle
                # 将该关节添加到单独控制列表中
                self.single_joint_control[dof_idx] = angle
                print(f"已设置{self.arm_side}侧手腕roll关节角度为: {angle:.4f} rad")

                # 如果有定时器，取消它
                if hasattr(self, 'wrist_roll_timer') and self.wrist_roll_timer:
                    self.wrist_roll_timer.cancel()
                    self.wrist_roll_timer = None

            return True
        else:
            print(f"错误: DOF索引{dof_idx}超出范围")
            return False

    def apply_torque_limits(self, tau: np.ndarray) -> np.ndarray:
        """
        应用关节力矩限制

        参数:
            tau: 计算得到的关节力矩
        返回:
            tau: 应用限制后的关节力矩
        """
        for i, joint_name in enumerate(self.arm_joint_names):
            name = joint_name.replace('_joint', '')  # 从关节名中提取基础名称
            if name in self.config.joint_torque_limits:
                limit = self.config.joint_torque_limits[name]
                # 获取对应的力矩索引
                torque_idx = self.dof_offset + i
                tau[torque_idx] = np.clip(tau[torque_idx], -limit, limit)
        return tau

    def update(self) -> None:
        """更新控制器状态并计算控制输出"""
        # 检查是否有新的目标位置
        try:
            new_target = self.target_position_queue.get_nowait()
            if new_target is not None:
                self.last_target_pos = new_target
                self.has_user_input = True
        except queue.Empty:
            pass

        if not self.has_user_input:
            # 在没有用户输入时，保持默认姿态，但保留单独控制的关节角度
            # 先复制默认姿态
            self.target_dof_pos = self.config.default_angles.copy()

            # 应用单独控制的关节角度
            for dof_idx, angle in self.single_joint_control.items():
                if dof_idx < len(self.target_dof_pos):
                    self.target_dof_pos[dof_idx] = angle
            return

        # 获取机械臂末端雅可比矩阵
        jacp = np.zeros((3, self.model.nv))  # 位置雅可比矩阵
        jacr = np.zeros((3, self.model.nv))  # 旋转雅可比矩阵

        # 计算末端执行器的雅可比矩阵
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.end_effector_id)

        # 获取当前末端位置
        current_pos = self.data.xpos[self.end_effector_id]

        # 计算位置误差
        pos_error = self.last_target_pos - current_pos

        # 检查当前目标点是否已经到达
        error_norm = np.linalg.norm(pos_error)
        if error_norm < 0.05 and self.path_executing and self.current_path_index < len(self.rrt_path) - 1:
            # 到达当前路径点，转到下一个路径点
            self.current_path_index += 1
            next_point = np.array(self.rrt_path[self.current_path_index])
            self.target_position_queue.put(next_point)
            print(f"到达路径点 {self.current_path_index}/{len(self.rrt_path)}, 下一点: {next_point}")
            return

        # 坐标变换：将误差转换到机器人基座坐标系
        base_id = self.model.body('pelvis').id
        base_quat = self.data.xquat[base_id]

        # 创建旋转矩阵
        def quat_to_mat(quat):
            w, x, y, z = quat
            return np.array([
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
            ])

        # 将误差转换到基座坐标系
        R = quat_to_mat(base_quat)
        pos_error = R.T @ pos_error

        # 根据误差大小动态调整步长
        scale = min(0.5, max(0.1, error_norm))
        pos_error *= scale

        # 使用改进的阻尼最小二乘法求解逆运动学
        dq = self.damped_ls_ik(jacp, pos_error)

        # 保存当前单独控制的关节角度
        saved_angles = {}
        for dof_idx, angle in self.single_joint_control.items():
            if dof_idx < len(self.target_dof_pos):
                saved_angles[dof_idx] = angle

        # 更新目标关节角度（各自7个关节）
        start_idx = self.dof_offset
        end_idx = start_idx + 7
        self.target_dof_pos[start_idx:end_idx] += dq * self.move_speed

        # 恢复单独控制的关节角度（覆盖IK结果）
        for dof_idx, angle in saved_angles.items():
            if dof_idx < len(self.target_dof_pos):
                self.target_dof_pos[dof_idx] = angle

        # 应用关节限位
        self.apply_joint_limits()

    def compute_control(self) -> np.ndarray:
        """计算控制输出"""
        # 调用更新函数
        self.update()

        # 初始化一个全零的控制向量
        tau = np.zeros(self.model.nu, dtype=np.float32)

        # PD控制计算关节力矩
        arm_kp = np.zeros(len(self.arm_joint_indices), dtype=np.float32)
        arm_kd = np.zeros(len(self.arm_joint_indices), dtype=np.float32)
        current_pos = np.zeros(len(self.arm_joint_indices), dtype=np.float32)
        current_vel = np.zeros(len(self.arm_joint_indices), dtype=np.float32)

        # 获取当前关节位置、速度和PD增益
        for i, joint_idx in enumerate(self.arm_joint_indices):
            # 获取当前关节状态
            if 7 + joint_idx < len(self.data.qpos) and 6 + joint_idx < len(self.data.qvel):
                current_pos[i] = self.data.qpos[7 + joint_idx]
                current_vel[i] = self.data.qvel[6 + joint_idx]

            # 获取PD控制增益
            if joint_idx < len(self.config.kps):
                arm_kp[i] = self.config.kps[joint_idx]
                arm_kd[i] = self.config.kds[joint_idx]
            else:
                # 使用默认值
                arm_kp[i] = 150.0
                arm_kd[i] = 2.0

        # 获取目标关节位置
        target_dof_pos = np.zeros(len(self.arm_joint_indices), dtype=np.float32)
        for i, joint_idx in enumerate(self.arm_joint_indices):
            if self.dof_offset + i < len(self.target_dof_pos):
                target_dof_pos[i] = self.target_dof_pos[self.dof_offset + i]
            elif joint_idx < len(self.config.default_angles):
                target_dof_pos[i] = self.config.default_angles[joint_idx]

        # 计算关节力矩
        arm_tau = self.pd_control(
            target_dof_pos,
            current_pos,
            arm_kp,
            np.zeros_like(arm_kp),
            current_vel,
            arm_kd
        )

        # 应用关节力矩限制
        for i, joint_name in enumerate(self.arm_joint_names):
            # 从关节名中提取基础名称
            name = joint_name.replace('_joint', '')
            if name in self.config.joint_torque_limits and i < len(arm_tau):
                limit = self.config.joint_torque_limits[name]
                arm_tau[i] = np.clip(arm_tau[i], -limit, limit)

        # 将力矩应用到对应的关节
        for i, joint_idx in enumerate(self.arm_joint_indices):
            if joint_idx < self.model.nu and i < len(arm_tau):
                tau[joint_idx] = arm_tau[i]

        return tau

    def start(self) -> None:
        """启动控制器"""
        self.running = True

    def stop(self) -> None:
        """停止控制器"""
        self.running = False
        # 取消手腕roll关节平滑过渡的定时器
        if hasattr(self, 'wrist_roll_timer') and self.wrist_roll_timer:
            self.wrist_roll_timer.cancel()
            self.wrist_roll_timer = None


class ArmLeftController(ArmBaseController):
    """左机械臂控制器类"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: ArmConfig):
        """初始化左机械臂控制器"""
        # 左臂关节名称
        self.arm_joint_names = [
            'left_shoulder_pitch_joint',
            'left_shoulder_roll_joint',
            'left_shoulder_yaw_joint',
            'left_elbow_joint',
            'left_wrist_roll_joint',
            'left_wrist_pitch_joint',
            'left_wrist_yaw_joint'
        ]

        super().__init__(model, data, config, "left")
        self.left_end_effector_id = self.end_effector_id

        # 打印关节名称和索引对应关系
        print("左臂关节名称和索引:")
        for i, name in enumerate(self.arm_joint_names):
            joint_idx = self.arm_joint_ids[i] if i < len(self.arm_joint_ids) else None
            print(
                f"  {name}: 索引 {joint_idx}, DOF索引 {self.arm_joint_indices[i] if i < len(self.arm_joint_indices) else None}")


class ArmRightController(ArmBaseController):
    """右机械臂控制器类"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: ArmConfig):
        """初始化右机械臂控制器"""
        # 右臂关节名称
        self.arm_joint_names = [
            'right_shoulder_pitch_joint',
            'right_shoulder_roll_joint',
            'right_shoulder_yaw_joint',
            'right_elbow_joint',
            'right_wrist_roll_joint',
            'right_wrist_pitch_joint',
            'right_wrist_yaw_joint'
        ]

        super().__init__(model, data, config, "right")
        self.right_end_effector_id = self.end_effector_id

        # 打印关节名称和索引对应关系
        print("右臂关节名称和索引:")
        for i, name in enumerate(self.arm_joint_names):
            joint_idx = self.arm_joint_ids[i] if i < len(self.arm_joint_ids) else None
            print(
                f"  {name}: 索引 {joint_idx}, DOF索引 {self.arm_joint_indices[i] if i < len(self.arm_joint_indices) else None}")


class ArmRightWithHandController:
    """右机械臂带手控制器类"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: ArmConfig):
        """
        初始化机械臂带手控制器
        """
        self.model = model
        self.data = data
        self.config = config

        # 创建右臂控制器作为基础控制器
        self.arm_controller = ArmRightController(model, data, config)
        # 设置右手手指握取角度 (调整为更温和的值以减少抖动)
        self.hand_grasp_angles = np.array([1.3, 0.1, 0.8, 0.4, 0.6, 0.9, 0.6, 0.9, 0.6, 0.9, 0.6, 0.9])
        # self.hand_grasp_angles = np.array([1.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # 右手关节索引 (索引41-53)
        self.hand_joint_indices = list(range(41, 53))

        # 自动识别右手关节的索引
        self.hand_joint_names = [
            'R_thumb_proximal_yaw_joint',
            'R_thumb_proximal_pitch_joint',
            'R_thumb_intermediate_joint',
            'R_thumb_distal_joint',
            'R_index_proximal_joint',
            'R_index_intermediate_joint',
            'R_middle_proximal_joint',
            'R_middle_intermediate_joint',
            'R_ring_proximal_joint',
            'R_ring_intermediate_joint',
            'R_pinky_proximal_joint',
            'R_pinky_intermediate_joint'
        ]

        # 获取右手关节在模型中的索引
        self.hand_joint_ids = self._get_hand_joint_ids()

        # 打印手部关节索引信息
        print("右手关节索引:")
        for i, name in enumerate(self.hand_joint_names):
            print(
                f"  {name}: 模型索引 {self.hand_joint_ids[i]}, DOF索引 {self.hand_joint_indices[i] if i < len(self.hand_joint_indices) else None}")

        # 获取当前手部关节角度作为初始位置
        hand_qpos = np.zeros(12)
        for i, joint_id in enumerate(self.hand_joint_ids):
            try:
                if joint_id is not None:
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    hand_qpos[i] = self.data.qpos[qpos_adr]
            except Exception as e:
                print(f"警告: 无法获取手部关节 {self.hand_joint_names[i]} 的位置: {e}")
                hand_qpos[i] = 0.0

        # 初始化右手目标位置为当前位置，避免突然运动
        self.hand_target_pos = hand_qpos.copy()

        # 逐渐过渡到默认状态的插值因子
        self.interp_factor = 0.0
        self.transition_speed = 0.001  # 每次更新增加的插值因子
        self.transitioning = True  # 标记是否正在过渡中

        # 右手状态标记 - True表示握住，False表示松开
        self.is_grasping = False

        # 添加右手指令队列
        self.hand_cmd_queue = queue.Queue()

    def _get_hand_joint_ids(self) -> List[int]:
        """获取手部关节在模型中的索引"""
        joint_ids = []
        for name in self.hand_joint_names:
            try:
                joint_id = self.model.joint(name).id
                joint_ids.append(joint_id)
            except KeyError:
                print(f"警告: 找不到关节 {name}")
                # 返回None表示无效关节
                joint_ids.append(None)

        # 打印索引对应关系
        print("右手关节ID和DOF索引对应关系:")
        for i, joint_id in enumerate(joint_ids):
            if joint_id is not None:
                dof_adr = self.model.jnt_dofadr[joint_id] if joint_id < len(self.model.jnt_dofadr) else None
                qpos_adr = self.model.jnt_qposadr[joint_id] if joint_id < len(self.model.jnt_qposadr) else None
                print(f"  关节 {self.hand_joint_names[i]}: ID={joint_id}, DOF索引={dof_adr}, QPOS索引={qpos_adr}")

        return joint_ids

    def set_target_position(self, position: np.ndarray) -> bool:
        """设置末端执行器的目标位置（代理到arm_controller）"""
        return self.arm_controller.set_target_position(position)

    def grasp(self) -> None:
        """握住物体 - 闭合手指"""
        self.hand_cmd_queue.put("grasp")
        self.is_grasping = True
        print("执行握住操作")

    def release(self) -> None:
        """释放物体 - 打开手指"""
        self.hand_cmd_queue.put("release")
        self.is_grasping = False
        print("执行释放操作")

    def toggle_grasp(self) -> None:
        """切换握持状态"""
        if self.is_grasping:
            self.release()
        else:
            self.grasp()

    def apply_hand_joint_limits(self) -> None:
        """应用手部关节角度限制"""
        for i, joint_name in enumerate(self.hand_joint_names):
            name = joint_name.replace('_joint', '')  # 从关节名中提取基础名称
            if name in self.config.joint_limits:
                limits = self.config.joint_limits[name]
                self.hand_target_pos[i] = np.clip(
                    self.hand_target_pos[i], limits[0], limits[1]
                )

    def apply_hand_torque_limits(self, tau: np.ndarray) -> np.ndarray:
        """
        应用手部关节力矩限制

        参数:
            tau: 计算得到的关节力矩
        返回:
            tau: 应用限制后的关节力矩
        """
        for i, joint_name in enumerate(self.hand_joint_names):
            name = joint_name.replace('_joint', '')  # 从关节名中提取基础名称
            if name in self.config.joint_torque_limits and self.hand_joint_ids[i] is not None:
                limit = self.config.joint_torque_limits[name]
                # 获取对应的力矩索引
                tau_idx = self.model.actuator(f"{joint_name}").id
                tau[tau_idx] = np.clip(tau[tau_idx], -limit, limit)
        return tau

    def update(self) -> None:
        """更新控制器状态并计算控制输出"""
        # 首先更新手臂控制器
        self.arm_controller.update()

        # 获取当前手部关节角度
        current_hand_qpos = np.zeros(12)
        for i, joint_id in enumerate(self.hand_joint_ids):
            if joint_id is not None:
                qpos_adr = self.model.jnt_qposadr[joint_id]
                current_hand_qpos[i] = self.data.qpos[qpos_adr]

        # 处理手部命令
        try:
            hand_cmd = self.hand_cmd_queue.get_nowait()
            # 重置过渡状态
            self.interp_factor = 0.0
            self.transitioning = True

            if hand_cmd == "grasp":
                # 目标设置为当前位置到握持位置的渐变
                self.target_end_pos = self.hand_grasp_angles.copy()
            elif hand_cmd == "release":
                # 目标设置为当前位置到默认位置的渐变
                self.target_end_pos = self.config.hand_default_angles.copy()

            # 记录起始位置为当前位置
            self.target_start_pos = current_hand_qpos.copy()

        except queue.Empty:
            pass

        # 处理平滑过渡
        if self.transitioning:
            # 增加插值因子
            self.interp_factor = min(1.0, self.interp_factor + self.transition_speed)

            # 线性插值计算目标位置
            self.hand_target_pos = (1.0 - self.interp_factor) * self.target_start_pos + \
                                   self.interp_factor * self.target_end_pos

            # 检查是否过渡完成
            if self.interp_factor >= 1.0:
                self.transitioning = False

        # 应用关节限位
        self.apply_hand_joint_limits()

    def compute_control(self) -> np.ndarray:
        """计算控制输出"""
        # 首先计算手臂的控制输出
        tau = self.arm_controller.compute_control()

        # 然后更新手部状态
        self.update()

        # 初始化一个全零的手部力矩向量
        hand_tau = np.zeros(len(self.hand_joint_indices), dtype=np.float32)

        # 获取当前手部关节角度和速度
        hand_qpos = np.zeros(len(self.hand_joint_indices), dtype=np.float32)
        hand_qvel = np.zeros(len(self.hand_joint_indices), dtype=np.float32)

        # 获取各个手部关节的位置和速度
        for i, joint_idx in enumerate(self.hand_joint_indices):
            if i < len(self.hand_joint_ids) and self.hand_joint_ids[i] is not None:
                qpos_adr = self.model.jnt_qposadr[self.hand_joint_ids[i]]
                qvel_adr = self.model.jnt_dofadr[self.hand_joint_ids[i]]
                hand_qpos[i] = self.data.qpos[qpos_adr]
                hand_qvel[i] = self.data.qvel[qvel_adr]

        # 使用配置文件中的手部增益参数
        hand_kp = self.config.hand_kps
        hand_kd = self.config.hand_kds

        # 计算手部关节的力矩
        for i in range(len(self.hand_joint_indices)):
            if i < len(hand_kp) and i < len(self.hand_target_pos):
                hand_tau[i] = (self.hand_target_pos[i] - hand_qpos[i]) * hand_kp[i] - hand_qvel[i] * hand_kd[i]

        # 应用手部关节力矩限制
        for i, joint_name in enumerate(self.hand_joint_names):
            if i < len(hand_tau):
                name = joint_name.replace('_joint', '')  # 从关节名中提取基础名称
                if name in self.config.joint_torque_limits:
                    limit = self.config.joint_torque_limits[name]
                    hand_tau[i] = np.clip(hand_tau[i], -limit, limit)

        # 将手部力矩应用到对应关节
        for i, joint_idx in enumerate(self.hand_joint_indices):
            if i < len(hand_tau) and joint_idx < self.model.nu:
                tau[joint_idx] = hand_tau[i]

        return tau

    def start(self) -> None:
        """启动控制器"""
        self.arm_controller.start()
        # 初始化过渡目标
        self.target_start_pos = self.hand_target_pos.copy()
        self.target_end_pos = self.config.hand_default_angles.copy()

    def stop(self) -> None:
        """停止控制器"""
        self.arm_controller.stop()
        # 取消手腕roll关节平滑过渡的定时器
        if hasattr(self, 'wrist_roll_timer') and self.wrist_roll_timer:
            self.wrist_roll_timer.cancel()
            self.wrist_roll_timer = None


class ArmLeftWithHandController:
    """左机械臂带手控制器类"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: ArmConfig):
        """
        初始化机械臂带手控制器
        """
        self.model = model
        self.data = data
        self.config = config

        # 创建左臂控制器作为基础控制器
        self.arm_controller = ArmLeftController(model, data, config)

        # 左手关节索引 (索引22-28)
        self.hand_joint_indices = list(range(22, 34))

        # 设置左手手指握取角度 (调整为更温和的值以减少抖动)
        self.hand_grasp_angles = np.array([1.3, 0.1, 0.8, 0.4, 0.6, 0.9, 0.6, 0.9, 0.6, 0.9, 0.6, 0.9])

        # 自动识别左手关节的索引
        self.hand_joint_names = [
            'L_thumb_proximal_yaw_joint',
            'L_thumb_proximal_pitch_joint',
            'L_thumb_intermediate_joint',
            'L_thumb_distal_joint',
            'L_index_proximal_joint',
            'L_index_intermediate_joint',
            'L_middle_proximal_joint',
            'L_middle_intermediate_joint',
            'L_ring_proximal_joint',
            'L_ring_intermediate_joint',
            'L_pinky_proximal_joint',
            'L_pinky_intermediate_joint'
        ]

        # 获取左手关节在模型中的索引
        self.hand_joint_ids = self._get_hand_joint_ids()

        # 打印手部关节索引信息
        print("左手关节索引:")
        for i, name in enumerate(self.hand_joint_names):
            print(
                f"  {name}: 模型索引 {self.hand_joint_ids[i]}, DOF索引 {self.hand_joint_indices[i] if i < len(self.hand_joint_indices) else None}")

        # 获取当前手部关节角度作为初始位置
        hand_qpos = np.zeros(12)
        for i, joint_id in enumerate(self.hand_joint_ids):
            try:
                if joint_id is not None:
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    hand_qpos[i] = self.data.qpos[qpos_adr]
            except Exception as e:
                print(f"警告: 无法获取手部关节 {self.hand_joint_names[i]} 的位置: {e}")
                hand_qpos[i] = 0.0

        # 初始化左手目标位置为当前位置，避免突然运动
        self.hand_target_pos = hand_qpos.copy()

        # 逐渐过渡到默认状态的插值因子
        self.interp_factor = 0.0
        self.transition_speed = 0.001  # 每次更新增加的插值因子
        self.transitioning = True  # 标记是否正在过渡中

        # 左手状态标记 - True表示握住，False表示松开
        self.is_grasping = False

        # 添加左手指令队列
        self.hand_cmd_queue = queue.Queue()

    def _get_hand_joint_ids(self) -> List[int]:
        """获取手部关节在模型中的索引"""
        joint_ids = []
        for name in self.hand_joint_names:
            try:
                joint_id = self.model.joint(name).id
                joint_ids.append(joint_id)
            except KeyError:
                print(f"警告: 找不到关节 {name}")
                # 返回None表示无效关节
                joint_ids.append(None)

        # 打印索引对应关系
        print("左手关节ID和DOF索引对应关系:")
        for i, joint_id in enumerate(joint_ids):
            if joint_id is not None:
                dof_adr = self.model.jnt_dofadr[joint_id] if joint_id < len(self.model.jnt_dofadr) else None
                qpos_adr = self.model.jnt_qposadr[joint_id] if joint_id < len(self.model.jnt_qposadr) else None
                print(f"  关节 {self.hand_joint_names[i]}: ID={joint_id}, DOF索引={dof_adr}, QPOS索引={qpos_adr}")

        return joint_ids

    def set_target_position(self, position: np.ndarray) -> bool:
        """设置末端执行器的目标位置（代理到arm_controller）"""
        return self.arm_controller.set_target_position(position)

    def grasp(self) -> None:
        """握住物体 - 闭合手指"""
        self.hand_cmd_queue.put("grasp")
        self.is_grasping = True
        print("执行左手握住操作")

    def release(self) -> None:
        """释放物体 - 打开手指"""
        self.hand_cmd_queue.put("release")
        self.is_grasping = False
        print("执行左手释放操作")

    def toggle_grasp(self) -> None:
        """切换握持状态"""
        if self.is_grasping:
            self.release()
        else:
            self.grasp()

    def apply_hand_joint_limits(self) -> None:
        """应用手部关节角度限制"""
        for i, joint_name in enumerate(self.hand_joint_names):
            name = joint_name.replace('_joint', '')  # 从关节名中提取基础名称
            if name in self.config.joint_limits:
                limits = self.config.joint_limits[name]
                self.hand_target_pos[i] = np.clip(
                    self.hand_target_pos[i], limits[0], limits[1]
                )

    def apply_hand_torque_limits(self, tau: np.ndarray) -> np.ndarray:
        """
        应用手部关节力矩限制

        参数:
            tau: 计算得到的关节力矩
        返回:
            tau: 应用限制后的关节力矩
        """
        for i, joint_name in enumerate(self.hand_joint_names):
            name = joint_name.replace('_joint', '')  # 从关节名中提取基础名称
            if name in self.config.joint_torque_limits and self.hand_joint_ids[i] is not None:
                limit = self.config.joint_torque_limits[name]
                # 获取对应的力矩索引
                tau_idx = self.model.actuator(f"{joint_name}").id
                tau[tau_idx] = np.clip(tau[tau_idx], -limit, limit)
        return tau

    def update(self) -> None:
        """更新控制器状态并计算控制输出"""
        # 首先更新手臂控制器
        self.arm_controller.update()

        # 获取当前手部关节角度
        current_hand_qpos = np.zeros(12)
        for i, joint_id in enumerate(self.hand_joint_ids):
            if joint_id is not None:
                qpos_adr = self.model.jnt_qposadr[joint_id]
                current_hand_qpos[i] = self.data.qpos[qpos_adr]

        # 处理手部命令
        try:
            hand_cmd = self.hand_cmd_queue.get_nowait()
            # 重置过渡状态
            self.interp_factor = 0.0
            self.transitioning = True

            if hand_cmd == "grasp":
                # 目标设置为当前位置到握持位置的渐变
                self.target_end_pos = self.hand_grasp_angles.copy()
            elif hand_cmd == "release":
                # 目标设置为当前位置到默认位置的渐变
                self.target_end_pos = self.config.hand_default_angles.copy()

            # 记录起始位置为当前位置
            self.target_start_pos = current_hand_qpos.copy()

        except queue.Empty:
            pass

        # 处理平滑过渡
        if self.transitioning:
            # 增加插值因子
            self.interp_factor = min(1.0, self.interp_factor + self.transition_speed)

            # 线性插值计算目标位置
            self.hand_target_pos = (1.0 - self.interp_factor) * self.target_start_pos + \
                                   self.interp_factor * self.target_end_pos

            # 检查是否过渡完成
            if self.interp_factor >= 1.0:
                self.transitioning = False

        # 应用关节限位
        self.apply_hand_joint_limits()

    def compute_control(self) -> np.ndarray:
        """计算控制输出"""
        # 首先计算手臂的控制输出
        tau = self.arm_controller.compute_control()

        # 然后更新手部状态
        self.update()

        # 初始化一个全零的手部力矩向量
        hand_tau = np.zeros(len(self.hand_joint_indices), dtype=np.float32)

        # 获取当前手部关节角度和速度
        hand_qpos = np.zeros(len(self.hand_joint_indices), dtype=np.float32)
        hand_qvel = np.zeros(len(self.hand_joint_indices), dtype=np.float32)

        # 获取各个手部关节的位置和速度
        for i, joint_idx in enumerate(self.hand_joint_indices):
            if i < len(self.hand_joint_ids) and self.hand_joint_ids[i] is not None:
                qpos_adr = self.model.jnt_qposadr[self.hand_joint_ids[i]]
                qvel_adr = self.model.jnt_dofadr[self.hand_joint_ids[i]]
                hand_qpos[i] = self.data.qpos[qpos_adr]
                hand_qvel[i] = self.data.qvel[qvel_adr]

        # 使用配置文件中的手部增益参数
        hand_kp = self.config.hand_kps
        hand_kd = self.config.hand_kds

        # 计算手部关节的力矩
        for i in range(len(self.hand_joint_indices)):
            if i < len(hand_kp) and i < len(self.hand_target_pos):
                hand_tau[i] = (self.hand_target_pos[i] - hand_qpos[i]) * hand_kp[i] - hand_qvel[i] * hand_kd[i]

        # 应用手部关节力矩限制
        for i, joint_name in enumerate(self.hand_joint_names):
            if i < len(hand_tau):
                name = joint_name.replace('_joint', '')  # 从关节名中提取基础名称
                if name in self.config.joint_torque_limits:
                    limit = self.config.joint_torque_limits[name]
                    hand_tau[i] = np.clip(hand_tau[i], -limit, limit)

        # 将手部力矩应用到对应关节
        for i, joint_idx in enumerate(self.hand_joint_indices):
            if i < len(hand_tau) and joint_idx < self.model.nu:
                tau[joint_idx] = hand_tau[i]

        return tau

    def start(self) -> None:
        """启动控制器"""
        self.arm_controller.start()
        # 初始化过渡目标
        self.target_start_pos = self.hand_target_pos.copy()
        self.target_end_pos = self.config.hand_default_angles.copy()

    def stop(self) -> None:
        """停止控制器"""
        self.arm_controller.stop()
        # 取消手腕roll关节平滑过渡的定时器
        if hasattr(self, 'wrist_roll_timer') and self.wrist_roll_timer:
            self.wrist_roll_timer.cancel()
            self.wrist_roll_timer = None


class SquatController:
    """腿部下蹲和走路控制器"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: ArmConfig):
        """
        初始化下蹲和走路控制器

        参数:
            model: MuJoCo 模型
            data: MuJoCo 数据
            config: 配置参数
        """
        self.model = model
        self.data = data
        self.config = config

        # 腿部控制相关的关节索引（前12个关节是腿部关节）
        self.leg_joint_indices = list(range(12))

        # 初始化RL相关参数
        self.action = np.zeros(12, dtype=np.float32)
        self.target_dof_pos = config.default_angles[:12].copy() if len(config.default_angles) >= 12 else np.zeros(12)

        # 使用配置文件中的cmd_init初始化命令
        if hasattr(config, 'cmd_init'):
            self.cmd = config.cmd_init.copy()
        else:
            self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # 身高控制命令
        if hasattr(config, 'height_cmd'):
            self.height_cmd = config.height_cmd
        else:
            self.height_cmd = 0.80

        self.default_height_cmd = self.height_cmd

        # 最小高度
        if hasattr(config, 'min_height'):
            self.min_height = config.min_height
        else:
            self.min_height = 0.30

        # 下蹲状态控制
        self.is_squatting = False
        self.squat_phase = 0.8  # 0.0-1.0表示下蹲程度
        self.squat_speed = 0.001  # 下蹲速度
        self.squat_direction = 1.0  # 1.0表示下蹲，-1.0表示站起

        # 走路控制
        self.is_walking = False

        # 行走速度参数
        self.walk_speed = 0.1  # 默认值
        self.turn_speed = 1.0  # 默认值

        # 获取蹲/行走关节名称列表
        self.joint_names = [
            # 腿部（共 12 个关节）
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            # 躯干（3 个关节）
            "waist_yaw_joint",
            # 左臂（7 个关节）
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            # 右臂（7 个关节）
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint"
        ]

        # 检查模型的关节数量
        self.n_joints = len(self.joint_names)

        # 观察历史长度
        if hasattr(config, 'obs_history_len'):
            self.obs_history_len = config.obs_history_len
        else:
            self.obs_history_len = 6  # 默认历史长度

        # 加载策略网络
        self.policy = None

        # 控制计数器
        self.counter = 0

        # 对于观察向量的缩放参数
        if hasattr(config, 'dof_pos_scale'):
            self.dof_pos_scale = config.dof_pos_scale
        else:
            self.dof_pos_scale = 1.0

        if hasattr(config, 'dof_vel_scale'):
            self.dof_vel_scale = config.dof_vel_scale
        else:
            self.dof_vel_scale = 0.05

        if hasattr(config, 'ang_vel_scale'):
            self.ang_vel_scale = config.ang_vel_scale
        else:
            self.ang_vel_scale = 0.25

        # 使用config中的cmd_scale
        if hasattr(config, 'cmd_scale') and config.cmd_scale is not None:
            self.cmd_scale = config.cmd_scale
        else:
            self.cmd_scale = np.array([2.0, 2.0, 0.25])

        if hasattr(config, 'action_scale'):
            self.action_scale = config.action_scale
        else:
            self.action_scale = 0.25

        # 尝试加载策略网络
        policy_path = config.policy_path.replace("{LEGGED_GYM_ROOT_DIR}", dir_root)
        try:
            self.policy = torch.jit.load(policy_path)
            print(f"成功加载策略网络: {policy_path}")

            # 初始化观察向量
            single_obs, self.single_obs_dim = self._compute_observation()
            self.obs_history = collections.deque(maxlen=self.obs_history_len)
            for _ in range(self.obs_history_len):
                self.obs_history.append(np.zeros_like(single_obs))

            # 准备完整观察向量
            self.full_obs = np.zeros(self.single_obs_dim * self.obs_history_len, dtype=np.float32)

        except Exception as e:
            print(f"无法加载策略网络: {e}")
            raise RuntimeError(f"策略网络加载失败: {e}")

    # 添加走路控制功能
    def toggle_walking(self):
        """切换走路状态"""
        global switch
        self.is_walking = not self.is_walking
        if self.is_walking:
            print("走路模式已激活")
            # 确保下蹲状态关闭
            self.is_squatting = False
            self.height_cmd = self.default_height_cmd
            switch = True
        else:
            print("走路模式已停止")
            # 重置命令
            self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            switch = False

    def update_walking_cmd(self):
        """更新走路命令"""
        global switch, xy, cmd
        # 检查全局走路开关
        if not self.is_walking or not switch:
            return

        # 获取机器人当前位置和朝向
        try:
            pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
            if pelvis_id != -1:
                # 获取机器人位置和方向
                current_pos = self.data.xpos[pelvis_id]
                current_quat = self.data.xquat[pelvis_id]

                # 计算行走命令
                new_cmd = decide_movement(
                    current_pos[0], current_pos[1],
                    xy[0], xy[1],
                    current_quat[0], current_quat[1], current_quat[2], current_quat[3]
                )
                if new_cmd:  # 确保命令不是None
                    self.cmd = np.array(new_cmd, dtype=np.float32)
                    cmd = new_cmd  # 更新全局命令

                # 计算到目标的距离
                distance = math.sqrt((xy[0] - current_pos[0]) ** 2 + (xy[1] - current_pos[1]) ** 2)
                # 可选的进度打印，每10秒一次
                if time.time() % 10 < self.model.opt.timestep:
                    print(
                        f"当前位置: ({current_pos[0]:.2f}, {current_pos[1]:.2f}), 目标: ({xy[0]:.2f}, {xy[1]:.2f}), 距离: {distance:.2f}")
            else:
                print("警告: 找不到pelvis节点")
        except Exception as e:
            print(f"更新行走命令出错: {e}")

    def _compute_observation(self):
        """计算当前状态的观察向量"""
        # 获取关节状态
        qj = np.zeros(self.n_joints, dtype=np.float32)
        dqj = np.zeros(self.n_joints, dtype=np.float32)

        for i, key in enumerate(self.joint_names):
            qj[i] = self.data.joint(key).qpos
            dqj[i] = self.data.joint(key).qvel

        # 获取姿态信息
        quat = self.data.qpos[3:7].copy()
        omega = self.data.qvel[3:6].copy()

        # 处理默认关节角度
        if len(self.config.default_angles) < self.n_joints:
            padded_defaults = np.zeros(self.n_joints, dtype=np.float32)
            padded_defaults[:len(self.config.default_angles)] = self.config.default_angles
        else:
            padded_defaults = self.config.default_angles[:self.n_joints]

        # 缩放值
        qj_scaled = (qj - padded_defaults) * self.dof_pos_scale
        dqj_scaled = dqj * self.dof_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        omega_scaled = omega * self.ang_vel_scale

        # 计算单个观察向量维度
        single_obs_dim = 3 + 1 + 3 + 3 + self.n_joints + self.n_joints + 12

        # 创建单个观察向量
        single_obs = np.zeros(single_obs_dim, dtype=np.float32)

        # 填充观察向量
        single_obs[0:3] = self.cmd * self.cmd_scale
        single_obs[3:4] = np.array([self.height_cmd])
        single_obs[4:7] = omega_scaled
        single_obs[7:10] = gravity_orientation
        single_obs[10:10 + self.n_joints] = qj_scaled
        single_obs[10 + self.n_joints:10 + 2 * self.n_joints] = dqj_scaled
        single_obs[10 + 2 * self.n_joints:10 + 2 * self.n_joints + 12] = self.action

        return single_obs, single_obs_dim

    def toggle_squat(self):
        """切换下蹲状态"""
        global switch
        self.is_squatting = not self.is_squatting

        # 如果激活下蹲，确保走路模式关闭
        if self.is_squatting:
            # 只关闭走路，不影响其他功能
            self.is_walking = False
            self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            # 只在走路模式启用时改变全局开关
            if switch:
                print("下蹲激活时关闭走路全局开关")
                switch = False

            self.squat_phase = 0.8
            self.squat_direction = 1.0
        else:
            self.squat_direction = -1.0

            # 恢复默认高度
            self.height_cmd = self.default_height_cmd
            print(f"下蹲模式已停止, 恢复高度: {self.default_height_cmd:.2f}")

    def update(self):
        """更新控制器状态"""
        # 更新步行命令
        if self.is_walking:
            self.update_walking_cmd()

        # 不在下蹲或行走状态，恢复默认姿势
        if not self.is_squatting and not self.is_walking:
            self.target_dof_pos = self.config.default_angles[:12].copy() if len(
                self.config.default_angles) >= 12 else np.zeros(12)
            return

        # 每隔control_decimation步更新一次策略
        if self.counter % self.config.control_decimation == 0 and self.policy is not None:
            try:
                # 计算当前观察
                single_obs, _ = self._compute_observation()
                self.obs_history.append(single_obs)

                # 构建完整的观察历史
                for i, hist_obs in enumerate(self.obs_history):
                    start_idx = i * self.single_obs_dim
                    end_idx = start_idx + self.single_obs_dim
                    if end_idx <= len(self.full_obs):
                        self.full_obs[start_idx:end_idx] = hist_obs

                # 策略推理
                obs_tensor = torch.from_numpy(self.full_obs).unsqueeze(0)
                self.action = self.policy(obs_tensor).detach().numpy().squeeze()

                # 变换动作到目标关节角度
                self.target_dof_pos = self.action * self.action_scale + self.config.default_angles[:12]

            except Exception as e:
                print(f"下蹲/走路执行错误: {e}")

        self.counter += 1

    def compute_control(self):
        """计算控制输出"""
        # 更新控制器状态
        self.update()

        # 初始化控制力矩
        tau = np.zeros(self.model.nu, dtype=np.float32)

        # 获取当前腿部关节位置和速度
        current_pos = np.zeros(len(self.leg_joint_indices), dtype=np.float32)
        current_vel = np.zeros(len(self.leg_joint_indices), dtype=np.float32)

        foot_names = self.joint_names[:12]
        for i, key in enumerate(foot_names):
            current_pos[i] = self.data.joint(key).qpos
            current_vel[i] = self.data.joint(key).qvel

        # 获取PD控制增益
        leg_kp = self.config.kps[:12] if len(self.config.kps) >= 12 else np.ones(12) * 100
        leg_kd = self.config.kds[:12] if len(self.config.kds) >= 12 else np.ones(12) * 2

        # 计算腿部关节力矩
        leg_tau = pd_control(
            self.target_dof_pos,
            current_pos,
            leg_kp,
            np.zeros_like(leg_kp),
            current_vel,
            leg_kd
        )

        # # 应用腿部关节力矩
        for i, joint_idx in enumerate(self.leg_joint_indices):
            tau[joint_idx] = leg_tau[i]

        return tau

    def start(self):
        """启动控制器"""
        print("下蹲和走路控制器已启动")

    def stop(self):
        """停止控制器"""
        print("下蹲和走路控制器已停止")


def get_gravity_orientation(quat):
    """Get gravity vector in body frame"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q"""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    q_conj = np.array([w, -x, -y, -z])

    return np.array([
        v[0] * (q_conj[0] ** 2 + q_conj[1] ** 2 - q_conj[2] ** 2 - q_conj[3] ** 2) +
        v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3]) +
        v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),

        v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3]) +
        v[1] * (q_conj[0] ** 2 - q_conj[1] ** 2 + q_conj[2] ** 2 - q_conj[3] ** 2) +
        v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),

        v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2]) +
        v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1]) +
        v[2] * (q_conj[0] ** 2 - q_conj[1] ** 2 - q_conj[2] ** 2 + q_conj[3] ** 2)
    ])


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """计算PD控制力矩"""
    return (target_q - q) * kp + (target_dq - dq) * kd


# 从 mujoco_deploy_g1.py 添加的方向计算函数
def quaternion_to_direction(w, x, y, z):
    """
    将四元数转换为二维平面的方向向量 (dx, dy)。
    假设四元数表示的是绕 Z 轴的旋转。
    """
    # 提取旋转角度
    angle = math.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    # 计算方向向量
    dx = math.cos(angle)
    dy = math.sin(angle)
    return dx, dy


def calculate_angle_diff(v1, v2):
    """
    计算两个向量之间的夹角（以弧度为单位）。
    """
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    cos_theta = max(-1, min(1, cos_theta))  # 防止数值误差导致超出范围
    angle = math.acos(cos_theta)
    # 确定方向（顺时针或逆时针）
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    if cross_product < 0:
        angle = -angle
    return angle


def decide_movement(x_start, y_start, x_end, y_end, w, x, y, z):
    """
    判断是否需要前进或转弯。
    如果距离小于等于 0.1，则停止移动。
    """
    # 计算起点和终点的距离
    distance = math.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
    if distance <= 0.1:
        global switch
        switch = False
        print(f"已到达目标点附近({distance:.3f}), 停止移动")
        return 0, 0, 0

    # 计算目标方向向量
    target_dx = x_end - x_start
    target_dy = y_end - y_start
    target_magnitude = math.sqrt(target_dx ** 2 + target_dy ** 2)
    target_dx /= target_magnitude
    target_dy /= target_magnitude

    # 获取当前方向向量
    current_dx, current_dy = quaternion_to_direction(w, x, y, z)

    # 计算角度差
    angle_diff = calculate_angle_diff(
        (current_dx, current_dy), (target_dx, target_dy))

    # 判断是否需要转弯
    angle_threshold = math.radians(5)  # 5度的阈值
    forward_speed = min(0.5, distance * 0.3)  # 根据距离动态调整前进速度，并限制最大速度
    turn_speed = 1.0  # 转弯速度

    cmd_x, cmd_y, cmd_z = 0, 0, 0
    if abs(angle_diff) <= angle_threshold:
        cmd_x = forward_speed
    else:
        # 如果角度差很大，只转弯不前进
        if angle_diff > 0:
            cmd_z = turn_speed
        else:
            cmd_z = -turn_speed

    return cmd_x, cmd_y, cmd_z


class InputThread(threading.Thread):
    """用户输入线程类"""

    def __init__(self, left_controller, right_controller, squat_controller):
        """
        初始化输入线程

        参数:
            left_controller: 左臂控制器
            right_controller: 右臂控制器
            squat_controller: 下蹲和走路控制器
        """
        super().__init__()
        self.left_controller = left_controller
        self.right_controller = right_controller
        self.squat_controller = squat_controller
        self.daemon = True
        self.input_queue = queue.Queue()

    def run(self) -> None:
        """线程主函数"""
        global xy, switch

        print("请输入命令:")
        print("- 输入 'left_pos x y z' 设置左臂目标位置, 例如: left_pos 1.2 -1.5 1.0")
        print("- 输入 'right_pos x y z' 设置右臂目标位置, 例如: right_pos 1.2 -1.5 1.0")
        print("- 输入 'left_toggle' 切换左手状态")
        print("- 输入 'right_toggle' 切换右手状态")
        print("- 输入 'left_wrist_roll angle' 设置左手腕roll角度, 例如: left_wrist_roll 1.57")
        print("- 输入 'right_wrist_roll angle' 设置右手腕roll角度, 例如: right_wrist_roll 1.57")
        print("- 输入 'squat' 开始/停止下蹲")
        print("- 输入 'walk' 开始/停止走路模式")
        print("- 输入 'goto x y' 设置行走目标位置, 例如: goto 2.0 3.0")
        print("- 输入 'help' 显示帮助信息")

        while True:
            user_input = input("> ")
            parts = user_input.split()

            if not parts:
                continue

            cmd = parts[0].lower()

            # 特殊命令的参数处理
            try:
                if cmd == "left_pos" and len(parts) >= 4 and self.left_controller:
                    positions = list(map(float, parts[1:4]))
                    target_pos = np.array(positions)
                    if self.left_controller.set_target_position(target_pos):
                        print(f"已设置左臂目标位置为: {target_pos}，使用RRT进行路径规划")
                    else:
                        print("目标位置超出工作空间范围")

                elif cmd == "right_pos" and len(parts) >= 4 and self.right_controller:
                    positions = list(map(float, parts[1:4]))
                    target_pos = np.array(positions)
                    if self.right_controller.set_target_position(target_pos):
                        print(f"已设置右臂目标位置为: {target_pos}，使用RRT进行路径规划")
                    else:
                        print("目标位置超出工作空间范围")

                elif cmd == "left_wrist_roll" and len(parts) >= 2 and self.left_controller:
                    angle = float(parts[1])
                    self.left_controller.arm_controller.set_wrist_roll_angle(angle)

                elif cmd == "right_wrist_roll" and len(parts) >= 2 and self.right_controller:
                    angle = float(parts[1])
                    self.right_controller.arm_controller.set_wrist_roll_angle(angle)

                elif cmd == "left_toggle":
                    self.left_controller.toggle_grasp()
                    print("左手切换命令已接收")

                elif cmd == "right_toggle":
                    self.right_controller.toggle_grasp()
                    print("右手切换命令已接收")

                elif cmd == "squat":
                    print("下蹲命令已接收")
                    # 所有有效命令都放入队列
                    self.input_queue.put(cmd)

                elif cmd == "walk":
                    print("走路命令已接收")
                    # 所有有效命令都放入队列
                    self.input_queue.put(cmd)

                elif cmd == "goto" and len(parts) >= 3:
                    old_xy = xy.copy() if 'xy' in globals() and isinstance(xy, list) and len(xy) == 2 else None
                    xy = [float(parts[1]), float(parts[2])]
                    switch = True
                    if old_xy:
                        # 计算与之前目标的距离
                        dist = math.sqrt((xy[0] - old_xy[0]) ** 2 + (xy[1] - old_xy[1]) ** 2)
                        print(f"已设置新目标位置: ({xy[0]}, {xy[1]}), 与上一目标相距: {dist:.2f}")
                    else:
                        print(f"已设置目标位置: ({xy[0]}, {xy[1]})")

                    # 所有有效命令都放入队列
                    self.input_queue.put(cmd)

                elif cmd == "help":
                    print("有效命令包括:")
                    print("- left_pos x y z (设置左臂位置)")
                    print("- right_pos x y z (设置右臂位置)")
                    print("- left_toggle (切换左手握持状态)")
                    print("- right_toggle (切换右手握持状态)")
                    print("- left_wrist_roll angle (设置左手腕roll角度, 范围约[-1.97, 1.97])")
                    print("- right_wrist_roll angle (设置右手腕roll角度, 范围约[-1.97, 1.97])")
                    print("- squat (开始/停止下蹲)")
                    print("- walk (开始/停止走路模式)")
                    print("- goto x y (设置行走目标位置)")
                    print("- help (显示帮助)")

                else:
                    print("无效命令！输入 help 查看有效命令。")
            except ValueError:
                # 命令或参数解析出错时的处理
                if cmd == "left_pos" or cmd == "right_pos":
                    print("请输入有效的数字坐标")
                elif cmd == "left_wrist_roll" or cmd == "right_wrist_roll":
                    print("请输入有效的角度数值")
                elif cmd == "goto":
                    print("请输入有效的数字坐标")
                else:
                    print("命令参数格式错误，请检查输入")

    def get_input(self):
        """获取最新的用户输入命令，非阻塞"""
        try:
            cmd = self.input_queue.get_nowait()
            return cmd
        except queue.Empty:
            return None


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()

    # 加载配置文件
    config_path = f"{dir_root}/deploy/config/{args.config_file}"
    config = ArmConfig.from_yaml(config_path)

    # 打印工作空间限制
    print(f"工作空间限制:")
    print(f"  X轴: {config.workspace_limits['x']}")
    print(f"  Y轴: {config.workspace_limits['y']}")
    print(f"  Z轴: {config.workspace_limits['z']}")

    with open(config_path, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = yaml_config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", dir_root)

    # 加载机器人模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = config.simulation_dt
    model.opt.viscosity = np.random.uniform(0.1, 0.5)
    model.opt.density = np.random.uniform(800, 1200)
    # 打印关节信息以帮助调试
    n_joints = data.qpos.shape[0] - 7  # 减去基座的7个自由度
    print(f"\n模型关节信息:")
    print(f"  总关节数: {n_joints}")
    print(f"  控制器数量: {model.nu}")
    print(f"  默认关节角度数组长度: {len(config.default_angles)}")

    # 打印关节名称和索引
    print("\n关节名称和索引:")
    for i in range(model.njnt):
        name = model.joint(i).name
        qpos_start = model.jnt_qposadr[i]
        print(f"  关节 {i}: {name}, qpos索引: {qpos_start}")

    # 创建机械臂控制器 - 左右臂带手控制器
    right_controller = ArmRightWithHandController(model, data, config)
    left_controller = ArmLeftWithHandController(model, data, config)

    # 创建下蹲控制器 - 始终启用
    squat_controller = SquatController(model, data, config)

    # 进行一次前向动力学计算以更新机器人状态
    mujoco.mj_forward(model, data)

    # 打印末端执行器的初始位置
    right_end_effector_id = model.body('right_wrist_yaw_link').id
    right_initial_pos = data.xpos[right_end_effector_id]
    print(f"\n右臂末端执行器初始位置: {right_initial_pos.round(4)}")
    print(f"此位置是否在工作空间内: {right_controller.arm_controller._check_position_limits(right_initial_pos)}")

    left_end_effector_id = model.body('left_wrist_yaw_link').id
    left_initial_pos = data.xpos[left_end_effector_id]
    print(f"\n左臂末端执行器初始位置: {left_initial_pos.round(4)}")
    print(f"此位置是否在工作空间内: {left_controller.arm_controller._check_position_limits(left_initial_pos)}")

    # 启动用户输入线程
    input_thread = InputThread(left_controller, right_controller, squat_controller)
    input_thread.start()

    # walk_thread = threading.Thread(target=walker_thread)
    # walk_thread.daemon = True
    # walk_thread.start()

    # 启动所有控制器
    right_controller.start()
    left_controller.start()
    squat_controller.start()

    # 自动开始下蹲
    squat_controller.toggle_squat()

    # 启动可视化界面
    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running():  # 只要查看器在运行就继续执行
                step_start = time.time()

                # 计算控制输出
                tau = np.zeros(model.nu)  # 初始化力矩数组

                # 计算右臂控制输出 (包括右臂和右手)
                right_tau = right_controller.compute_control()

                # 计算左臂控制输出 (包括左臂和左手)
                left_tau = left_controller.compute_control()

                # 计算下蹲控制输出
                squat_tau = squat_controller.compute_control()

                # 创建腰部关节控制力矩 (默认保持在0位置)
                waist_indices = [12, 13, 14]  # 腰部关节索引
                waist_joint_names = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]  # 腰部关节名称

                # 应用所有控制力矩
                for i in range(model.nu):
                    if i >= 15 and i < 34:  # 左臂和左手关节 (15-33)
                        tau[i] = left_tau[i]
                    elif i >= 34 and i < 53:  # 右臂和右手关节 (34-53)
                        tau[i] = right_tau[i]
                    elif i in waist_indices:  # 腰部关节 (12-14)
                        # 对腰部关节应用简单的PD控制，保持在默认位置
                        joint_name = waist_joint_names[i - 12]  # 获取对应的关节名称
                        current_pos = data.joint(joint_name).qpos
                        current_vel = data.joint(joint_name).qvel

                        # 从配置文件获取目标位置
                        target_pos = config.default_angles[i] if i < len(config.default_angles) else 0.0

                        # 从配置文件获取控制增益
                        waist_kp = config.kps[i] if i < len(config.kps) else 300.0
                        waist_kd = config.kds[i] if i < len(config.kds) else 5.0

                        # 计算PD控制力矩
                        waist_tau = (target_pos - current_pos) * waist_kp - current_vel * waist_kd
                        tau[i] = waist_tau
                    elif i < 12:  # 腿部关节 (0-11)
                        # 使用下蹲控制器的力矩
                        tau[i] = squat_tau[i]

                # 应用控制力矩到模型
                data.ctrl[:] = tau
                # 物理仿真步进
                mujoco.mj_step(model, data)

                # 获取用户输入，处理命令
                user_input = input_thread.get_input()
                if user_input:  # 确保有输入才处理
                    if user_input == "squat":
                        squat_controller.toggle_squat()

                    elif user_input == "walk":
                        squat_controller.toggle_walking()

                    elif user_input == "goto":
                        # 目标位置已经在输入线程中设置
                        if not squat_controller.is_walking:
                            squat_controller.toggle_walking()  # 确保走路模式开启
                        else:
                            squat_controller.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                # 显示位置信息
                if time.time() % 5 < config.simulation_dt:
                    # 显示右臂信息
                    current_pos = data.xpos[right_controller.arm_controller.right_end_effector_id]
                    print(f"\n右臂当前位置: {current_pos.round(4)}")
                    if right_controller.is_grasping:
                        print("右手状态: 握住")
                    else:
                        print("右手状态: 松开")
                    if right_controller.arm_controller.has_user_input:
                        print(f"右臂目标位置: {right_controller.arm_controller.last_target_pos.round(4)}")
                        pos_error = right_controller.arm_controller.last_target_pos - current_pos
                        print(f"右臂位置误差: {np.linalg.norm(pos_error):.4f}")

                    # 显示左臂信息
                    current_pos = data.xpos[left_controller.arm_controller.left_end_effector_id]
                    print(f"\n左臂当前位置: {current_pos.round(4)}")
                    if left_controller.is_grasping:
                        print("左手状态: 握住")
                    else:
                        print("左手状态: 松开")
                    if left_controller.arm_controller.has_user_input:
                        print(f"左臂目标位置: {left_controller.arm_controller.last_target_pos.round(4)}")
                        pos_error = left_controller.arm_controller.last_target_pos - current_pos
                        print(f"左臂位置误差: {np.linalg.norm(pos_error):.4f}")

                    # 显示下蹲信息
                    if squat_controller.is_squatting:
                        print(f"\n下蹲状态: {squat_controller.squat_phase:.2f}")

                # 更新可视化 - 使用viewer.user_scn添加自定义geom以显示路径和障碍物
                with viewer.lock():
                    # 在每帧清除user_scn中的自定义几何体
                    viewer.user_scn.ngeom = 0

                    # 添加障碍物可视化
                    for i, obs in enumerate(obstacles):
                        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                            # 设置为长方体
                            mujoco.mjv_initGeom(
                                g,
                                mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([obs[3]/2, obs[4]/2, obs[5]/2]),  # 使用障碍物的长宽高的一半作为尺寸
                                np.array([obs[0], obs[1], obs[2]]),  # 障碍物位置
                                np.eye(3).flatten(),
                                np.array([1.0, 0, 0, 0.5])
                            )
                            viewer.user_scn.ngeom += 1

                    # 添加路径可视化到场景
                    if right_controller.arm_controller.sites_added:
                        right_controller.arm_controller.add_path_sites_to_scene(viewer.user_scn)

                    if left_controller.arm_controller.sites_added:
                        left_controller.arm_controller.add_path_sites_to_scene(viewer.user_scn)

                # 更新可视化
                viewer.sync()

                # 控制仿真步进频率
                time_until_next_step = config.simulation_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        except KeyboardInterrupt:
            print("程序已终止")
        finally:
            print("程序结束")
            right_controller.stop()
            left_controller.stop()
            squat_controller.stop()


if __name__ == "__main__":
    main()

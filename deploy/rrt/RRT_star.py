import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# 定义三维节点类
class Node3D:
    def __init__(self, x, y, z):
        self.x = x  # x坐标
        self.y = y  # y坐标
        self.z = z  # z坐标
        self.parent = None  # 父节点
        self.cost = 0.0  # 路径成本


# 三维RRT*算法
class RRTStar3D:
    def __init__(self, start, goal, obstacles,
                 bounds, step_size=1.5,
                 max_iter=1000, search_radius=3.0):
        """
        参数说明:
        start: 起点 (x, y, z)
        goal: 终点 (x, y, z)
        obstacles: 障碍物列表 (每个障碍物格式: (x, y, z, length, width, height))
        bounds: 三维空间边界 (x_min, x_max, y_min, y_max, z_min, z_max)
        """
        self.start = Node3D(*start)
        self.goal = Node3D(*goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.nodes = [self.start]
        self.path = []

    def plan(self):
        for _ in range(self.max_iter):
            # 1. 随机采样（5%概率采样目标点）
            rand_node = self.get_random_node()

            # 2. 找到最近节点
            nearest_node = self.find_nearest(rand_node)

            # 3. 扩展新节点
            new_node = self.steer(nearest_node, rand_node)

            # 4. 碰撞检测
            if self.check_collision(nearest_node, new_node):
                # 5. 寻找邻近节点
                near_nodes = self.find_near_nodes(new_node)

                # 6. 选择最优父节点
                self.choose_parent(new_node, near_nodes)

                # 7. 添加到树中
                self.nodes.append(new_node)

                # 8. 重新布线优化
                self.rewire(new_node, near_nodes)

                # 检查是否到达目标
                if self.calc_distance(new_node, self.goal) < self.step_size:
                    final_node = self.steer(new_node, self.goal)
                    if self.check_collision(new_node, final_node):
                        self.goal.parent = new_node
                        self.goal.cost = new_node.cost + self.step_size
                        self.nodes.append(self.goal)
                        self.path = self.generate_path()
                        return self.path
        return None

    def get_random_node(self):
        if random.random() < 0.05:  # 5%概率偏向目标点
            return self.goal
        x = random.uniform(self.bounds[0], self.bounds[1])
        y = random.uniform(self.bounds[2], self.bounds[3])
        z = random.uniform(self.bounds[4], self.bounds[5])
        return Node3D(x, y, z)

    def find_nearest(self, rand_node):
        return min(self.nodes, key=lambda n: self.calc_distance(n, rand_node))

    def steer(self, from_node, to_node):
        # 计算三维方向向量
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # 按步长扩展
        scale = self.step_size / dist
        new_node = Node3D(
            from_node.x + dx * scale,
            from_node.y + dy * scale,
            from_node.z + dz * scale
        )
        new_node.parent = from_node
        new_node.cost = from_node.cost + self.step_size
        return new_node

    def check_collision(self, n1, n2):
        # 检测线段n1-n2与矩形障碍物的碰撞
        for (ox, oy, oz, length, width, height) in self.obstacles:
            # 计算矩形的边界
            x_min, x_max = ox - length/2, ox + length/2
            y_min, y_max = oy - width/2, oy + width/2
            z_min, z_max = oz - height/2, oz + height/2
            
            # 线段参数方程: p(t) = p1 + t*(p2-p1), t ∈ [0,1]
            start = np.array([n1.x, n1.y, n1.z])
            end = np.array([n2.x, n2.y, n2.z])
            direction = end - start
            
            # 计算线段与矩形各面的交点参数t
            t_values = []
            
            # x = x_min 和 x = x_max 平面
            if direction[0] != 0:
                t1 = (x_min - start[0]) / direction[0]
                t2 = (x_max - start[0]) / direction[0]
                t_values.extend([t1, t2])
            
            # y = y_min 和 y = y_max 平面
            if direction[1] != 0:
                t3 = (y_min - start[1]) / direction[1]
                t4 = (y_max - start[1]) / direction[1]
                t_values.extend([t3, t4])
            
            # z = z_min 和 z = z_max 平面
            if direction[2] != 0:
                t5 = (z_min - start[2]) / direction[2]
                t6 = (z_max - start[2]) / direction[2]
                t_values.extend([t5, t6])
            
            # 检查所有有效的交点参数 (0 <= t <= 1)
            for t in t_values:
                if 0 <= t <= 1:
                    # 计算交点坐标
                    intersection = start + t * direction
                    x, y, z = intersection
                    
                    # 检查交点是否在矩形内
                    if (x_min <= x <= x_max and 
                        y_min <= y <= y_max and 
                        z_min <= z <= z_max):
                        return False  # 碰撞
            
            # 检查线段是否完全在矩形内部
            if (x_min <= n1.x <= x_max and y_min <= n1.y <= y_max and z_min <= n1.z <= z_max):
                return False
            if (x_min <= n2.x <= x_max and y_min <= n2.y <= y_max and z_min <= n2.z <= z_max):
                return False
                
        return True  # 无碰撞

    def find_near_nodes(self, new_node):
        return [n for n in self.nodes
                if self.calc_distance(n, new_node) <= self.search_radius]

    def choose_parent(self, new_node, near_nodes):
        for node in near_nodes:
            if node.cost + self.calc_distance(node, new_node) < new_node.cost:
                if self.check_collision(node, new_node):
                    new_node.parent = node
                    new_node.cost = node.cost + self.calc_distance(node, new_node)

    def rewire(self, new_node, near_nodes):
        for node in near_nodes:
            if new_node.cost + self.calc_distance(new_node, node) < node.cost:
                if self.check_collision(new_node, node):
                    node.parent = new_node
                    node.cost = new_node.cost + self.calc_distance(new_node, node)

    def generate_path(self):
        path = []
        node = self.goal
        while node.parent:
            path.append((node.x, node.y, node.z))
            node = node.parent
        path.append((self.start.x, self.start.y, self.start.z))
        return path[::-1]

    @staticmethod
    def calc_distance(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


# 示例使用
if __name__ == "__main__":
    # 定义障碍物 (x, y, z, length, width, height)
    obstacles = [
        (5, 5, 5, 3, 4, 2),
        (2, 8, 3, 2, 2, 3),
        (8, 2, 7, 1, 3, 2)
    ]
    
    # 创建RRT*对象
    rrt = RRTStar3D(
        start=(0, 0, 0),
        goal=(12, 12, 12),
        obstacles=obstacles,
        bounds=(0, 12, 0, 12, 0, 12),
        step_size=1,
        search_radius=2.5
    )

    # 执行路径规划
    path = rrt.plan()

    # 可视化
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制障碍物矩形
    for (x, y, z, length, width, height) in obstacles:
        # 创建立方体的8个顶点
        x_min, x_max = x - length/2, x + length/2
        y_min, y_max = y - width/2, y + width/2
        z_min, z_max = z - height/2, z + height/2
        
        # 定义立方体的8个顶点
        vertices = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ]
        
        # 定义立方体的6个面，每个面由4个顶点索引组成
        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [2, 3, 7, 6],
            [0, 3, 7, 4], [1, 2, 6, 5]
        ]
        
        # 绘制每个面
        for face in faces:
            # 提取每个面的四个顶点坐标
            face_vertices = [vertices[i] for i in face]
            # 添加第一个顶点构成闭合路径
            face_vertices.append(face_vertices[0])
            
            # 分离x、y、z坐标
            xs = [v[0] for v in face_vertices]
            ys = [v[1] for v in face_vertices]
            zs = [v[2] for v in face_vertices]
            
            # 绘制面的边缘
            ax.plot(xs, ys, zs, 'k-', alpha=0.3)
            
            # 使用fill3d来填充面
            poly = Poly3DCollection([list(zip(xs[:4], ys[:4], zs[:4]))], alpha=0.3)
            poly.set_color('gray')
            ax.add_collection3d(poly)

    # 绘制树结构
    for node in rrt.nodes:
        if node.parent:
            ax.plot([node.x, node.parent.x],
                    [node.y, node.parent.y],
                    [node.z, node.parent.z],
                    'g-', lw=0.5, alpha=0.5)

    # 绘制最终路径
    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-', lw=2, label='Optimal Path')

    ax.scatter(rrt.start.x, rrt.start.y, rrt.start.z, c='blue', s=100, label='Start')
    ax.scatter(rrt.goal.x, rrt.goal.y, rrt.goal.z, c='orange', s=100, label='Goal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

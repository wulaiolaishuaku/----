
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from scipy.interpolate import BSpline
from math import atan2
from std_msgs.msg import UInt8
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

class DWA:
    def __init__(self):
        # 设定机器人参量
        self.max_speed = 0.8  # 最大线速度
        self.max_yaw_rate = 1  # 最大角速度
        self.dt = 0.1  # 时间间隔
        self.wheel_base = 0.5  # 机器人轮距
        self.robot_radius = 0.6  # 机器人半径
        self.linear_velocity = 0
        self.angular_velocity = 0
        self.angle = 0
        # 初始化障碍物列表
        self.obstacles = []
        self.linear_pid = PIDController(kp=5, ki=0.1, kd=0.5)  # 调整线速度的 PID 参数
        self.angular_pid = PIDController(kp=5.0, ki=0.05, kd=0.5)  # 调整角速度的 PID 参数
        self.tag_sta = 0
        self.count = 0
        self.change_count = 0
        self.last_data = 0
        self.toggle_state = False
        # ROS 的发布器
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)  
        rospy.Subscriber('/base_station_position', Point, self.relative_position_callback)
        rospy.Subscriber('/tag_status', UInt8, self.tag_status)
        self.target = np.array([0.0, 0.0])  # 目标点的相对位置
        
    def lidar_callback(self, msg):
        self.detect_obstacles(msg.ranges)
        
    def relative_position_callback(self, msg):
        # 通过geometry_msgs/Point获取目标的相对位置
        self.target = np.array([msg.y, msg.x])  # 提取目标的x和y坐标
        if self.target[0] !=0 or self.target[1] !=0:
            self.angle = atan2(self.target[1], self.target[0])  # 计算目标的角度

    def tag_status(self,msg):
        # 获取当前收到的数据
        current_data = msg.data
        # 检查数据是否从0变化为128
        if self.last_data == 0 and current_data == 128:
            self.change_count += 1  # 增加变化计数

        # 打印计数信息
        rospy.loginfo("Change Count: %d", self.change_count)

        # 更新last_data为当前值
        self.last_data = current_data


    def detect_obstacles(self, ranges):
        # 激光雷达数据预处理，检测障碍物
        self.obstacles = []  # 清空障碍物列表
        for i, distance in enumerate(ranges):
            if distance < 1.2:  # 设定一个阈值，表示障碍物距离
                angle = i * 360 / len(ranges)  # 激光雷达的角度
                x = distance * np.cos(np.radians(angle))
                y = distance * np.sin(np.radians(angle))
                self.obstacles.append(np.array([x, y]))

    def calculate_velocity(self):
        # 计算DWA速度指令，考虑目标点和障碍物
        v = np.linspace(0, self.max_speed, num=10)  # 线速度从0到最大，生成一个从0到最大线速度的线性空间，分成10个值，表示机器人可以选择的不同线速度。
        omega = np.linspace(-self.max_yaw_rate, self.max_yaw_rate, num=10)  # 角速度从最小到最大
        best_v = 0
        best_omega = 0
        max_score = -float('inf')
        best_trajectory = None
        # 遍历所有可能的线速度和角速度
        for vi in v:
            for oi in omega:
                trajectory = self.predict_trajectory(vi, oi) # 预测该轨迹下未来轨迹
                score = self.score_trajectory(trajectory)  # 评估该轨迹的得分
                if score > max_score:
                    max_score = score
                    best_v = vi
                    best_omega = oi
                    best_trajectory = trajectory  # 更新最佳轨迹
        return best_v, best_omega,best_trajectory

    def predict_trajectory(self, v, omega):
        # 预测未来轨迹
        trajectory = []
        position = np.array([0.0, 0.0])  # 机器人在世界坐标系中的位置
        yaw = 0.0  # 机器人的朝向角
        for t in range(int(1 / self.dt)):
            position[0] += v * np.cos(yaw) * self.dt
            position[1] += v * np.sin(yaw) * self.dt
            yaw += omega * self.dt
            trajectory.append(position.copy())
        return trajectory

    def smooth_trajectory(self, trajectory):
        # 使用B样条平滑轨迹
        trajectory = np.array(trajectory)

        # 设定起点和目标点
        start_point = np.array([0.0, 0.0])  # 这里假设起点为原点，具体可以根据实际情况设定
        end_point = self.target  # 目标点由DWA中的目标位置决定

        # 将起点和目标点作为控制点的一部分
        control_points = np.vstack([start_point, trajectory, end_point])  # 把起点、DWA轨迹和目标点组合成控制点

        # 选择适当的B样条阶数和节点
        degree = 3  # 3阶B样条
        knots = np.linspace(0, 1, len(control_points) - degree + 1)  # 节点

        spl = BSpline(knots, control_points, degree)

        # 生成平滑轨迹
        smooth_trajectory = spl(np.linspace(0, 1, 10))  # 生成平滑轨迹的100个点
        # self.track_first_point(smooth_trajectory)
        # print("smooth_trajectory:", smooth_trajectory)
        return smooth_trajectory

    def calculate_control2(self, trajectory, dt):
            # 计算目标点的距离和角度误差
            target_point = trajectory[4]
            distance_error = np.linalg.norm(target_point)  # 目标点与当前位置的距离误差
            angle_error = np.arctan2(target_point[1], target_point[0])  # 目标点方向的角度误差

            # 通过 PID 控制器计算线速度和角速度
            linear_velocity = self.linear_pid.compute(distance_error, dt)
            angular_velocity = self.angular_pid.compute(angle_error, dt)

            # 限制最大速度
            linear_velocity = np.clip(linear_velocity, -self.max_speed, self.max_speed)
            angular_velocity = np.clip(angular_velocity, -self.max_yaw_rate, self.max_yaw_rate)
            return linear_velocity, angular_velocity

    # def calculate_control(self, smooth_trajectory, dt):
    #     # 计算目标点的距离和角度误差
    #     target_point = smooth_trajectory[2]
    #     distance_error = np.linalg.norm(target_point)  # 目标点与当前位置的距离误差
    #     angle_error = np.arctan2(target_point[1], target_point[0])  # 目标点方向的角度误差

    #     # 通过 PID 控制器计算线速度和角速度
    #     linear_velocity = self.linear_pid.compute(distance_error, dt)
    #     angular_velocity = self.angular_pid.compute(angle_error, dt)

    #     # 限制最大速度
    #     linear_velocity = np.clip(linear_velocity, -self.max_speed, self.max_speed) # 将linear_velocity 限制在 -self.max_speed 和 self.max_speed 之间
    #     angular_velocity = np.clip(angular_velocity, -self.max_yaw_rate, self.max_yaw_rate)

    #     return linear_velocity, angular_velocity

    # def calculate_control3(self, smooth_trajectory, dt):
    #     """
    #     为多个点计算线速度和角速度。
        
    #     参数:
    #         smooth_trajectory: list of [x, y] 表示轨迹点的列表
    #         dt: float 时间间隔

    #     返回:
    #         velocities: list of float 线速度列表
    #         angular_velocities: list of float 角速度列表
    #     """
    #     # 如果轨迹点少于2个，无法计算速度和角速度
    #     if len(smooth_trajectory) < 2:
    #         return [], []

    #     velocities = []  # 存储线速度
    #     angular_velocities = []  # 存储角速度

    #     # 遍历点，计算线速度和角速度
    #     for i in range(len(smooth_trajectory) - 1):
    #         # 当前点和下一个点
    #         x1, y1 = smooth_trajectory[i]
    #         x2, y2 = smooth_trajectory[i + 1]

    #         # 一阶差分计算线速度
    #         dx = (x2 - x1) / dt
    #         dy = (y2 - y1) / dt
    #         v = (dx**2 + dy**2)**0.5
    #         velocities.append(v)

    #         # 计算角速度（需要至少三个点）
    #         if i < len(smooth_trajectory) - 2:
    #             x3, y3 = smooth_trajectory[i + 2]
    #             theta1 = np.arctan2(y2 - y1, x2 - x1)
    #             theta2 = np.arctan2(y3 - y2, x3 - x2)
    #             dtheta = (theta2 - theta1) / dt
    #             angular_velocities.append(dtheta)
    #         else:
    #             # 最后一个点无法计算角速度，补0
    #             angular_velocities.append(0.0)

    #     return velocities, angular_velocities

    def score_trajectory(self, trajectory):
        # 评估轨迹的得分
        distance_to_target = np.linalg.norm(np.array(trajectory[-1]) - self.target) 
        obstacle_penalty = 0

        # # 避免障碍物：如果轨迹经过障碍物附近，惩罚得分
        # for obstacle in self.obstacles:
        #     distance_to_obstacle = np.linalg.norm(np.array(trajectory[-1]) - obstacle)

        #     if distance_to_obstacle < self.robot_radius:
        #         obstacle_penalty += 1  # 增加惩罚
        last_points = trajectory[-2:]  # 取轨迹上的最后三个点
        for obstacle in self.obstacles:
            # 遍历轨迹上的所有点
            for point in last_points:
                distance_to_obstacle = np.linalg.norm(np.array(point) - np.array(obstacle))         
                # 如果距离小于机器人半径，则增加惩罚
                if distance_to_obstacle < self.robot_radius:
                    obstacle_penalty += 1  # 增加惩罚
                    break  # 一旦发现轨迹上有点靠近障碍物，可以跳出循环，避免重复增加惩罚
        return -distance_to_target - 4 * obstacle_penalty  # 距离目标越近，得分越高，障碍物惩罚越重

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # 计算当前与目标的距离
            current_distance_to_target_x = self.target[0]
            current_distance_to_target_y = self.target[1]
            current_distance_to_target = np.sqrt(current_distance_to_target_x ** 2 + current_distance_to_target_y ** 2)
            # print("Current distance to target:\n", current_distance_to_target)
            if current_distance_to_target < 0.9:  # 如果距离小于0.5
                v = 0  # 停止线速度
                omega = 0  # 停止角速度
                print("Reached the target, stopping.\n")

            else:
                # v, omega = self.calculate_velocity()  # 计算速度--》预测轨迹--》评价得分
                # 计算最佳速度和轨迹
                best_v, best_omega, best_trajectory = self.calculate_velocity()
                if best_trajectory is not None:
                    smoothed_trajectory = self.smooth_trajectory(best_trajectory)
                    # v_list,omega_list = self.calculate_control3(smoothed_trajectory,0.1)
                    if abs(self.angle) < 1:
                        if self.change_count % 2 == 0:
                            smoothed_trajectory = self.smooth_trajectory(best_trajectory)
                            v,omega = self.calculate_control2(smoothed_trajectory,0.1)
                        else: 
                            v = 0
                            omega = 0
                    else:
                        if self.change_count % 2 == 0:
                            v = best_v
                            omega = best_omega
                        else: 
                            v = 0
                            omega = 0
                # print("v",v)
                # print("w",omega)

            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega
            self.cmd_pub.publish(cmd)
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('dwa_node')
    dwa = DWA()
    dwa.run()

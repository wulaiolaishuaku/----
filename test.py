import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point

class DWA:
    def __init__(self):
        # 设定机器人参量
        self.max_speed = 0.5  # 最大线速度
        self.max_yaw_rate = 1  # 最大角速度
        self.dt = 0.1  # 时间间隔
        self.wheel_base = 0.5  # 机器人轮距
        self.robot_radius = 0.4  # 机器人半径

        # 初始化位置信息
        # self.position = np.array([0.0, 0.0])  # 机器人在世界坐标系中的位置
        # self.yaw = 0.0  # 机器人的朝向角

        # 初始化障碍物列表
        self.obstacles = []

        # ROS 的发布器
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)  
        rospy.Subscriber('/base_station_position', Point, self.relative_position_callback)

        self.target = np.array([0.0, 0.0])  # 目标点的相对位置

    def lidar_callback(self, msg):
        # 这里可以用激光雷达数据进行障碍物检测
        self.detect_obstacles(msg.ranges)

    def relative_position_callback(self, msg):
        # 通过geometry_msgs/Point获取目标的相对位置
        # print("Received relative position\n", msg.y, msg.x)
        self.target = np.array([msg.y, msg.x])  # 提取目标的x和y坐标

    def detect_obstacles(self, ranges):
        # 激光雷达数据预处理，检测障碍物
        self.obstacles = []  # 清空障碍物列表
        for i, distance in enumerate(ranges):
            if distance < 0.5:  # 设定一个阈值，表示障碍物距离
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
        # 遍历所有可能的线速度和角速度
        for vi in v:
            for oi in omega:
                trajectory = self.predict_trajectory(vi, oi) # 预测该轨迹下未来轨迹
                score = self.score_trajectory(trajectory)  # 评估该轨迹的得分

                if score > max_score:
                    max_score = score
                    best_v = vi
                    best_omega = oi

        return best_v, best_omega

    def predict_trajectory(self, v, omega):
        # 预测未来轨迹
        trajectory = []
         # 初始化位置信息
        self.position = np.array([0.0, 0.0])  # 机器人在世界坐标系中的位置
        self.yaw = 0.0  # 机器人的朝向角
        for t in range(int(1 / self.dt)):
            self.position[0] += v * np.cos(self.yaw) * self.dt
            self.position[1] += v * np.sin(self.yaw) * self.dt
            self.yaw += omega * self.dt
            trajectory.append(self.position.copy())
        return trajectory

    def score_trajectory(self, trajectory):
        # 评估轨迹的得分
        distance_to_target = np.linalg.norm(np.array(trajectory[-1]) - self.target)
        obstacle_penalty = 0

        # 避免障碍物：如果轨迹经过障碍物附近，惩罚得分
        for obstacle in self.obstacles:
            distance_to_obstacle = np.linalg.norm(np.array(trajectory[-1]) - obstacle)
            if distance_to_obstacle < self.robot_radius:
                obstacle_penalty += 1  # 增加惩罚

        return -distance_to_target - obstacle_penalty  # 距离目标越近，得分越高，障碍物惩罚越重

    # def run(self):
    #     rate = rospy.Rate(20)  # 10 Hz
    #     while not rospy.is_shutdown():
    #         v, omega = self.calculate_velocity() #计算速度--》预测轨迹--》评价得分
    #         cmd = Twist()
    #         cmd.linear.x = v
    #         cmd.angular.z = omega
    #         self.cmd_pub.publish(cmd)
    #         rate.sleep()
    def run(self):
        rate = rospy.Rate(10)  # 20 Hz
        while not rospy.is_shutdown():
            # 计算当前与目标的距离
            current_distance_to_target_x = self.target[0]
            current_distance_to_target_y = self.target[1]
            current_distance_to_target = np.sqrt(current_distance_to_target_x ** 2 + current_distance_to_target_y ** 2)
            print("Current distance to target:\n", current_distance_to_target)
            if current_distance_to_target < 0.5:  # 如果距离小于0.5
                v = 0  # 停止线速度
                omega = 0  # 停止角速度
                print("Reached the target, stopping.\n")

            else:
                v, omega = self.calculate_velocity()  # 计算速度--》预测轨迹--》评价得分

            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega
            self.cmd_pub.publish(cmd)
            rate.sleep()



if __name__ == '__main__':
    rospy.init_node('dwa_node')
    dwa = DWA()
    dwa.run()

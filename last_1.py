# Made By WangTao
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from scipy.interpolate import BSpline
from math import atan2
from std_msgs.msg import UInt8
from concurrent.futures import ThreadPoolExecutor  # 并行计算模块
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
        self.robot_radius = 0.52  # 机器人半径
        self.linear_velocity = 0
        self.angular_velocity = 0
        self.angle = 0
        # 初始化障碍物列表
        self.obstacles = []
        self.distances = []
        self.linear_pid = PIDController(kp=10, ki=0.1, kd=0.5)  # 调整线速度的 PID 参数
        self.angular_pid = PIDController(kp=12.0, ki=0.1, kd=0.5)  # 调整角速度的 PID 参数
        self.tag_sta = 0
        self.count = 0
        self.change_count = 0
        self.last_data = 0   
        self.first_value = 0
        self.value_at_90 = 0
        self.value_at_60 = 0
        self.value_at_30 = 0
        self.value_at_270 = 0
        self.value_at_300 = 0
        self.value_at_330 = 0
        self.toggle_state = False
        # ROS 的发布器
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)  
        rospy.Subscriber('/base_station_position', Point, self.relative_position_callback)
        rospy.Subscriber('/tag_status', UInt8, self.tag_status)
        self.target = np.array([0.0, 0.0])  # 目标点的相对位置

    def lidar_callback(self, msg):
        # print(msg.angle_min)
        self.detect_obstacles(msg.ranges)
    
    def relative_position_callback(self, msg):
        # 通过geometry_msgs/Point获取目标的相对位置
        self.target = np.array([msg.y, msg.x])  # 提取目标的x和y坐标
        if self.target[0] !=0 or self.target[1] !=0:
            self.angle = atan2(self.target[1], self.target[0])  # 计算目标的角度
            # print("angle:",self.angle)

    def tag_status(self,msg):
        # 获取当前收到的数据
        current_data = msg.data
        # 检查数据是否从0变化为128
        if self.last_data == 0 and current_data == 128:
            self.change_count += 1  # 增加变化计数
        self.last_data = current_data
    
    def detect_obstacles(self, ranges):
        # 激光雷达数据预处理，检测障碍物
        # print(ranges)
        self.obstacles = []
        num_points = len(ranges)  # 激光雷达数据点数量
        self.first_value = ranges[0]  # 获取 ranges 的第一个值
        angle_resolution = 0.36  # 激光雷达的角度分辨率
        target_angle_90 = 90  # 目标角度
        target_angle_10 =10
        target_angle_15 =15
        target_angle_20 =20
        target_angle_25 =25
        target_angle_30 =30
        target_angle_40 =40
        target_angle_50 =50
        target_angle_60 = 60
        target_angle_70 = 70
        target_angle_80 = 80
        target_angle_270 = 270
        target_angle_275 = 275
        target_angle_330 = 330
        target_angle_300 = 300
        target_angle_305 = 305
        target_angle_280 = 280
        target_angle_285 = 285
        target_angle_290 = 290
        target_angle_295 = 295
        target_angle_310 = 310
        target_angle_320 = 320
        target_angle_340 = 340
        target_angle_350 = 350
        index_10 = int(target_angle_10 / angle_resolution)  
        index_20 = int(target_angle_20 / angle_resolution)   
        index_30 = int(target_angle_30 / angle_resolution) 
        index_40 = int(target_angle_40 / angle_resolution)  
        index_50 = int(target_angle_50 / angle_resolution)  
        index_60 = int(target_angle_60 / angle_resolution) 
        index_70 = int(target_angle_70 / angle_resolution)  
        index_80 = int(target_angle_80 / angle_resolution)  
        index_90 = int(target_angle_90 / angle_resolution)
        index_270 = int(target_angle_270 / angle_resolution)
        index_275 = int(target_angle_275 / angle_resolution)
        index_280 = int(target_angle_280 / angle_resolution)
        index_290 = int(target_angle_290 / angle_resolution)
        index_310 = int(target_angle_310 / angle_resolution)
        index_320 = int(target_angle_320 / angle_resolution)
        index_330 = int(target_angle_330 / angle_resolution)
        index_340 = int(target_angle_340 / angle_resolution)
        index_350 = int(target_angle_350 / angle_resolution)
        index_300 = int(target_angle_300 / angle_resolution)
        if index_90 < len(ranges) and index_270 < len(ranges) and index_30 < len(ranges) and index_330 < len(ranges) and index_60 < len(ranges) and index_300 < len(ranges):
            self.value_at_10 = ranges[index_10]
            self.value_at_20 = ranges[index_20]
            self.value_at_30 = ranges[index_30]
            self.value_at_40 = ranges[index_40]
            self.value_at_50 = ranges[index_50]
            self.value_at_60 = ranges[index_60]
            self.value_at_70 = ranges[index_70]
            self.value_at_80 = ranges[index_80]
            self.value_at_90 = ranges[index_90]
            self.value_at_270 = ranges[index_270]
            self.value_at_275 = ranges[index_275]
            self.value_at_280 = ranges[index_280]
            self.value_at_290 = ranges[index_290]
            self.value_at_300 = ranges[index_300]
            self.value_at_310 = ranges[index_310]
            self.value_at_320 = ranges[index_320]
            self.value_at_330 = ranges[index_330]
            self.value_at_340 = ranges[index_340]
            self.value_at_350 = ranges[index_350]
            # print("value_at_20:",self.value_at_30) 
            # print("value_at_30:",self.value_at_30)
            # print("value_at_300:",self.value_at_60)
            # print("value_at_270:",self.value_at_270)
        max_distance = 3.0  # 阈值，表示障碍物距离
        # total_angles = 360  # 激光雷达扫描角度范围
        for i, distance in enumerate(ranges):
            if distance < max_distance :  # 只处理小于阈值的障碍物
                angle = i * angle_resolution  # 计算角度
                x = distance * np.cos(np.radians(angle))  # 极坐标转笛卡尔坐标
                y = distance * np.sin(np.radians(angle))
                self.obstacles.append(np.array([x, y]))  # 保存障碍物坐标  
        return self.obstacles
    
    # def calculate_distances_to_origin(self):
    #     # 使用欧几里得距离计算每个障碍物到原点 (0, 0) 的距离
    #     self.distances = []
    #     for obstacle in self.obstacles:
    #         x, y = obstacle
    #         distance = np.sqrt(x**2 + y**2)  # 计算距离
    #         self.distances.append(distance)
    #     return self.distances

    
    def calculate_velocity(self):
        # 计算DWA速度指令，考虑目标点和障碍物
        # dis = self.calculate_distances_to_origin()
        # min_distance = np.min(dis)  # 最小的距离
        # print(min_distance)
        # if min_distance < 1:
        v = np.linspace(0, self.max_speed * 0.5, num=5)
        omega = np.linspace(-self.max_yaw_rate * 1.1, self.max_yaw_rate * 1.1, num=30)
        # elif min_distance < 3.0:s
        # # 当障碍物距离中等时，适度采样
        #     v = np.linspace(0, self.max_speed , num=8)
        #     omega = np.linspace(-self.max_yaw_rate , self.max_yaw_rate , num=20)
        # else :
        #     v = np.linspace(0, self.max_speed, num=10)  # 线速度从0到最大，生成一个从0到最大线速度的线性空间，分成10个值，表示机器人可以选择的不同线速度。
        #     # print("v:",v)
        #     omega = np.linspace(-self.max_yaw_rate, self.max_yaw_rate, num=15)  # 角速度从最小到最大
        # print("w:",omega)
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
        return best_v, best_omega, best_trajectory


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

    def score_trajectory(self, trajectory):
        # 评估轨迹的得分
        # 计算目标点（轨迹最后一个点）与目标位置的距离
        distance_to_target = np.linalg.norm(np.array(trajectory[-1]) - self.target)
        # 将障碍物列表转换为一个二维数组 (障碍物坐标的集合)
        obstacle_positions = np.array(self.obstacles)
        # print("obstacle_positions:",obstacle_positions)
        # 将轨迹点列表转换为二维数组（此处取轨迹的最后三个点）
        # last_points_array = np.array(trajectory[-5:])
        last_points_array = np.array(trajectory)
        # print("last_points_array",last_points_array)
        # 计算所有轨迹点与所有障碍物之间的欧几里得距离
        # 利用广播机制，计算轨迹点与障碍物之间的距离矩阵
        if obstacle_positions.size == 0:  # 检查障碍物数组是否为空
            return np.inf  # 返回一个较大的得分，表示轨迹不可行
        distances = np.linalg.norm(last_points_array[:, np.newaxis, :] - obstacle_positions, axis=2)
        forward_penalty = 0
        if self.first_value < 1.0:
            forward_penalty += self.first_value 
        else:
            forward_penalty = 0
        # 判断距离是否小于机器人半径，并计算惩罚
        obstacle_penalty = np.sum(distances < self.robot_radius)
        # 计算得分，距离目标越近得分越高，障碍物惩罚越重
        return -distance_to_target - 100 * obstacle_penalty 

    def run(self):
        rate = rospy.Rate(20)  # 10 Hz
        while not rospy.is_shutdown():
            # 计算当前与目标的距离
            current_distance_to_target_x = self.target[0]
            current_distance_to_target_y = self.target[1]
            current_distance_to_target = np.sqrt(current_distance_to_target_x ** 2 + current_distance_to_target_y ** 2)
            # print("Current distance to target:\n", current_distance_to_target)
            if current_distance_to_target < 1.0 :  # 如果距离小于0.5
                v = 0  # 停止线速度
                omega = 0  # 停止角速度
                # print("Reached the target, stopping.\n")
            else :
                # v, omega = self.calculate_velocity()  # 计算速度--》预测轨迹--》评价得分
                # 计算最佳速度和轨迹  
                best_v, best_omega, best_trajectory = self.calculate_velocity() 
                if best_trajectory is not None: 
                    # v_list,omega_list = self.calculate_control3(smoothed_trajectory,0.1)   
                    if self.first_value > 1.2 or self.value_at_10 > 1.2 or self.value_at_20 >1.2 or self.value_at_350 > 1.2 or self.value_at_340 > 1.2 or self.value_at_30 > 1.2 or self.value_at_330 > 1.2:  
                        if abs(self.angle) < 2 :   
                            if self.change_count % 2 == 0 :   
                                if self.value_at_90 < 0.4 or self.value_at_30 < 0.6 or self.value_at_60 < 0.6 or self.value_at_10 < 0.6 or self.value_at_20 < 0.6 or self.value_at_40 < 0.5 or self.value_at_50 < 0.5 or self.value_at_70 < 0.5 or self.value_at_80 < 0.5:
                                    v = 0.1  
                                    omega = -0.3  
                                elif self.value_at_300 < 0.6 or self.value_at_270 < 0.4 or self.value_at_330 < 0.6 or self.value_at_320 < 0.6 or self.value_at_310 < 0.6 or self.value_at_290 < 0.5 or self.value_at_280 < 0.5 or self.value_at_340 < 0.6 or self.value_at_350 < 0.6:
                                    v = 0.1  
                                    omega = 0.3    
                                else : 
                                    smoothed_trajectory = self.smooth_trajectory(best_trajectory)  
                                    v,omega = self.calculate_control2(smoothed_trajectory,0.1) 
                                print("v:",v)  
                                print("w:",omega) 
                            else : 
                                v = 0  
                                omega = 0  
                        else : 
                            if self.change_count % 2 == 0 :  
                                # v = best_v  
                                # omega = best_omega  
                                if self.first_value < 0.5 or self.value_at_350 < 0.6 or self.value_at_340 < 0.6 or self.value_at_330 < 0.6 or self.value_at_320 < 0.6 or self.value_at_310 < 0.6 or self.value_at_300 < 0.6 or self.value_at_10 <0.5 or self.value_at_20 <  0.5 or self.value_at_30 < 0.5 or self.value_at_40 < 0.5 or self.value_at_50 < 0.5 or self.value_at_290 < 0.6:
                                    v = - 0.2  
                                    omega = 0  
                                else:  
                                    v = best_v   
                                    omega = best_omega  
                                print("v:",v)   
                                print("w:",omega)  
                            else :  
                                v = 0 
                                omega = 0 
                    else :  
                        if self.change_count % 2 == 0 :   
                            # if self.value_at_90 > self.value_at_270:         
                            if abs(self.angle) > 2 :  
                                # v = best_v
                                # omega = best_omega
                                if self.first_value < 0.5 or self.value_at_350 < 0.6 or self.value_at_340 < 0.6 or self.value_at_330 < 0.6 or self.value_at_320 < 0.6 or self.value_at_310 < 0.6 or self.value_at_300 < 0.5 or self.value_at_10 <0.5 or self.value_at_20 <  0.5 or self.value_at_30 < 0.5 or self.value_at_40 < 0.5 or self.value_at_50 < 0.5 or self.value_at_290 < 0.6:
                                    v = - 0.2  
                                    omega = 0  
                                else:
                                    v = best_v 
                                    omega = best_omega
                            else :
                                if self.angle > 0 :
                                    omega = 0.7   
                                    v = 0   
                                    # time.sleep(5)  # 保持 5 秒
                                    # print("omega",omega)
                                else :
                                    omega = -0.7
                                    v = 0                        
                                    # print("omega",omega)
                        else : 
                            v = 0 
                            omega = 0 
            
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega
            self.cmd_pub.publish(cmd)
            rate.sleep() # 控制循环频率
            
if __name__ == '__main__':
    rospy.init_node('dwa_node')
    dwa = DWA()
    dwa.run()

























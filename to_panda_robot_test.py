#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import rospy, sys
import moveit_commander
from moveit_commander import RobotCommander, MoveGroupCommander, PlanningSceneInterface
from geometry_msgs.msg import PoseStamped, Pose
from copy import deepcopy
from to_position_server_test import Barriers, System
from to_barrier import get_barrier_data
import threading
import math
global flag
global result_thread_barrier


def point_inside_sphere(x_p, y_p, z_p, x_c, y_c, z_c, r):
    # 计算点到球心的距离
    d = math.sqrt((x_p - x_c) ** 2 + (y_p - y_c) ** 2 + (z_p - z_c) ** 2)

    # 判断点是否在球内部
    if d <= r:
        return True
    else:
        return False


def is_float_equal(a, b, epsilon=1e-4):
    return abs(a - b) < epsilon

def thread_function(arm, b_x, b_y, b_z):
    global result_thread_barrier
    # time.sleep(0.01)
    while flag:
        # 获取当前端点位置坐标
        wpose = arm.get_current_pose(end_effector_link).pose
        # print(f'wpose = {wpose}')
        x_current = wpose.position.x
        y_current = wpose.position.y
        z_current = wpose.position.z

        # print(f'当前端点位置：{x_current}, {y_current}, {z_current}')

        barriers = Barriers()

        barrier_data = get_barrier_data()
        for l in barrier_data:
            name = l[0]
            x_ = l[1]
            y_ = l[2]
            z_ = l[3]
            r_ = l[4]
            print(f'障碍物 {name}\tx:{x_}\ty:{y_}z:{z_}r:{r_}')
            barriers.add_barriers(a=x_, b=y_, c=z_, r=r_)

        a_new, b_new, c_new, r_new, distance_min = barriers.get_barriers(x=x_current, y=y_current, z=z_current)

        if distance_min < 0.05:
            print('遇到障碍物')
            # if is_float_equal(a=a_, b=a_new) and is_float_equal(a=b_, b=b_new) and is_float_equal(a=c_, b=c_new):
            #     pass
            # else:
            #     arm.stop()
            #     print(a_)
            #     print(b_)
            #     print(c_)
            #     print('*' * 80)
            #     print(a_new)
            #     print(b_new)
            #     print(c_new)
            #     print('障碍物位置发生变化，重新计算路径')
            #     result_thread_barrier = True
            #     break

            if distance_min < 0.15:
                arm.stop()
                result_thread_barrier = True
                break



def runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 运行时间: {end_time - start_time} 秒")
        return result
    return wrapper

@runtime
def run(x_target, y_target, z_target):
    # 设置机械臂工作空间中的目标位姿，位置使用x、y、z坐标描述，
    # 姿态使用四元数描述，基于base_link坐标系
    target_pose = PoseStamped()
    target_pose.header.frame_id = reference_frame
    target_pose.header.stamp = rospy.Time.now()
    target_pose.pose.position.x = x_target
    target_pose.pose.position.y = y_target
    target_pose.pose.position.z = z_target
    target_pose.pose.orientation.x = 0.0
    target_pose.pose.orientation.y = 0.01
    target_pose.pose.orientation.z = -0.9
    target_pose.pose.orientation.w = 0.38

    # 设置机械臂终端运动的目标位姿，使机器人的终端到达目标位置
    arm.set_pose_target(target_pose, end_effector_link)

    # 规划运动路径
    # traj = arm.plan()
    plan_success, traj, planning_time, error_code = arm.plan()

    # 按照规划的运动路径控制机械臂运动
    arm.execute(traj)

@runtime
def run_list(x_list, y_list, z_list):
    finish = False
    arm.set_start_state_to_current_state()

    # 获取当前位姿数据最为机械臂运动的起始位姿
    global plan
    start_pose = arm.get_current_pose(end_effector_link).pose
    # 初始化路点列表
    waypoints = []
    # 将初始位姿加入路点列表
    # waypoints.append(start_pose)
    wpose = deepcopy(start_pose)

    # 设置路点数据，并加入路点列表
    for i in range(len(x_list)):
    # for i in range(3):

        wpose.position.x = x_list[i]
        wpose.position.y = y_list[i]
        wpose.position.z = z_list[i]
        waypoints.append(deepcopy(wpose))
        # wpose.position.x += 0.1
        # waypoints.append(deepcopy(wpose))


    fraction = 0.0  # 路径规划覆盖率
    maxtries = 50  # 最大尝试规划次数
    attempts = 0  # 已经尝试规划次数

    # 尝试规划一条笛卡尔空间下的路径，依次通过所有路点
    while fraction < 1.0 and attempts < maxtries:
        (plan, fraction) = arm.compute_cartesian_path(
            waypoints,  # waypoint poses，路点列表
            0.01,  # eef_step，终端步进值
            0.0,  # jump_threshold，跳跃阈值
            True)

        # 尝试次数累加
        attempts += 1

        # fraction = -fraction
        print(f'fraction = {fraction}')
        # 打印运动规划进程
        if attempts % 10 == 0:
            rospy.loginfo("Still trying after " + str(attempts) + " attempts...")

        # 如果路径规划成功（覆盖率100%）,则开始控制机械臂运动
        if fraction == 1.0:
            rospy.loginfo("Path computed successfully. Moving the arm.")
            print('OK')
            finish = True
            arm.execute(plan, wait=True)
            break

        # 如果路径规划失败，则打印失败信息
        else:
            rospy.loginfo(
                "Path planning failed with only " + str(fraction) + " success after " + str(maxtries) + " attempts.")

        if attempts > 20 and fraction >= 0.5:
            print('规划部分路径，先执行')
            arm.execute(plan, wait=True)
            break

        if attempts > 30 and fraction >= 0.2:
            print('规划部分路径，先执行')
            arm.execute(plan, wait=True)
            break

        if attempts > 40 and fraction >= 0.1:
            print('规划部分路径，先执行')
            arm.execute(plan, wait=True)
            break

    # arm.execute(plan, wait=True)

    if finish:
        print('finish')

    return finish




if __name__ == "__main__":

    print("OK")

    # 初始化move_group的API
    moveit_commander.roscpp_initialize(sys.argv)
    print("ok2")
    # 初始化ROS节点
    rospy.init_node('demo')
    print('ok3')
    # 初始化需要使用move group控制的机械臂中的arm group
    arm = moveit_commander.MoveGroupCommander('panda_arm')
    print('ok4')
    # 获取终端link的名称
    end_effector_link = arm.get_end_effector_link()
    print(f'end_effector_link = {end_effector_link}')

    # # 回到初始姿态
    # arm.set_named_target('ready')
    # arm.go()
    rospy.sleep(3)
    print('ready')

    # 初始化场景对象，用来监听外部环境的变化
    scene = PlanningSceneInterface()

    # 设置目标位置所使用的参考坐标系
    reference_frame = 'panda_link0'
    arm.set_pose_reference_frame(reference_frame)

    # 当运动规划失败后，允许重新规划
    arm.allow_replanning(True)

    # 设置位置(单位：米)和姿态（单位：弧度）的允许误差
    arm.set_goal_position_tolerance(0.01)
    arm.set_goal_orientation_tolerance(0.05)

    arm.set_max_acceleration_scaling_factor(0.05)
    arm.set_max_velocity_scaling_factor(0.05)


    # 末端端点目标位置
    # goal_list1 = [0.3, 0.5, 0.59]
    # goal_list2 = [0.3, -0.5, 0.59]
    goal_list1 = [0.55, -0.5, 0.17]
    goal_list2 = [0.55, 0.5, 0.17]

    i_ = 0
    for i in range(20):
        if i_ % 2 == 0:
            # 回到初始姿态
            # arm.set_named_target('ready')
            # arm.go()
            goal_x = goal_list1[0]
            goal_y = goal_list1[1]
            goal_z = goal_list1[2]
        else:
            goal_x = goal_list2[0]
            goal_y = goal_list2[1]
            goal_z = goal_list2[2]

        # goal_x = 0.3
        # goal_y = 0.5
        # goal_z = 0.59

        # 获取当前端点位置坐标
        wpose = arm.get_current_pose(end_effector_link).pose
        # print(f'wpose = {wpose}')
        x_current = wpose.position.x
        y_current = wpose.position.y
        z_current = wpose.position.z


        print(f'当前端点位置：{x_current}, {y_current}, {z_current}')

        # 获取障碍物状态
        barriers = Barriers()

        barrier_data = get_barrier_data()
        for l in barrier_data:
            name = l[0]
            x = l[1]
            y = l[2]
            z = l[3]
            r = l[4]
            print(f'障碍物 {name}\tx:{x}\ty:{y}z:{z}r:{r}')
            barriers.add_barriers(a=x, b=y, c=z, r=r)

        # barriers.add_barriers(a=0.3, b=0.2, c=0.45, r=0.05)

        a_, b_, c_, r_, distance_min = barriers.get_barriers(x=x_current, y=y_current, z=z_current)

        # a_ = 0.3
        # b_ = 0.4
        # c_ = 0.6
        # r_ = 0.1

        # # 判断终点是否在障碍物缓冲区内部
        # if point_inside_sphere(x_p=goal_x, y_p=goal_y, z_p=goal_z,
        #                        x_c=a_, y_c=b_, z_c=c_, r=0.25):
        #     print('终点在缓冲区内部，请移动障碍物')
        #
        #     time.sleep(3)
        #     continue


        system = System()
        start_time = time.time()



        x, y, z = system.run(x1_o=x_current, x2_o=y_current, x3_o=z_current,
                         goal_x=goal_x, goal_y=goal_y, goal_z=goal_z,
                         a_=a_, b_=b_, c_=c_, r_=r_+0.05, distance_min=distance_min, return_list=True, barrier_data=barrier_data)
        print(f'target : {x}, {y}, {z}')

        end_time = time.time()
        print(f"获取目标位置运行时间: {end_time - start_time} 秒")

        # scene.remove_world_object("ball")
        # sphere_pose = PoseStamped()
        # sphere_pose.header.frame_id = 'panda_link0'
        # sphere_pose.pose.position.x = a_
        # sphere_pose.pose.position.y = b_
        # sphere_pose.pose.position.z = c_
        # scene.add_sphere("ball", sphere_pose, 0.05)

        for i, l in enumerate(barrier_data):
            name = l[0]
            x_b = l[1]
            y_b = l[2]
            z_b = l[3]
            r_b = l[4]
            print(f'缓冲区 ball_{i} \tx:{x_b}\ty:{y_b}z:{z_b}r:{r_b}')
            scene.remove_world_object(f"ball_{i}")
            sphere_pose = PoseStamped()
            sphere_pose.header.frame_id = 'panda_link0'
            sphere_pose.pose.position.x = x_b
            sphere_pose.pose.position.y = y_b
            sphere_pose.pose.position.z = z_b
            scene.add_sphere(f"ball_{i}", sphere_pose, r_b)

        # run_list(x_list=x, y_list=y, z_list=z)



        flag = True
        result_thread_barrier = False
        # 创建子线程 监听障碍物
        thread = threading.Thread(target=thread_function, args=(arm, a_, b_, c_))
        thread.start()

        if len(x) == 0:
            finish = False
        else:
            finish = run_list(x_list=x, y_list=y, z_list=z)

        print(a_)
        print(b_)
        print(c_)
        print(r_)
        # rospy.sleep(5)
        flag = False
        thread.join()

        if not result_thread_barrier and finish:
            i_ += 1

        if i_ == 20:
            break







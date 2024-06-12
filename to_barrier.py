# -*- coding: utf-8 -*-
# @Author : Bernard

import rospy
from gazebo_msgs.srv import *
import re

def gwp_client():
    rospy.wait_for_service('/gazebo/get_world_properties')
    try:
        gwp = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        resp = gwp()
        return resp
    except rospy.ServiceException as e:
        print("Service call failed: %s") % e
        return False


def gms_client(model_name,relative_entity_name):
    rospy.wait_for_service('/gazebo/get_model_state')
    try:
        gms = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        resp = gms(model_name,relative_entity_name)
        return resp
    except rospy.ServiceException as e:
        print ("Service call failed: %s")%e
        return False


def get_barrier_data():
    result = []

    res = gwp_client()
    if not res:
        return False
    barrier_names = res.model_names[:]

    try:
        barrier_names.remove('table')
    except:
        pass

    try:
        barrier_names.remove('ground_plane')
    except:
        pass
    barrier_names.remove('panda')


    relative_entity_name = 'world'
    radius_dict = {'o1': 0.1,
                   'box1': 0.5 * 2 * 0.05, # 正方体
                   'box2': 0.5 * 2 * 0.1, # 正方体
                   'box3': 0.5 * 0.33, # 长方体
                   # 'box2':
                }
    for model_name in barrier_names:
        res = gms_client(model_name, relative_entity_name)
        x = res.pose.position.x
        y = res.pose.position.y
        z = res.pose.position.z
        find_ = re.sub(r'_.*$', '', model_name)
        r = radius_dict[find_]
        result.append([model_name, x, y, z, r])

    return result

if __name__ == "__main__":

    print(get_barrier_data())




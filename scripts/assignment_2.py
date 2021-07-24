#!/usr/bin/env python

import rospy
import sys
import numpy as np
import actionlib
import time
import cv2
import math
import sys
import os
import multi_move_base 

from pprint import pprint
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid, Odometry
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from numpy import inf
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from datetime import datetime, timedelta

print(sys.version)
curr_file_loc = os.path.dirname(os.path.realpath(__file__))
media_dir_path = os.path.join(curr_file_loc, "tamer_files_2")
print(curr_file_loc)

TIMEOUT = timedelta(minutes=10)

if not os.path.exists(media_dir_path):
    os.mkdir(media_dir_path)
    print("Created directory for aux files.")

MAP_IMG_PATH = os.path.join(media_dir_path, "map_img.jpg")
LOCAL_MAP_IMG_PATH = os.path.join(media_dir_path, "local_map_img.jpg")
WALL_IMG_PATH = os.path.join(media_dir_path, "wall_img.jpg")
CONT_IMG_PATH = os.path.join(media_dir_path, "contour_img.jpg")
EDT_IMG_PATH = os.path.join(media_dir_path, "edt_img.jpg")
EDT_ANOT_IMG_PATH = lambda level: os.path.join(media_dir_path, "edt_img_circs_{}.jpg".format(level))
LOCAL_SPHERES_IMG_PATH = lambda sphere_ind: os.path.join(media_dir_path, "sphere_img_{}.jpg".format(sphere_ind))

robot_location = robot_rotation = robot_orientation = None
global_map = None
global_map_info = None
global_map_origin = None
global_points = []
positions = []
spheres_centers = []

def callback_odom(msg):
    '''
    Obtains Odometer readings and update global Variables
    '''
    global robot_location, robot_rotation, robot_orientation, robot_location_pos
    robot_location_pos = msg.pose.pose
    location = [msg.pose.pose.position.x, msg.pose.pose.position.y]
    robot_location = location
    orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation)
    rot = [roll, pitch, yaw]
    robot_rotation = rot
    robot_orientation = orientation


def convert_to_np(grid):
    shape = grid.info.width, grid.info.height
    arr = np.array(grid.data, dtype='float32').reshape(shape)
    return arr.T
    
def get_map():
    map_getter = rospy.ServiceProxy('static_map', GetMap)
    grid = map_getter().map
    grid_info = grid.info
    map_arr = convert_to_np(grid)
    map_origin = [grid_info.origin.position.x, grid_info.origin.position.y]
    return map_arr, grid_info, map_origin, grid

def map_to_img(map_arr, save=True, path=None):
    map_img = map_arr.copy()
    map_img[map_img==0] = 255 # free
    map_img[map_img==100] = 0 # object
    map_img[map_img==-1] = 0  # else
    map_img = map_img.astype(np.uint8)
    map_img = map_img.clip(0, 255)
    if save:
        cv2.imwrite(path or MAP_IMG_PATH, map_img)
    return map_img

def walls_to_img(map_arr):
    wall_img = map_arr.copy()
    wall_img[wall_img==0] = 0 # free
    wall_img[wall_img==100] = 255 # object
    wall_img[wall_img==-1] = 0  # else
    wall_img = wall_img.astype(np.uint8)
    wall_img = wall_img.clip(0, 255)
    cv2.imwrite(WALL_IMG_PATH, wall_img)
    return wall_img

def contour_to_img(): 
    global MAP_IMG_PATH, CONT_IMG_PATH
    image = cv2.imread(MAP_IMG_PATH)
    img_grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(img_grey, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    image_contour = cv2.drawContours(erosion.copy(), contours, -1, (0, 255, 0), 1)
    cv2.imwrite(CONT_IMG_PATH,image_contour)
    return image_contour

def walls_edt_img(walls_img, contour_img):
    edt = walls_img.copy()
    edt = cv2.distanceTransform(contour_img, cv2.DIST_L2, 3)
    cv2.normalize(edt, edt, 0, 1.0, cv2.NORM_MINMAX)
    edt *= 255.0
    edt = np.uint8(edt)
    edt *= np.uint8(np.bool_(contour_img))
    cv2.imwrite(EDT_IMG_PATH, edt)
    return edt

def distance_compute_(p1, p2):
    pts = [np.array([p.position.x, p.position.y, p.position.z]) for p in (p1, p2)]
    return np.linalg.norm([pts[0], pts[1]])

def distance_compute(pos1,pos2,Type = 'd'):
    '''
    Distance Compute between two positions
    '''
    x1 = pos1[0]
    y1 = pos1[1]
    x2 = pos2[0]
    y2 = pos2[1]
    d = ((x1-x2)**2) + ((y1-y2)**2)
    if Type == 'd':
        return math.sqrt(d)
    if Type == 'eu':
        return d
    if Type == 'manhattan':
        return abs(x1-x2)+abs(y1-y2)

def euler_to_quaternion(yaw):
        w = np.cos(yaw/2)
        z = np.sin(yaw/2) 
        return (z, w)

def generate_goals(edt, contour, level=2, levels_num=8, step_factor=10):
    global robot_location, global_map_origin, EDT_ANOT_IMG_PATH, EDT_IMG_PATH
    grid = edt.copy()
    assert level + 1  <= levels_num
    rows, cols = np.where((grid > 255 * level // levels_num) & (grid < 255 * (level+1) // levels_num) & (contour > 0))
    indexes = list(range(0, len(rows), len(rows) // step_factor))
    while robot_location is None:
        print("waiting for location")
        time.sleep(1)
    indexes = sorted(indexes, key=lambda i: distance_compute(np.array([rows[i], cols[i]])*0.05 + global_map_origin, robot_location))    
    tmp = np.column_stack((rows[indexes], cols[indexes]))

    # anotate map
    edt_level = cv2.imread(EDT_IMG_PATH)
    edt_level[rows, cols] = [0,172,254] # color level range in Orange
    for ix, point in enumerate(tmp):
        point = point[1], point[0]
        cv2.putText(edt_level, str(ix), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0,0,255), 1)
        cv2.circle(edt_level,tuple(point),2,(0,255,0), thickness=-1)
    
    robot_location_on_map = (int((robot_location[1]-global_map_origin[1])/0.05), int((robot_location[0]-global_map_origin[0])/0.05))
    cv2.circle(edt_level, robot_location_on_map,3,(255,0,0), thickness=-1) # mark robot in Blue
    cv2.putText(edt_level, "R", robot_location_on_map, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    cv2.imwrite(EDT_ANOT_IMG_PATH(level),edt_level)
    
    return tmp

def vacuum_cleaning(agent_id):
       
    #multi_move_base.move(0,1,0.5)
    x = 0
    y = 1   
    print('-'*100, agent_id, type(agent_id))
    print('cleaning ({},{})'.format(x,y))
    result = multi_move_base.move(agent_id, x,y)
    
    print('moving agent %d' % agent_id)
    x = 1
    y = 0   
    print('cleaning ({},{})'.format(x,y))
    result = multi_move_base.move(agent_id, x,y)
    
    
    #multi_move_base.move(1,2,0.5)
    #multi_move_base.move(x,y)
    #raise NotImplementedError

def inspection():
    print('start inspection')
    raise NotImplementedError



# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':

    # Initializes a rospy node to let the SimpleActionClient publish and subscribe
    rospy.init_node('assignment_2')

    exec_mode = sys.argv[1] 
    print('exec_mode:' + exec_mode)        

    agent_id = int(sys.argv[2])
    print('agent id: %d' % agent_id)        
    if exec_mode == 'cleaning':        
        vacuum_cleaning(agent_id)
    elif exec_mode == 'inspection':
        inspection()
    else:
        print("Code not found")
        raise NotImplementedError

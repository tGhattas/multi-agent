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

# from tqdm import tqdm
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
media_dir_path = os.path.join(curr_file_loc, "tamer_files")
print(curr_file_loc)

TIMEOUT = timedelta(minutes=10)

if not os.path.exists(media_dir_path):
    os.mkdir(media_dir_path)
    print("Created directory for aux files.")

MAP_IMG_PATH = os.path.join(media_dir_path, "map_img.jpg")
LOCAL_MAP_IMG_PATH = lambda aid: os.path.join(media_dir_path, "agent_{}_local_map_img.jpg" % aid)
WALL_IMG_PATH = os.path.join(media_dir_path, "wall_img.jpg")
CONT_IMG_PATH = os.path.join(media_dir_path, "contour_img.jpg")
EDT_IMG_PATH = os.path.join(media_dir_path, "edt_img.jpg")
EDT_ANOT_IMG_PATH = lambda level: os.path.join(media_dir_path, "edt_img_circs_{}.jpg".format(level))
LOCAL_SPHERES_IMG_PATH = lambda sphere_ind: os.path.join(media_dir_path, "sphere_img_{}.jpg".format(sphere_ind))
ROBOT_ROUTE_IMG_PATH = os.path.join(media_dir_path, "route_img.jpg")


robot_location = robot_rotation = robot_orientation = None
global_map = None
route_map = None
global_map_info = None
global_map_origin = None
global_points = []
positions = []
spheres_centers = []

## for inspection task
robot_location_pos = None
side_scan_start_angle = 20
side_scan_range = 60
front_scan_range = 16
distance_from_wall = 0.4

x = np.zeros(360)
front_wall_dist = 0  # front wall distance
left_wall_dist = 0  # left wall distance
right_wall_dist = 0  # right wall distance

# PID parameters
kp = 4
kd = 450
ki = 0

k1 = kp + ki + kd
k2 = -kp - 2 * kd
k3 = kp
########################

def callback_laser(data):
    '''
    Obtains Laser readings and update global Variables
    '''
    global left_wall_dist, right_wall_dist, x, front_wall_dist, front_scan_range, side_scan_start_angle, side_scan_range

    x = list(data.ranges)
    for i in range(360):
        if x[i] == inf:
            x[i] = 7
        if x[i] == 0:
            x[i] = 6

        # store scan data
    left_wall_dist = min(x[side_scan_start_angle:side_scan_start_angle + side_scan_range])  # left wall distance
    right_wall_dist = min(x[360 - side_scan_start_angle - side_scan_range:360 - side_scan_start_angle])  # right wall distance
    front_wall_dist = min(min(x[0:int(front_scan_range / 2)], x[int(360 - front_scan_range / 2):360]))  # front wall distance



def callback_odom(msg, anotate=True):
    '''
    Obtains Odometer readings and update global Variables
    '''
    global robot_location, robot_rotation, robot_orientation, robot_location_pos, route_map
    robot_location_pos = msg.pose.pose
    location = [msg.pose.pose.position.x, msg.pose.pose.position.y]
    if anotate:
        robot_loc_on_map = to_map_img_point(*location)
        cv2.circle(route_map, robot_loc_on_map, 1, (0, 255, 0), thickness=-1)
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

def vacuum_cleaning():
    global MAP_IMG_PATH, global_map, global_map_info, global_map_origin, TIMEOUT
    print('start vacuum_cleaning')
    start_ts = datetime.now()
    global_map, global_map_info, global_map_origin, grid = get_map() 
    sub_odom = rospy.Subscriber('/odom', Odometry, callback_odom)

    rospy.init_node('movebase_client_py')
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    map_img = map_to_img(global_map)
    walls_img = walls_to_img(global_map)
    contour_img = contour_to_img()
    edt_img = walls_edt_img(walls_img, contour_img)

    
    
    ## run
    
    levels_num = 6
    for level in range(1, levels_num):
    # for level in tqdm(range(1, levels_num), desc="Levels Progress"):
        current_ts = datetime.now()
        if current_ts - start_ts > TIMEOUT:
            break
        print('-'*15)
        print('-'*5, 'Level {}'.format(level))
        print('-'*15)
        frontiers = generate_goals(edt_img, contour_img, level=level, levels_num=levels_num, step_factor=5)
        for i in range(len(frontiers)):
            frontier = frontiers[i] * 0.05 + global_map_origin

            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            print("going for :" + str(frontier))

            goal.target_pose.pose.position.x = frontier[0]
            goal.target_pose.pose.position.y = frontier[1]
            qz, qw = euler_to_quaternion(math.atan2(frontier[1], frontier[0]))
            goal.target_pose.pose.orientation.w = qw 
            goal.target_pose.pose.orientation.z = qz

            client.send_goal(goal)
            wait = client.wait_for_result(rospy.Duration(60))


def not_identical_to_other_center(x, y, radius=40):
    global spheres_centers
    for a, b in spheres_centers:
        if math.sqrt((a-x)**2 + (b-y)**2) < radius:
            return False
    return True


def local_mapper():
    global global_map_origin, global_map, global_map_info, LOCAL_MAP_IMG_PATH, spheres_centers, LOCAL_SPHERES_IMG_PATH
    local_map = rospy.wait_for_message('move_base/local_costmap/costmap', OccupancyGrid)
    local_points = np.transpose(np.array(local_map.data).reshape(
                                (local_map.info.width, local_map.info.height)))
    local_position = np.array([local_map.info.origin.position.x, local_map.info.origin.position.y])
    local_map_img = map_to_img(local_points, path=LOCAL_MAP_IMG_PATH(self.id))


    ## detect spheres
    img = cv2.imread(LOCAL_MAP_IMG_PATH(self.id), cv2.IMREAD_COLOR)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    smooth = cv2.GaussianBlur(grey, (5,5), 1.5**2)

    circles = cv2.HoughCircles(smooth, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=20)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)
            cv2.circle(img, (x, y), 1, (0, 0, 255), 3)
            center_local_position = np.array((x, y)) * 0.05 +  local_position
            global_local_points_index = (center_local_position - global_map_origin) / 0.05
            global_to_local_x = int(global_local_points_index[0])
            global_to_local_y = int(global_local_points_index[1])
        
            if not_identical_to_other_center(global_to_local_x, global_to_local_y):
                print('-'*10)
                print(spheres_centers)
                print(global_to_local_x, global_to_local_y)
                print('-'*10)
                cv2.imwrite(LOCAL_SPHERES_IMG_PATH(len(spheres_centers)), img)
                spheres_centers.append((global_to_local_x, global_to_local_y))


def inspection():
    global k1, k2, k3, kp, kd, ki, front_wall_dist, x, right_wall_dist, left_wall_dist, route_map
    global distance_from_wall, robot_location_pos, global_map_origin, global_map_info, global_map
    print('start inspection')

    start_ts = datetime.now()
    rospy.init_node('wall_following_control')
    odom_sub = rospy.Subscriber('/odom', Odometry, callback_odom)
    velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    scan_subscriber = rospy.Subscriber('/scan', LaserScan, callback_laser)
    rate = rospy.Rate(10)  # 20hz
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    while robot_location is None:
        time.sleep(0.01)
    start_pos = robot_location

    global_map, global_map_info, global_map_origin, grid = get_map()
    route_map = map_img.copy()
    route_map = cv2.cvtColor(route_map, cv2.COLOR_GRAY2BGR)
    prev_error = 0
    bird_left_nest = False

    while distance_from_wall < 1.5:
        current_ts = datetime.now()
        if current_ts - start_ts > TIMEOUT:
            break

        local_mapper()

        dist_from_start = distance_compute(start_pos, robot_location)

        delta = distance_from_wall - right_wall_dist  # distance error

        if dist_from_start > 1.5:
            bird_left_nest = True
        if bird_left_nest and dist_from_start <= 1.5:
            # update dist from wall
            distance_from_wall += 0.1
            bird_left_nest = False
         

        # PID controller
        PID_output = kp * delta + kd * (delta - prev_error)

        # stored states
        prev_error = delta

        # clip PID output
        angular_zvel = np.clip(PID_output, -1.2, 1.2)
        linear_vel = np.clip((front_wall_dist - 0.35), -0.1, 0.4)

        # log IOs
        log = '\n distance from right wall in cm ={} / {}\n'.format(int(right_wall_dist * 100), distance_from_wall * 100)
        log += ' distance from front wall in cm ={}\n'.format(int(front_wall_dist * 100))
        log += ' distance from nest ={}\n'.format(dist_from_start)
        log += ' linear_vel={} angular_vel={} \n'.format(linear_vel, angular_zvel)
        log += ' current dist from walls threshold={} \n'.format(distance_from_wall)
        log += ' detected spheres={} \n'.format(len(spheres_centers))
        rospy.loginfo(log)

        # publish cmd_vel
        vel_msg = Twist(Vector3(linear_vel, 0, 0), Vector3(0, 0, angular_zvel))
        velocity_publisher.publish(vel_msg)
        rate.sleep()

    velocity_publisher.publish(Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)))

    if route_map is not None:
            cv2.imwrite(ROBOT_ROUTE_IMG_PATH, route_map)
    
    print('{} spheres were found'.format(len(spheres_centers)))
    return len(spheres_centers)


def inspection_color():
    print('start inspection_color')
    raise NotImplementedError



# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':

    exec_mode = sys.argv[1] 
    print('exec_mode:' + exec_mode)
    if exec_mode == 'cleaning':
        vacuum_cleaning()
    elif exec_mode == 'inspection':
        inspection()
    elif exec_mode == 'inspection_color':
        inspection_color()
    else:
        print("Code not found")
        raise NotImplementedError



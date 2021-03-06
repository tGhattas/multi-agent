#!/usr/bin/env python

import rospy
import numpy as np
import time
import cv2
import math
import sys
import os
import multi_move_base 
import actionlib
import json

from numpy import inf
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.srv import GetMap, GetPlan
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion
from datetime import datetime, timedelta
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from multiprocessing import Process
from threading import Thread, Lock

print(sys.version)
curr_file_loc = os.path.dirname(os.path.realpath(__file__))
media_dir_path = os.path.join(curr_file_loc, "tamer_files_2")
print(curr_file_loc)

TIMEOUT = timedelta(minutes=5)
rival_id = None


if not os.path.exists(media_dir_path):
    os.mkdir(media_dir_path)
    print("Created directory for aux files.")

MAP_IMG_PATH = os.path.join(media_dir_path, "map_img.jpg")
LOCAL_MAP_IMG_PATH = lambda aid: os.path.join(media_dir_path, "agent_{}_local_map_img.jpg".format(aid))
WALL_IMG_PATH = os.path.join(media_dir_path, "wall_img.jpg")
CONT_IMG_PATH = os.path.join(media_dir_path, "contour_img.jpg")
EDT_IMG_PATH = os.path.join(media_dir_path, "edt_img.jpg")
EDT_ANOT_IMG_PATH = lambda level: os.path.join(media_dir_path, "edt_img_circs_{}.jpg".format(level))
LOCAL_SPHERES_IMG_PATH = lambda sphere_ind: os.path.join(media_dir_path, "sphere_img_{}.jpg".format(sphere_ind))
ROBOT_ROUTE_IMG_PATH = lambda aid: os.path.join(media_dir_path, "agent_{}_route_img.jpg".format(aid))
GENERIC_PATH = lambda path_: os.path.join(media_dir_path, path_)


robot_location = robot_rotation = robot_orientation = None
global_map = None
route_map = None
global_map_info = None
global_map_origin = None
global_points = []
positions = []
spheres_centers = []
spheres_centers_list_lock = Lock()
move_base_clients = {}
pub_dirt_list = []


############################# Callbacks

def callback_odom(msg, anotate=False):
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

############################# Callbacks - end

#############################  aux

def not_identical_to_other_center(x, y, radius=60):
    global spheres_centers
    for a, b in spheres_centers:
        if math.sqrt((a-x)**2 + (b-y)**2) < radius:
            return False
    return True

def get_agent_loc(agent_id):
    msg = rospy.wait_for_message('tb3_{}/odom'.format(agent_id), Odometry)
    pos = msg.pose.pose.position
    return pos.x, pos.y

def convert_to_np(grid):
    shape = grid.info.width, grid.info.height
    arr = np.array(grid.data, dtype='float32').reshape(shape)
    return arr.T
    
def get_map(agent_id=0):
    map_getter = rospy.ServiceProxy('tb3_{}/static_map'.format(agent_id), GetMap)
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
    results = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(results) > 2:
        _, contours, hierarchy = results
    else:
        contours, hierarchy = results

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

def distance_compute(pos1, pos2, metric='d'):
    '''
    Distance Compute between two positions
    '''
    x1 = pos1[0]
    y1 = pos1[1]
    x2 = pos2[0]
    y2 = pos2[1]
    d = ((x1-x2)**2) + ((y1-y2)**2)
    if metric == 'd':
        return math.sqrt(d)
    if metric == 'eu':
        return d
    if metric == 'manhattan':
        return abs(x1-x2)+abs(y1-y2)
    if metric == 'path':
        pass

def euler_to_quaternion(yaw):
        w = np.cos(yaw/2)
        z = np.sin(yaw/2) 
        return (z, w)

def subcribe_location(agent_id=0, callback=callback_odom, anotate=True):
    if anotate:
        rospy.Subscriber('tb3_{}/odom'.format(agent_id), Odometry, lambda msg: callback_odom(msg, True))
    else:
        rospy.Subscriber('tb3_{}/odom'.format(agent_id), Odometry, callback)

def subcribe_laser(agent_id=0, callback=None):
    rospy.Subscriber('tb3_{}/scan'.format(agent_id), LaserScan, callback)

def get_vel_publisher(agent_id=0):
    return rospy.Publisher('tb3_{}/cmd_vel'.format(agent_id), Twist, queue_size=10)

def to_map_img_point(x, y):
    return int((y-global_map_origin[1])/0.05), int((x-global_map_origin[0])/0.05)

def move(client, goal, degree, convert_to_map_coords=False):
    global global_map_origin
    x, y = goal
    if convert_to_map_coords:
        position = np.array([x, y]) * 0.05 + global_map_origin
    else:
        position = np.array([x, y])
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "/map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = position[0]
    goal.target_pose.pose.position.y = position[1]
    z, w = euler_to_quaternion(degree)
    goal.target_pose.pose.orientation.x = 0
    goal.target_pose.pose.orientation.y = 0
    goal.target_pose.pose.orientation.z = z
    goal.target_pose.pose.orientation.w = w
    client.send_goal(goal)
    wait = client.wait_for_result(rospy.Duration(60))

#############################  aux - end


#############################  C & C

def get_agent_plan(start, end, agent_id=0):
    rospy.loginfo('agent {} waiting for move_base/make_plan service...'.format(agent_id))
    rospy.wait_for_service('/tb3_%d/move_base/make_plan' % agent_id, 30)
    get_plan = rospy.ServiceProxy('/tb3_%d/move_base/make_plan' % agent_id, GetPlan)
    rospy.loginfo('agent {} got for move_base/make_plan service.'.format(agent_id))
    plan = GetPlan()
    plan.start = start
    plan.goal = end
    plan.tolerance = .5
    res = get_plan(plan.start, plan.goal, plan.tolerance)
    myPath = len(res.plan.poses)
    return res.plan

def get_pose_stamped(seq, frame_id, stamp, x, y):
    pose = PoseStamped()
    pose.header.seq = seq
    pose.header.frame_id = frame_id
    pose.header.stamp = stamp
    pose.pose.position.x = x
    pose.pose.position.y = y
    return pose

def dirt_ETAs(agent_id):
    global pub_dirt_list
    update_dirt_list()
    res_ids = []
    dirt_id_map = {}
    dirt_eta_map = {}
    agent_loc = get_agent_loc(agent_id)
    rival_loc = get_agent_loc(1-agent_id)

    agent_start = get_pose_stamped(0, "/tb3_%d/map" % agent_id, rospy.Time(0), agent_loc[0], agent_loc[1])
    rival_start = get_pose_stamped(0, "/tb3_%d/map" % agent_id, rospy.Time(0), rival_loc[0], rival_loc[1])

    for dirt_pos in pub_dirt_list:
        goal = get_pose_stamped(0, "/tb3_%d/map" % agent_id, rospy.Time(0), dirt_pos[0], dirt_pos[1])

        plan = get_agent_plan(agent_start, goal, agent_id)
        agent_eta = len(plan.poses)

        plan = get_agent_plan(rival_start, goal, agent_id)
        rival_eta = len(plan.poses)
        
        closer_agent = agent_id if rival_eta > agent_eta else 1-agent_id 
        res_ids.append(closer_agent)
        dirt_id_map[tuple(dirt_pos)] = closer_agent
        dirt_eta_map[tuple(dirt_pos)] = {agent_id:agent_eta, 1-agent_id:rival_eta}
    return res_ids, dirt_id_map, dirt_eta_map 

def closer_dirts(agent_id=0):
    global pub_dirt_list
    update_dirt_list()
    res = []
    map_ = {}
    agent_loc = get_agent_loc(agent_id)
    rival_loc = get_agent_loc(1-agent_id)
    for dirt_pos in pub_dirt_list:
        dirt_pos_np = np.array(dirt_pos)
        closer_agent = agent_id if distance_compute(dirt_pos_np, agent_loc) < distance_compute(dirt_pos_np, rival_loc) else 1-agent_id
        res.append(closer_agent)
        map_[tuple(dirt_pos)] = closer_agent
    return res, map_

def sort_dirts(dirts, annotate=True, agent_id=0, by_path=False):
    global robot_location, global_map_origin, EDT_ANOT_IMG_PATH, EDT_IMG_PATH
    agent_loc = get_agent_loc(agent_id)
    if not by_path:
        sorted_dirt_list = sorted(dirts, key=lambda _: distance_compute(np.array(_), agent_loc))    
    else:
        assert isinstance(dirts, dict), 'dirts should be a dict if by_path=True'
        dirts_list = [(dirt, eta_dict[agent_id]) for (dirt, eta_dict) in dirts.items()]
        sorted_dirt_list = sorted(dirts_list, key=lambda _: _[1])    
        sorted_dirt_list = [_[0] for _ in sorted_dirt_list]
    if annotate:
        # anotate map
        edt_level = cv2.imread(EDT_IMG_PATH)
        # edt_level[rows, cols] = [0,172,254] # color level range in Orange
        for ix, point in enumerate(sorted_dirt_list):
            point = to_map_img_point(*point)
            cv2.putText(edt_level, str(ix), point, cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 1)
            cv2.circle(edt_level, point, 2, (0, 255, 0), thickness=-1)
        
        robot_location_on_map = to_map_img_point(*agent_loc)

        cv2.circle(edt_level, robot_location_on_map, 3, (255, 0, 0), thickness=-1) # mark robot in Blue
        cv2.putText(edt_level, "R", robot_location_on_map, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imwrite(EDT_ANOT_IMG_PATH(agent_id), edt_level)
    
    return sorted_dirt_list


def multi_move(x, y, agent_id=0):
    global move_base_clients
    if not agent_id in move_base_clients:
        # get and cache move base client
        client = actionlib.SimpleActionClient('/tb3_%d/move_base' % agent_id, MoveBaseAction)
        client.wait_for_server()
        move_base_clients[agent_id] = client
    move_base_client = move_base_clients[agent_id]
    rospy.loginfo('agent {} heading to {}'.format(agent_id, (x, y)))
    angle = math.atan2(y, x)
    move(move_base_client, (x, y), angle)

def basic_cleaning(dirts_list, agent_id=0):
    rospy.loginfo('agent {} started BASIC cleaning'.format(agent_id))
    sorted_dirts = sort_dirts(dirts_list, agent_id=agent_id)
    for g in sorted_dirts:
        x, y = g
        multi_move(x, y, agent_id)

def get_rival_goal(rival_id):
    try:
        # channel = 'tb3_%d/move_base/current_goal' % rival_id
        # channel = 'tb3_%d/move_base/DWAPlannerROS/global_plan' % rival_id
        # channel = 'tb3_%d/move_base/NavfnROS/plan' % rival_id
        channel = 'tb3_%d/move_base/DWAPlannerROS/global_plan' % rival_id
        rival_goal_msg = rospy.wait_for_message(channel, Path, 5)
        # rival_goal_msg = rospy.wait_for_message(channel, PoseStamped, 5)
        dest = rival_goal_msg.poses[-1]
        rival_goal = (dest.pose.position.x, dest.pose.position.y)
        rospy.loginfo('agent {} got rival goal as {}'.format(1-rival_id, rival_goal))
     
    except rospy.exceptions.ROSException as e:
        rospy.logerr(e)
        rospy.logerr('agent {} setting trash rival goal.'.format(1-rival_id))
        rival_goal = None
    return rival_goal

def update_dirt_list():
    global pub_dirt_list
    msg = rospy.wait_for_message('dirt', String)
    try:
        string = msg.data
        string = string.replace('(', '[')
        string = string.replace(')', ']')
        pub_dirt_list = json.loads(string)
        rospy.loginfo("Recieved dirt list: {}".format(pub_dirt_list))
    except Exception as e:
        print(e)
        print(msg)
        raise Exception("Dirt list published in a format other than string = " % str(msg))
    

def rotate(target, agent_id=0):
    rate = rospy.Rate(10)
    velocity_publisher = get_vel_publisher(agent_id)
    while not rospy.is_shutdown():
        global robot_rotation
        while robot_rotation is None:
            rospy.loginfo('agent waiting for rotation.')
            time.sleep(0.1)
        yaw = robot_rotation[-1]
        rot_cmd = Twist()
        rot_cmd.angular.z = 0.5 * (target-yaw)
        if abs(target-yaw) < 1.0:
            velocity_publisher.publish(Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))) # stop signal
            rospy.loginfo('agent {} finished rotating.'.format(agent_id))
            break
        velocity_publisher.publish(rot_cmd)
        rospy.loginfo('agent {} is rotating.'.format(agent_id))
        rate.sleep()

def competitive_cleaning(agent_id=0, path_based_dist=True):
    global pub_dirt_list, robot_location
    rospy.loginfo('agent {} started COMPETITIVE cleaning'.format(agent_id))
    rate = rospy.Rate(10)
    update_dirt_list()

    prev_loc = [_ for _ in robot_location]
    timestamp = datetime.now()
    

    while len(pub_dirt_list):
        # try to unstuck by rotating
        elapsed_time = datetime.now() - timestamp
        if distance_compute(prev_loc, robot_location) < 0.5 and elapsed_time > timedelta(seconds=30):
            rospy.loginfo("try to unstuck agent {}...".format(agent_id))
            rotate(math.pi/2, agent_id)
        else:
            prev_loc = [_ for _ in robot_location]
            timestamp = datetime.now()

        if not path_based_dist:
            sorted_dirts = sort_dirts(pub_dirt_list, agent_id=agent_id)
            rival_sorted_dirts = sort_dirts(pub_dirt_list, agent_id=1-agent_id, annotate=False)
            closer_dirts_ind, closer_dirt_map = closer_dirts(agent_id)
        else:
            try:
                closer_dirts_ind, closer_dirt_map, dirt_eta_map = dirt_ETAs(agent_id)
                sorted_dirts = sort_dirts(dirt_eta_map, agent_id=agent_id, by_path=True)
                rival_sorted_dirts = sort_dirts(dirt_eta_map, agent_id=1-agent_id, annotate=False, by_path=True)
            except rospy.ServiceException as e:
                rospy.logerr(e)
                sorted_dirts = sort_dirts(pub_dirt_list, agent_id=agent_id)
                rival_sorted_dirts = sort_dirts(pub_dirt_list, agent_id=1-agent_id, annotate=False)
                closer_dirts_ind, closer_dirt_map = closer_dirts(agent_id)


        agent_1_stronger = sum(closer_dirts_ind) > len(pub_dirt_list) // 2

        his = mine = []
        for ag_id, dirt_pos in zip(closer_dirts_ind, pub_dirt_list):
            if ag_id != agent_id:
                his.append(dirt_pos)
            else:
                mine.append(dirt_pos)

        goals = mine + his # first clean more confident dirts
        g = goals.pop(0)
        if len(goals):
            rival_goal = get_rival_goal(1-agent_id)
            if rival_goal and distance_compute(g, np.array(rival_goal)) < 0.5 and closer_dirt_map[(g[0], g[1])] != agent_id:
                # skip an impossible goal
                rospy.loginfo('agent {} skipping impossible goal.'.format(agent_id))
                continue
            x, y = g
            multi_move(x, y, agent_id)
            update_dirt_list()
        rate.sleep()
        


def vacuum_cleaning(agent_id):
    global MAP_IMG_PATH, global_map, global_map_info, global_map_origin, TIMEOUT
    global rival_id, pub_dirt_list, route_map

    rospy.init_node('vacuum_cleaning_{}'.format(agent_id))
    rospy.loginfo('agent {} started cleaning'.format(agent_id))
    
    global_map, global_map_info, global_map_origin, grid = get_map(agent_id)     
    map_img = map_to_img(global_map)
    route_map = map_img.copy()
    route_map = cv2.cvtColor(route_map, cv2.COLOR_GRAY2BGR)
    walls_img = walls_to_img(global_map)
    contour_img = contour_to_img()
    edt_img = walls_edt_img(walls_img, contour_img)

    subcribe_location(agent_id)

    rival_id = 1-agent_id

    while len(pub_dirt_list) == 0:
        update_dirt_list()
        time.sleep(0.1)
    dirt_list_cpy = [_ for _ in pub_dirt_list]
    try:
        # if agent_id==0:
        #     basic_cleaning(pub_dirt_list, agent_id)
        # else:
        competitive_cleaning(agent_id)
        
    except rospy.exceptions.ROSException as e:
        rospy.logerr('------ROSException thrown={}'.format(str(e)))
        rospy.info('agent {} running basic cleaning.'.format(agent_id))
        basic_cleaning(pub_dirt_list, agent_id)

    finally:
        if route_map is not None:
            if len(dirt_list_cpy):
                for dirt_pos in dirt_list_cpy:
                    dirt_loc_on_map = to_map_img_point(*dirt_pos)
                    cv2.circle(route_map, dirt_loc_on_map, 2, (0, 0, 255), thickness=-1)
            cv2.imwrite(ROBOT_ROUTE_IMG_PATH(agent_id), route_map)

##############################################################################################################################

#### Class of inspection robot

class Robot:

    def __init__(self, agent_id, reverse=False):
        self.id = agent_id
        self.reverse = reverse
        self.robot_location = None
        self.robot_rotation = None
        self.robot_location_pos = None
        self.robot_orientation = None
        self.global_points = []
        self.positions = []
        self.spheres_centers = []
        self.rotated = False

        self.side_scan_start_angle = 20
        self.side_scan_range = 60
        self.front_scan_range = 16
        self.distance_from_wall = 0.4
        self.dist_from_start = 0
        self.start_pos = None
        self.prev_error = 0
        self.bird_left_nest = False

        self.x = np.zeros(360)
        self.front_wall_dist = 0  # front wall distance
        self.left_wall_dist = 0  # left wall distance
        self.right_wall_dist = 0  # right wall distance

        # PID parameters
        self.kp = 4
        self.kd = 450
        self.ki = 0

        subcribe_location(agent_id, self.callback_odom, anotate=False)
        subcribe_laser(agent_id, self.callback_laser)
        self.velocity_publisher = get_vel_publisher(agent_id)

    def callback_laser(self, data):
        '''
        Obtains Laser readings and update global Variables
        '''
        self.x = list(data.ranges)
        for i in range(360):
            if self.x[i] == inf:
                self.x[i] = 7
            if self.x[i] == 0:
                self.x[i] = 6

            # store scan data
        self.left_wall_dist = min(self.x[self.side_scan_start_angle:self.side_scan_start_angle + self.side_scan_range])  # left wall distance
        self.right_wall_dist = min(self.x[360 - self.side_scan_start_angle - self.side_scan_range:360 - self.side_scan_start_angle])  # right wall distance
        self.front_wall_dist = min(min(self.x[0:int(self.front_scan_range / 2)], self.x[int(360 - self.front_scan_range / 2):360]))  # front wall distance

    def callback_odom(self, msg):
        '''
        Obtains Odometer readings and update global Variables
        '''
        self.robot_location_pos = msg.pose.pose
        location = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.robot_location = location
        orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation)
        rot = [roll, pitch, yaw]
        self.robot_rotation = rot
        self.robot_orientation = orientation
        if self.start_pos is None:
            self.start_pos = self.robot_location

    def local_mapper(self):
        global global_map_origin, global_map, global_map_info, LOCAL_MAP_IMG_PATH, spheres_centers
        global LOCAL_SPHERES_IMG_PATH, spheres_centers_list_lock, GENERIC_PATH
        local_map = rospy.wait_for_message('/tb3_{}/move_base/local_costmap/costmap'.format(self.id), OccupancyGrid)
        local_points = np.transpose(np.array(local_map.data).reshape(
                                    (local_map.info.width, local_map.info.height)))
        local_position = np.array([local_map.info.origin.position.x, local_map.info.origin.position.y])
        local_map_img = map_to_img(local_points, path=LOCAL_MAP_IMG_PATH(self.id))


        ## detect spheres
        img = cv2.imread(LOCAL_MAP_IMG_PATH(self.id), cv2.IMREAD_COLOR)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        smooth = cv2.GaussianBlur(grey, (9,9), 1.5**2)
        cv2.imwrite(GENERIC_PATH("gaussian_{}.jpg".format(self.id)), smooth)

        min_radius = 5
        circles = cv2.HoughCircles(smooth, cv2.HOUGH_GRADIENT, 1.5, 20, param1=40, param2=35, minRadius=min_radius, maxRadius=15)

        edges = cv2.Canny(grey, 10, 40, apertureSize=3)
        minLineLength = 10
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength, maxLineGap)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img, (x, y), 1, (0, 0, 255), 1)

                center_local_position = np.array((x, y)) * 0.05 + local_position
                global_local_points_index = (center_local_position - global_map_origin) / 0.05
                global_to_local_x = int(global_local_points_index[0])
                global_to_local_y = int(global_local_points_index[1])

                with spheres_centers_list_lock:
                    new = not_identical_to_other_center(global_to_local_x, global_to_local_y, radius=70)
                    if new and lines is not None:
                        for i in range(len(lines)):
                            try:
                                for x1,y1,x2,y2 in lines[i]:
                                    # calculate distance between sphere center and line
                                    d = np.linalg.norm(np.cross(np.array([x2,y2])-np.array([x1,y1]), np.array([x1,y1])-np.array([x, y]))) / np.linalg.norm(np.array([x2,y2])-np.array([x1,y1]))
                                    if d < min_radius:
                                        cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
                                        new = False
                                        break
                            except Exception as e:
                                rospy.logerr(e) # no need to raise further

                    if new:
                        log = '\n'+'-'*10
                        log += '\n'+str(spheres_centers)
                        log += '\n'+str((global_to_local_x, global_to_local_y))
                        log += '\n'+'-'*10
                        rospy.loginfo(log)
                        spheres_centers.append((global_to_local_x, global_to_local_y))
                if new:
                    cv2.imwrite(LOCAL_SPHERES_IMG_PATH("id{}_{}".format(self.id, len(spheres_centers))), img)
    
    def rotate(self, target):
        while not rospy.is_shutdown():
            while self.robot_rotation is None:
                rospy.loginfo('agent {} waiting for rotation.'.format(self.id))
                time.sleep(0.1)
            yaw = self.robot_rotation[-1]
            rot_cmd = Twist()
            rot_cmd.angular.z = 0.5 * (target-yaw)
            if abs(target-yaw) < 1.0:
                self.stop()
                rospy.loginfo('agent {} finished rotating.'.format(self.id))
                break
            self.velocity_publisher.publish(rot_cmd)
            rospy.loginfo('agent {} is rotating.'.format(self.id))
            rospy.Rate(10).sleep()

    def step(self):
        global spheres_centers
        
        self.local_mapper()

        if self.reverse:
            if not self.rotated:
                # rotate
                target = math.pi
                self.rotate(target)
                self.rotated = True

            delta = self.distance_from_wall - self.left_wall_dist  # distance error
            delta = -delta 
        else:
            delta = self.distance_from_wall - self.right_wall_dist  # distance error
        self.dist_from_start = distance_compute(self.start_pos, self.robot_location)
        if self.dist_from_start > 1.5:
           self.bird_left_nest = True
        if self.bird_left_nest and self.dist_from_start <= 1.5:
            # update dist from wall
            self.distance_from_wall += 0.1
            self.bird_left_nest = False
         

        # PID controller (PD actually)
        PID_output = self.kp * delta + self.kd * (delta - self.prev_error)

        # stored states
        self.prev_error = delta

        # clip PID output
        angular_zvel = np.clip(PID_output, -1.3, 1.3)
        linear_vel = np.clip((self.front_wall_dist - 0.35), -0.2, 0.3)


        # log IOs
        log = '\n agent {} distance from right wall in cm ={} / {}\n'.format(self.id, int(self.right_wall_dist * 100), self.distance_from_wall * 100)
        log += ' agent {} distance from left wall in cm ={} / {}\n'.format(self.id, int(self.left_wall_dist * 100), self.distance_from_wall * 100)
        log += ' agent {} distance from front wall in cm ={}\n'.format(self.id, int(self.front_wall_dist * 100))
        log += ' agent {} distance from nest ={}\n'.format(self.id, self.dist_from_start)
        log += ' agent {} linear_vel={} angular_vel={} \n'.format(self.id, linear_vel, angular_zvel)
        log += ' current dist from walls threshold={} \n'.format(self.distance_from_wall)
        log += ' detected spheres={} \n'.format(len(spheres_centers))
        rospy.loginfo(log)

        rospy.loginfo(' agent {} PID={}, delta={}'.format(self.id, PID_output, delta))

        # publish cmd_vel
        vel_msg = Twist(Vector3(linear_vel, 0, 0), Vector3(0, 0, angular_zvel))
        self.velocity_publisher.publish(vel_msg)

    def stop(self):
        self.velocity_publisher.publish(Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)))
import traceback

def thread_step(agent, start_ts=datetime.now()):
    rate = rospy.Rate(10)  # 20hz
    try:
        while agent.distance_from_wall < 1.5:
            current_ts = datetime.now()
            if current_ts - start_ts > TIMEOUT:
                break
            agent.step()
            rate.sleep()
    except Exception as e:
        rospy.logerr('agent {} exception raised:{}'.format(agent.id, str(e)))
        print(traceback.format_exc())
        raise e
    finally:
        agent.stop()
        

def inspection():
    global global_map_origin, global_map_info, global_map, GENERIC_PATH
    rospy.init_node('wall_following_control')
    rospy.loginfo('start inspection')
    
    agent_0 = Robot(0)
    agent_1 = Robot(1, reverse=True)

    # start_ts = datetime.now()
    
    rate = rospy.Rate(10)  # 20hz

    while agent_0.robot_location is None or agent_1.robot_location is None:
        time.sleep(0.01)

    global_map, global_map_info, global_map_origin, grid = get_map()
    prev_error = 0
    bird_left_nest = False

    try:
        
        agent_t0 = Thread(target=thread_step, args=(agent_0,))
        agent_t1 = Thread(target=thread_step, args=(agent_1,))

        agent_t0.start()
        agent_t1.start()

        agent_t0.join()
        agent_t1.join()

    finally:
        print("{} spheres were found.".format(len(spheres_centers)))
        print("spheres locations {}".format(spheres_centers))
        dists = {}
        
        map_img = map_to_img(global_map, save=False)
        map_img = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
        for s in spheres_centers:
            cv2.circle(map_img, (s[1],s[0]), 3, (0, 255, 0), thickness=-1) # mark robot in Blue
            dists[s] = {tuple(s_): distance_compute(s_,s) for s_ in spheres_centers}

        cv2.imwrite(GENERIC_PATH("spheres_locs.jpg"), map_img)
        print(dists)
        return len(spheres_centers)



# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':

    # Initializes a rospy node to let the SimpleActionClient publish and subscribe
    # rospy.init_node('assignment_2')

    exec_mode = sys.argv[1] 
    print('exec_mode:' + exec_mode)        

    agent_id = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    if len(sys.argv) >= 3:
        print('agent id: %d' % agent_id)        
    if exec_mode == 'cleaning':        
        vacuum_cleaning(agent_id)
    elif exec_mode == 'inspection':
        inspection()
    else:
        print("Code not found")
        raise NotImplementedError

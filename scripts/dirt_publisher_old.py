#!/usr/bin/env python
import rospy
import math
from math import floor
from itertools import product
from std_msgs.msg import String


class DirtPublisher:

    def __init__(self):
        self.dirt_pub = rospy.Publisher('dirt', String, queue_size=10, latch=True)
        self.dirt_pieces = '[3:4],[5:2],[0:1]'

    def run(self):
          # Main while loop
          while not rospy.is_shutdown():
            self.publish_objects()


    def publish_objects(self):            
            
        self.dirt_pub.publish(self.dirt_pieces)


if __name__ == '__main__':
    rospy.init_node('dirt_publisher')
    try:
        dirt_pub = DirtPublisher()
        dirt_pub.run()
    except rospy.ROSInterruptException:
        pass
#!/usr/bin/env python
import random
import numpy as np
import time
import fire
import string

#import ROS package
import rospy
from sensor_msgs.msg import Joy, Image, Imu, CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Float32, String
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge
#import trained policy
from pytorch_policy_agent import Policy
import ros_numpy
#from PIL import Image
import PIL.Image
import torch
import queue


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class Controller(object):
    tele_twist = Twist()
    def __init__(self, rate):
        self._timer = Timer()
        self._rate = rospy.Rate(rate)
        self._enable_auto_control = False
        self.current_control = None
        self.trajectory_index = None
        self.bridge = CvBridge()
        self.img_queue = queue.Queue()
        # Callback data store
        self.image0 = None
        self.image1 = None
        self.image2 = None
        self.intention = None
        # self.imu = None
        self.odom0 = None
        self.odom1 = None
        # self.labeled_control = None
        self.key = None
        # self.scan = None

        # Subscribe ros messages
        rospy.Subscriber('/robot_0/image_0', Image, self.cb_image0, queue_size=1, buff_size=2 ** 10)
        rospy.Subscriber('/robot_0/image_1', Image, self.cb_image1, queue_size=1, buff_size=2 ** 10)
        rospy.Subscriber('/robot_0/image_2', Image, self.cb_image2, queue_size=1, buff_size=2 ** 10)
        # rospy.Subscriber('/scan', LaserScan, self.cb_scan, queue_size=1, buff_size=2 ** 10)
        # rospy.Subscriber('/imu', Imu, self.cb_imu, queue_size=1, buff_size=2 ** 10)
        rospy.Subscriber('/robot_0/odom', Odometry, self.cb_odom0, queue_size=1, buff_size=2 ** 10)
        rospy.Subscriber('/robot_1/odom', Odometry, self.cb_odom1, queue_size=1, buff_size=2 ** 10)
        #rospy.Subscriber('/joy', Joy, self.cb_joy)
        #rospy.Subscriber('/keyboard_control', Keyboard, self.cb_keyboard)
        # rospy.Subscriber('/labeled_control', Twist, self.cb_labeled_control, queue_size=1)
        # rospy.Subscriber('/speed', Float32, self.cb_speed, queue_size=1)
        '''
        if self._mode == "DLM":
            rospy.Subscriber('/test_intention', String, self.cb_dlm_intention, queue_size=1)
        else:
            rospy.Subscriber('/intention_lpe', Image, self.cb_lpe_intention, queue_size=1, buff_size=2 ** 10)
        '''
        # Publish Control
        self.control_pub = rospy.Publisher('/robot_0/cmd_vel', Twist, queue_size=1)

        # Publish as training data
        #self.pub_intention = rospy.Publisher('/train/intention', String, queue_size=1)
        # self.pub_trajectory_index = rospy.Publisher('/train/trajectory_index', String, queue_size=1)
        #self.pub_teleop_vel = rospy.Publisher('/train/mobile_base/commands/velocity', Twist, queue_size=1)
        #self.pub_image = rospy.Publisher('/train/image', Image, queue_size=1)

    def cb_odom0(self, msg):

        self.odom0 = msg

    def cb_odom1(self, msg):

        self.odom1 = msg

    def cb_image0(self, msg):

        self.image0 = msg

    def cb_image1(self, msg):

        self.image1 = msg
        
    def cb_image2(self, msg):

        self.image2 = msg
    '''
    def load_image_seq(self):
        

        #print(rgbseq.shape)
        #return self.rgbseq
    '''

    def _on_loop(self, policy):
        """
        Logical Loop
        """
        self._timer.tick()
        rospy.sleep(0.1)
        #print(type(self.image))
        #<class 'sensor_msgs.msg._Image.Image'>

        #read image and convert to rgb format
        img0 = ros_numpy.numpify(self.image0)
        img1 = ros_numpy.numpify(self.image1)
        img2 = ros_numpy.numpify(self.image2)
        img = np.concatenate((img2,img1,img0),axis=1)
        #print(img.shape)
        rgbimage = PIL.Image.fromarray(img.astype(np.uint8))
        rgbimage = rgbimage.convert('RGB')
        rgbimage = np.asarray(rgbimage).astype(np.float32)#(112,336,3)
        #print(rgbimage.shape)
        rgbarray = np.moveaxis(rgbimage,-1,0)#(3,112,336)
        rgbarray = rgbarray.reshape(1,3,112,336)
        print(rgbarray.shape)
        seq_len = 5 
        frame_space = 3    
        '''
        if self.img_queue.qsize() == 0:
            print('initialize queue')
            for i in range (seq_len*frame_space-1):
                self.img_queue.put(rgbarray)          
        self.img_queue.put(rgbarray)#((1,seq_len*frame_space,3,224,224))
        #print(self.img_queue.qsize())#seq_len*frame_space

        seq_list = []
        for elem in list(self.img_queue.queue):
            seq_list.append(elem)
            #print(self.img_queue.qsize(),len(seq_list))

        for i in range ((seq_len-2)*frame_space+1):
            #seq_init = rgbarray
            if i==0:
                rgbseq = seq_list[0]
            elif i%frame_space==0:
                rgbseq = np.concatenate((rgbseq,seq_list[i]),axis=1)
                #print(i,rgbseq.shape,self.img_queue.qsize())

        rgbseq = np.concatenate((rgbseq,rgbarray),axis=1)
        self.img_queue.get()
        print(self.img_queue.qsize(),rgbseq.shape)
        '''
        '''
        length_of_seq = 6
        #frame_list = [0,2,5,8,10]
        frame_list = [0,1,2,3,4,5]
        #frame_list = [0]
        if self.img_queue.qsize() == 0:
            print('initialize queue')
            for i in range (length_of_seq-1):
                self.img_queue.put(rgbarray)          
        self.img_queue.put(rgbarray)#((1,seq_len*frame_space,3,224,224))
        #print(self.img_queue.qsize())#seq_len*frame_space

        seq_list = []
        for elem in list(self.img_queue.queue):
            seq_list.append(elem)
        #print(self.img_queue.qsize(),len(seq_list))

        for i in range (length_of_seq):
            #seq_init = rgbarray
            if i==0: 
                rgbseq = seq_list[0]
            elif i in frame_list:
                rgbseq = np.concatenate((rgbseq,seq_list[i]),axis=1)
                #print(i,rgbseq.shape,self.img_queue.qsize())

        #rgbseq = np.concatenate((rgbseq,rgbarray),axis=1)
        self.img_queue.get()
        #print(self.img_queue.qsize(),rgbseq.shape)

        '''

        #rgb_tensor = torch.from_numpy(rgbseq).cuda()
        rgb_tensor = torch.from_numpy(rgbarray).cuda()




        pred_control = policy.predict_control(rgb_tensor)

        self.tele_twist.linear.x = pred_control[0,0,0]
        self.tele_twist.angular.z = pred_control[0,0,1]


        #pose = np.array((self.odom0.pose.pose.position.x,self.odom0.pose.pose.position.y)
        #robot0 = np.asarray((round(self.odom0.pose.pose.position.x,3),round(self.odom0.pose.pose.position.y,3)))
        robot1 = np.asarray((round(self.odom1.pose.pose.position.x,3),round(self.odom1.pose.pose.position.y,3)))
        #dist1 = np.norm(robot0,robot1)
        print(self.tele_twist.linear.x,self.tele_twist.angular.z,robot1)

        # publish control
        self.control_pub.publish(self.tele_twist)      
        

    def execute(self, policy):
        while True:
            #self.cb_image()
            self._on_loop(policy)
            self._rate.sleep()


# wrapper for fire to get command arguments
def run_wrapper(rate=10):
    rospy.init_node("torch_controller", anonymous=True)

    controller = Controller(rate)
    policy = Policy()
    controller.execute(policy)

def main():
    fire.Fire(run_wrapper)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user! Bye Bye!')

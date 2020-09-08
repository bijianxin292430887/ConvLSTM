from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import scipy.misc
import rospy
import torch
from ConvLSTM_model import Net02,Net04,Net05,Net06,Net0402,Net14,Net015


class Policy(object):
    def __init__(self, vis=False):
        
        self.model = None
        self.input_frame = 5
        self.vis = vis
        if self.vis:
            self.count = 0
            self.fig = plt.figure()

    def load_model(self):
        #print('load model')
        model = Net04().cuda()
        model.load_state_dict(torch.load('data_skill_corner/model_004'))
        self.model = model

    def predict_control(self, image):
        #print(type(image))
        #print(image.shape)
        if self.model == None:
            self.load_model()
        
        #rospy.sleep(0.5)
        pred_control= self.model(image)
        pred_control=pred_control.cpu().detach().numpy()
        #print (pred_control.shape)#(1,1,2)
        '''
        pred_control1,pred_control2= self.model(image)
        pred_control1=pred_control1.cpu().detach().numpy()
        pred_control2=pred_control2.cpu().detach().numpy()
        pred_control = np.concatenate((pred_control1,pred_control2),axis = 1).reshape(1,1,2)
        '''
        return pred_control
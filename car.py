#!/usr/local/bin/python
import pdb
import sys,os,platform,matplotlib
#
# import matplotlib.animation as animation
# import matplotlib.pyplot as plt


import sys
import params
sys.path.append("..")
import scipy.io
import numpy as np
from scipy.integrate import odeint
from numpy import cos, sin, tan, arctan2, sqrt, sign, diag,arctan
from numpy.linalg import norm
current_path = os.path.dirname(os.path.abspath(__file__))
from PIL import Image
from math import pi
from scipy.optimize import newton_krylov, fsolve, anderson, broyden1, broyden2



car_colors = {'blue', 'gray', 'white', 'yellow', 'brown',
        'white1','green', 'white_cross', 'cyan', 'red1', 'orange'}
#car_colors = {'blue', 'gray', 'black', 'white', 'yellow', 'brown', 'white1','green', 'white_cross', 'cyan', 'red1', 'orange', 'white2'}
car_figs = dict()
for color in car_colors:
    car_figs[color] = current_path + '/imglib/cars/' + color + '_car.png'




class KinematicCar():
    '''Kinematic car class

    '''
    def __init__(self,
                 init_state=[0, 0, 0, 0],
                 segment = None,
                 dir = None,
                 goal = None,
                 length = 3,  # length of vehicle in pixels
                 acc_max = 9.81*0.4,  # maximum acceleration of vehicle
                 acc_min = -9.81*0.8,  # maximum deceleration of vehicle
                 steer_max = 0.8,  # maximum steering input in radians
                 steer_min = -0.8,  # minimum steering input in radians
                 vmax = 30,  # maximum velocity
                 color = 'blue'):
        if color not in car_colors:
            raise Exception("This car color doesn't exist!")
        self._length = length
        self._vmax = vmax
        self.acc_range = (acc_min, acc_max)
        self.steer_range = (steer_min, steer_max)
        self.wait_time = 0
        self.state = np.array(init_state, dtype='float')
        self.color = color
#        self.new_unpause = False
#        self.new_pause = False
        # extended state required for Bastian's primitive computation
        self.fig = Image.open(car_figs[color])
        self.segment = segment
        self.dir = dir
        self.goal = goal
        self.crossing_traj = None
        self.baseline_time = None
        self.contract_time = None

    def state_dot(self, state,time, acc,steer):

        """
        This function defines the system dynamics

        Inputs
        acc: acceleration input
        steer: steering input
        """
        # if already at maximum speed, can't no longer accelerate
        if state[2] >= self._vmax and acc>0:
            v_dot = 0
        elif state[2]<=0 and acc<-1e-3:
            v_dot = -state[2]
        else:
            v_dot = np.clip(acc, self.acc_range[0], self.acc_range[1])
        theta_dot = state[2] / self._length * tan(np.clip(steer, self.steer_range[0], self.steer_range[1]))
        x_dot = state[2] * cos(state[3])
        y_dot = state[2] * sin(state[3])
        dstate = [x_dot, y_dot, v_dot, theta_dot ]
        return dstate


    def next(self, inputs, dt):
        """
        next is a function that updates the current position of the car when inputs are applied for a duration of dt

        Inputs:
        inputs: acceleration and steering inputs
        dt: integration time

        Outputs:
        None - the states of the car will get updated
        """
        acc, steer = inputs
        # take only the real part of the solution
        if dt>0.1:
            self.state = odeint(self.state_dot, self.state, t=(0, dt), args=(acc,steer))[1]
        else:
            self.state = self.state + np.array(self.state_dot(self.state,0,acc,steer))*dt
        if self.segment==1:
            self.wait_time += dt

def find_corner_coordinates(x_state_center_before, y_state_center_before, x_desired, y_desired, theta, square_fig):
    """
    This function takes an image and an angle then computes
    the coordinates of the corner (observe that vertical axis here is flipped).
    If we'd like to put the point specfied by (x_state_center_before, y_state_center_before) at (x_desired, y_desired),
    this function returns the coordinates of the lower left corner of the new image
    """
    w, h = square_fig.size
    theta = -theta
    if abs(w - h) > 1:
        print('Warning: Figure has to be square! Otherwise, clipping or unexpected behavior may occur')
#        warnings.warn("Warning: Figure has to be square! Otherwise, clipping or unexpected behavior may occur")


    R = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    x_corner_center_before, y_corner_center_before = -w/2., -h/2. # lower left corner before rotation
    x_corner_center_after, y_corner_center_after = -w/2., -h/2. # doesn't change since figure size remains unchanged

    x_state_center_after, y_state_center_after = R.dot(np.array([[x_state_center_before], [y_state_center_before]])) # relative coordinates after rotation by theta

    x_state_corner_after = x_state_center_after - x_corner_center_after
    y_state_corner_after = y_state_center_after - y_corner_center_after

    # x_corner_unknown + x_state_corner_after = x_desired
    x_corner_unknown = int(x_desired - x_state_center_after + x_corner_center_after)
    # y_corner_unknown + y_state_corner_after = y_desired
    y_corner_unknown = int(y_desired - y_state_center_after + y_corner_center_after)
    return x_corner_unknown, y_corner_unknown

offset = [-1.3,0.0]
def draw_cars(vehicles, background):
    for vehicle in vehicles:
        x, y, v, theta  = vehicle.state
        x=params.map_scale_factor*(x+offset[0]*cos(theta)-offset[1]*sin(theta))
        y=params.map_scale_factor*(y+offset[0]*sin(theta)+offset[1]*cos(theta))
        # convert angle to degrees and positive counter-clockwise
        theta_d = -theta/np.pi * 180
        vehicle_fig = vehicle.fig
        w_orig, h_orig = vehicle_fig.size
        # set expand=True so as to disable cropping of output image
        vehicle_fig = vehicle_fig.rotate(theta_d, expand = False)
        scaled_vehicle_fig_size  =  tuple([int(params.car_scale_factor * i) for i in vehicle_fig.size])
        # rescale car
        vehicle_fig = vehicle_fig.resize(scaled_vehicle_fig_size, Image.ANTIALIAS)
        # at (full scale) the relative coordinates of the center of the rear axle w.r.t. the center of the figure is center_to_axle_dist
        x_corner, y_corner = find_corner_coordinates(-params.car_scale_factor * params.center_to_axle_dist, 0, x, y, theta, vehicle_fig)
        background.paste(vehicle_fig, (x_corner, y_corner), vehicle_fig)

def straight_con(x,dir,acc_range,steer_range,xf=None):
    alpha = 3
    amin,amax = acc_range
    if dir == 'N':
        des_theta = pi/2
        x_des = params.X1
        delta_y = -x[0]+x_des

    elif dir =='S':
        des_theta = -pi/2
        x_des = params.X0
        delta_y = x[0]-x_des
    elif dir =='E':
        des_theta = 0
        y_des = params.Y0
        delta_y = x[1]-y_des
    elif dir=='W':
        des_theta = -pi
        y_des = params.Y1
        delta_y = y_des-x[1]
    delta_theta = x[3]-des_theta
    while delta_theta>pi:
        delta_theta-=2*pi
    while delta_theta<-pi:
        delta_theta+=2*pi
    Kv = 1
    Ky = 1
    Ktheta = 5
    vdes = 5
    acc = -Kv*(x[2]-vdes)
    if xf is None:
        acc = np.clip(acc,amin,amax)
    else:
        if dir=='N':
            h = xf[1]-x[1]+(np.sign(xf[2])*xf[2]**2-np.sign(x[2])*x[2]**2)/2/(-amin)
        elif dir=='S':
            h = x[1]-xf[1]+(np.sign(xf[2])*xf[2]**2-np.sign(x[2])*x[2]**2)/2/(-amin)
        elif dir=='E':
            h = xf[0]-x[0]+(np.sign(xf[2])*xf[2]**2-np.sign(x[2])*x[2]**2)/2/(-amin)
        elif dir=='W':
            h = x[0]-xf[0]+(np.sign(xf[2])*xf[2]**2-np.sign(x[2])*x[2]**2)/2/(-amin)
        Lfh = xf[2]-x[2]
        Lgh = min(x[2]/amin,-1e-3)

        accmax = (-alpha*h-Lfh)/Lgh
        accmax = max(accmax,amin)
        acc = np.clip(acc,amin,accmax)
    steer = np.clip((-Ky*delta_y-Ktheta*delta_theta)/(abs(x[2]+0.5)),steer_range[0],steer_range[1])
    u = [acc,steer]
    return u
def turning_con(x,dir1,dir2,L,acc_range,steer_range):
    RL = params.RL
    RR = params.RR
    if dir1 =='N':
        if dir2 =='L':
            pivot = np.array([params.X1-RL,params.Y1-RL])
            if x[1]<pivot[1]:
                des_theta = pi/2
                delta_y = -x[0]+params.X1
                steer0 = 0
            else:
                des_theta = arctan2(x[1]-pivot[1],x[0]-pivot[0])+pi/2
                delta_y = RL-norm(x[0:2]-pivot)
                steer0 = arctan(L/RL)
        elif dir2 =='R':
            pivot = np.array([params.X1+RR,params.Y0-RR])
            if x[1]<pivot[1]:
                des_theta = pi/2
                delta_y = -x[0]+params.X1
                steer0 = 0
            else:
                des_theta = arctan2(x[1]-pivot[1],x[0]-pivot[0])-pi/2
                delta_y = norm(x[0:2]-pivot)-RR
                steer0 = -arctan(L/RR)
    elif dir1 =='S':
        if dir2 =='L':
            pivot = np.array([params.X0+RL,params.Y0+RL])
            if x[1]>pivot[1]:
                des_theta = -pi/2
                delta_y = x[0]-params.X0
                steer0 = 0
            else:
                des_theta = arctan2(x[1]-pivot[1],x[0]-pivot[0])+pi/2
                delta_y = RL-norm(x[0:2]-pivot)
                steer0 = arctan(L/RL)
        elif dir2 =='R':
            pivot = np.array([params.X0-RR,params.Y1+RR])
            if x[1]>pivot[1]:
                des_theta = -pi/2
                delta_y = x[0]-params.X0
                steer0 = 0
            else:
                des_theta = arctan2(x[1]-pivot[1],x[0]-pivot[0])-pi/2
                delta_y = norm(x[0:2]-pivot)-RR
                steer0 = -arctan(L/RR)
    elif dir1 == 'E':
        if dir2 =='L':
            pivot = np.array([params.X1-RL,params.Y0+RL])
            if x[0]<pivot[0]:
                des_theta = 0
                delta_y = x[1]-params.Y0
                steer0 = 0
            else:
                des_theta = arctan2(x[1]-pivot[1],x[0]-pivot[0])+pi/2
                delta_y = RL-norm(x[0:2]-pivot)
                steer0 = arctan(L/RL)
        elif dir2 =='R':
            pivot = np.array([params.X0-RR,params.Y0-RR])
            if x[0]<pivot[0]:
                des_theta = 0
                delta_y = x[1]-params.Y0
                steer0 = 0
            else:
                des_theta = arctan2(x[1]-pivot[1],x[0]-pivot[0])-pi/2
                delta_y = norm(x[0:2]-pivot)-RR
                steer0 = -arctan(L/RR)
    elif dir1 == 'W':
        if dir2 =='L':
            pivot = np.array([params.X0+RL,params.Y1-RL])
            if x[0]>pivot[0]:
                des_theta = -pi
                delta_y = params.Y1-x[1]
                steer0 = 0
            else:
                des_theta = arctan2(x[1]-pivot[1],x[0]-pivot[0])+pi/2
                delta_y = RL-norm(x[0:2]-pivot)
                steer0 = arctan(L/RL)
        elif dir2 =='R':
            pivot = np.array([params.X1+RR,params.Y1+RR])
            if x[0]>pivot[0]:
                des_theta = -pi
                delta_y = params.Y1-x[1]
                steer0 = 0
            else:
                des_theta = arctan2(x[1]-pivot[1],x[0]-pivot[0])-pi/2
                delta_y = norm(x[0:2]-pivot)-RR
                steer0 = -arctan(L/RR)

    delta_theta = x[3]-des_theta

    while delta_theta>pi:
        delta_theta-=2*pi
    while delta_theta<-pi:
        delta_theta+=2*pi
    Kv = 1
    Ky = 1
    Ktheta = 5
    vdes = 5
    acc = np.clip(-Kv*(x[2]-vdes),acc_range[0],acc_range[1])
    steer = np.clip(steer0+(-Ky*delta_y-Ktheta*delta_theta)/(abs(x[2]+0.5)),steer_range[0],steer_range[1])
    u = [acc,steer]
    return u


# TESTING
# x0 = np.array([params.X1+1,0,0,pi/2-0.1])
# veh = KinematicCar(x0)
# veh_set = [veh]
# intersection_fig = current_path + '/imglib/intersection_stop1.png'
# intersection = Image.open(intersection_fig)
# background = Image.open(intersection_fig)
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1]) # get rid of white border
# plt.axis('off')
# ts = 0.05
# def animate(frame_idx,veh_set): # update animation by dt
#     global background
#     ax.clear()
#     for veh in veh_set:
#         u = turning_con(veh.state,'N','L',veh._length)
#         veh.next(u,ts)
#     draw_cars(veh_set, background)
#     the_intersection = [ax.imshow(background, origin="lower")]
#     background.close()
#     background = Image.open(intersection_fig)
#     return the_intersection
# ani = animation.FuncAnimation(fig, animate, fargs=(veh_set,),frames=int(5/ts), interval=ts*1000, blit=True, repeat=False)
# plt.show()
# pdb.set_trace()

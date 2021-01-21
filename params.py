# figure params
length = 788
car_width = 399
axle_to_back = 160
center_to_axle_dist = length/2. - axle_to_back
front_to_axle = length-axle_to_back
pedestrian_scale_factor = 0.25
car_scale_factor = 0.13 # scale for when L = 50?
map_scale_factor = 33
num_subprims = 5
theta_compensate = 5
X0 = 20.3
X1 = 23.9
Y0 = 14
Y1 = 17.6
RL = 7.4
RR = 3.8
pixel_to_meter_scale_factor = 50 / 1.5 # 1.5 meters ~ 50 pixels
bdry=[X0-RR,X1+RR,Y0-RR,Y1+RR]
# constants
g = 9.81 # gravitational constant
#g = pixel_to_meter_scale_factor * 9.81 # gravitational constant

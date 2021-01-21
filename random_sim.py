#!/usr/local/bin/python
from car import *
from intersection import *
import pdb


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
output = 'movie.mp4'
# output = None
import sys,os,platform,matplotlib


if not output is None:
    matplotlib.use('Agg')

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# intersection_fig = current_path + '/imglib/intersection_stop1.png'
# intersection = Image.open(intersection_fig)

background = Image.open(intersection_fig)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1]) # get rid of white border
plt.axis('off')
intersection = Intersection(4,15,1)

def animate(frame_idx,intersection): # update animation by dt
    print(frame_idx*intersection.ts)
    global background
    ax.clear()
    intersection.step()
    if len(intersection.veh_set)<intersection.Nmax:
        if with_probability(intersection.pv):
            intersection.random_spawn_car()
    draw_cars(intersection.veh_set, background)
    the_intersection = [ax.imshow(background, origin="lower")]
    background.close()
    background = Image.open(intersection_fig)
    return the_intersection
anim = animation.FuncAnimation(fig, animate, fargs=(intersection,),frames=int(20/intersection.ts), interval=intersection.ts*1000, blit=True, repeat=False)
if not output is None:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    anim_name = output
    anim.save(anim_name,writer=writer)

plt.show()

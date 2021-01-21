# The Intersection Component
# Tung M. Phan
# California Institute of Technology
# August 6, 2018
#
import os, random
from PIL import Image
from car import *
from itertools import permutations
current_path = os.path.dirname(os.path.abspath(__file__))
#intersection_fig = dir_path + "/components/imglib/intersection_states/intersection_lights.png"
intersection_fig = current_path + '/imglib/intersection_stop1.png'
intersection = Image.open(intersection_fig)
v0 = 5
def dubin_f_x(x,con):
    h = 1e-6
    dudx = np.zeros([4,2])
    dudx[0] = (con(x+np.array([h,0,0,0]))-con(x-np.array([h,0,0,0])))/2/h
    dudx[1] = (con(x+np.array([0,h,0,0]))-con(x-np.array([0,h,0,0])))/2/h
    dudx[2] = (con(x+np.array([0,0,h,0]))-con(x-np.array([0,0,h,0])))/2/h
    dudx[3] = (con(x+np.array([0,0,0,h]))-con(x-np.array([0,0,0,h])))/2/h
    # pdb.set_trace()
    ja = np.concatenate((np.array([[0, 0, np.cos(x[3]), -x[2]*np.sin(x[3])],[0, 0, np.sin(x[3]), x[2]*np.cos(x[3])]])   ,dudx.transpose()))
    return ja

def generate_traj(x,con,stop_crit,ts=0.05):
    t = 0
    tt = []
    xx = []
    uu = []
    # Q = np.identity(4)
    # QQ = []

    while not stop_crit(x,t):
        # QQ.append(Q)
        u = con(x)
        xdot = np.array([x[2]*cos(x[3]),x[2]*sin(x[3]),u[0],u[1]])
        # ja = dubin_f_x(x,con)
        x = x + xdot*ts
        # Q = Q + np.matmul(ja,Q)*ts
        t = t + ts
        tt.append(t)
        xx.append(x)
        uu.append(u)
    return tt,xx,uu

def get_background():
    return Image.open(intersection_fig)
def with_probability(P=1):
    return np.random.uniform() <= P
def check_compatibility(traj1, traj2,t01=0,t02=0,L=4,W=2.2):
    if traj1 is None or traj2 is None:
        return False

    T=min(len(traj1)+t01,len(traj2)+t02)
    corners = np.array([[L/2,W/2],[L/2,-W/2],[-L/2,W/2],[-L/2,-W/2]])
    collision = False

    for t in range(min(t01,t02),T):
        if collision:
            break
        t1 = max(0,t-t01)
        t2 = max(0,t-t02)
        dx = traj1[t1][0:2]-traj2[t2][0:2]
        if norm(dx)<L:
            theta1 = traj1[t1][3]
            theta2 = traj2[t2][3]
            Trans1 = np.array([[cos(theta1),-sin(theta1)],[sin(theta1),cos(theta1)]])
            delta_x1 = np.dot(Trans1,corners.T).T+dx
            Trans2 = np.array([[cos(theta2),sin(theta2)],[-sin(theta2),cos(theta2)]])
            delta_x2 = np.dot(Trans2,delta_x1.T).T
            for i in range(delta_x2.shape[0]):
                if abs(delta_x2[i][0])<L/2 and abs(delta_x2[i][1])<W/2:
                    collision = True
                    # if t01>30:
                    #     pdb.set_trace()
                    break
    return collision
class Intersection():
    def __init__(self,N0,Nmax,pv,ts=0.05):
        self.Nmax = Nmax
        self.pv = pv
        self.veh_set = []
        self.ts = ts
        self.N0 = min(N0,4)
        self.t = 0
        while len(self.veh_set)<self.N0:
            self.random_spawn_car()
    def check_collision(self,dir):
        coll = False
        if dir == 'N':
            for veh in self.veh_set:
                if veh.segment==0 and veh.dir=='N' and veh.state[1]<5:
                    coll = True
                    break
        elif dir == 'S':
            for veh in self.veh_set:
                if veh.segment==0 and veh.dir=='S' and veh.state[1]>params.Y0+params.Y1-5:
                    coll = True
                    break
        elif dir == 'E':
            for veh in self.veh_set:
                if veh.segment==0 and veh.dir=='E' and veh.state[0]<5:
                    coll = True
                    break
        elif dir == 'W':
            for veh in self.veh_set:
                if veh.segment==0 and veh.dir=='W' and veh.state[0]>params.X0+params.X1-5:
                    coll = True
                    break
        return coll

    def get_xf(self,veh):
        offset = 4
        if veh.segment==2:
            return None
        else:
            if veh.dir=='N':
                if veh.segment==0 or veh.segment==1:
                    xf = np.array([params.X1,params.Y0-params.RR-1.6,0,pi/2])
                elif veh.segment==3:
                    xf = np.array([params.X1,100,0,pi/2])
                for veh1 in self.veh_set:
                    if veh1 !=veh and veh1.dir=='N':
                        if veh1.state[1]>veh.state[1] and veh1.state[1]<xf[1]+offset:
                            xf = veh1.state+np.array([0,-offset,0,0])
            elif veh.dir=='S':
                if veh.segment==0 or veh.segment==1:
                    xf = np.array([params.X0,params.Y1+params.RR+1.6,0,-pi/2])
                elif veh.segment==3:
                    xf = np.array([params.X0,-100,0,pi/2])
                for veh1 in self.veh_set:
                    if veh1 !=veh and veh1.dir=='S':
                        if veh1.state[1]<veh.state[1] and veh1.state[1]>xf[1]-offset:
                            xf = veh1.state+np.array([0,offset,0,0])
            elif veh.dir=='E':
                if veh.segment==0 or veh.segment==1:
                    xf = np.array([params.X0-params.RR-1.6,params.Y0,0,0])
                elif veh.segment==3:
                    xf = np.array([100,params.Y0,0,0])
                for veh1 in self.veh_set:
                    if veh1 !=veh and veh1.dir=='E':
                        if veh1.state[0]>veh.state[0] and veh1.state[0]<xf[0]+offset:
                            xf = veh1.state+np.array([-offset,0,0,0])
            elif veh.dir=='W':
                if veh.segment==0 or veh.segment==1:
                    xf = np.array([params.X1+params.RR+1.6,params.Y1,0,-pi])
                elif veh.segment==3:
                    xf = np.array([-100,params.Y1,0,-pi])
                for veh1 in self.veh_set:
                    if veh1 !=veh and veh1.dir=='W':
                        if veh1.state[0]<veh.state[0] and veh1.state[0]>xf[0]-offset:
                            xf = veh1.state+np.array([offset,0,0,0])
        return xf
    def random_spawn_car(self):
        dirs = ['N','S','E','W']
        goals = ['F','L','R']
        # goals = ['R']
        while dirs:
            dir = random.choice(dirs)
            if self.check_collision(dir):
                dirs.remove(dir)
            else:
                if dir=='N':
                    init_state=[params.X1, 0, v0, pi/2]
                elif dir=='S':
                    init_state=[params.X0, params.Y0+params.Y1, v0, -pi/2]
                elif dir=='E':
                    init_state=[0, params.Y0, v0, 0]
                elif dir=='W':
                    init_state=[params.X0+params.X1, params.Y1, v0, pi]
                veh = KinematicCar(
                init_state=init_state,
                segment = 0,
                dir = dir,
                goal = random.choice(goals),
                color = random.choice(list(car_colors)))
                self.veh_set.append(veh)
                return True
        return False
    def get_traj_pred(self,veh):
        if veh.segment==0:
            return None
        elif veh.segment==1 or veh.segment==2:
            if veh.goal=='F':
                con = lambda x:straight_con(x,veh.dir,veh.acc_range,veh.steer_range,None)
                if veh.dir=='N':
                    stop_crit = lambda x,t:x[1]>params.bdry[3] or t>10
                elif veh.dir=='S':
                    stop_crit = lambda x,t:x[1]<params.bdry[2] or t>10
                elif veh.dir=='E':
                    stop_crit = lambda x,t:x[0]>params.bdry[1] or t>10
                elif veh.dir=='W':
                    stop_crit = lambda x,t:x[0]<params.bdry[0] or t>10
            else:
                con = lambda x:turning_con(x,veh.dir,veh.goal,veh._length,veh.acc_range,veh.steer_range)
                if (veh.dir=='E' and veh.goal=='L') or (veh.dir=='W' and veh.goal=='R'): #north
                    stop_crit = lambda x,t:x[1]>params.bdry[3] or t>10
                elif (veh.dir=='E' and veh.goal=='R') or (veh.dir=='W' and veh.goal=='L'): #north
                    stop_crit = lambda x,t:x[1]<params.bdry[2] or t>10
                elif (veh.dir=='N' and veh.goal=='R') or (veh.dir=='S' and veh.goal=='L'): #north
                    stop_crit = lambda x,t:x[0]>params.bdry[1] or t>10
                elif (veh.dir=='N' and veh.goal=='L') or (veh.dir=='S' and veh.goal=='R'): #north
                    stop_crit = lambda x,t:x[0]<params.bdry[0] or t>10
        elif veh.segment==3:
            con = lambda x:straight_con(x,veh.dir,veh.acc_range,veh.steer_range,None)
            if veh.dir=='N':
                stop_crit = lambda x,t:x[1]>params.Y0+params.Y1 or t>10
            elif veh.dir=='S':
                stop_crit = lambda x,t:x[1]<0 or t>10
            elif veh.dir=='E':
                stop_crit = lambda x,t:x[0]>params.X0+params.X1 or t>10
            elif veh.dir=='W':
                stop_crit = lambda x,t:x[0]<0 or t>10

        tt,xx,uu=generate_traj(veh.state,con,stop_crit,self.ts)
        return tt,xx

    def get_baseline_time(self):
        seg1_veh_set = [veh for veh in self.veh_set if veh.segment==1]
        seg2_veh_set = [veh for veh in self.veh_set if veh.segment==2]
        t_wait = np.array([veh.wait_time for veh in seg1_veh_set])
        priority_idx = np.argsort(-t_wait).tolist()
        time_table = self.get_time_table(seg1_veh_set,seg2_veh_set,priority_idx)
        for i in range(0,len(seg1_veh_set)):
            seg1_veh_set[i].baseline_time = self.t+time_table[i]*self.ts
    def contract_controller(self):
        seg1_veh_set = [veh for veh in self.veh_set if veh.segment==1]
        seg2_veh_set = [veh for veh in self.veh_set if veh.segment==2]
        orders = list(permutations(range(len(seg1_veh_set))))
        baseline = [veh.baseline_time for veh in seg1_veh_set]
        opt_time_table = ((np.array(baseline)-self.t)/self.ts).astype(int)
        ub = opt_time_table.copy()
        for order in orders:
            time_table = self.get_time_table(seg1_veh_set,seg2_veh_set,list(order),ub)
            if not time_table is None and sum(time_table)<sum(opt_time_table):
                opt_time_table=time_table.copy()
        # if len(seg1_veh_set)>2:
        #     pdb.set_trace()
        for i in range(0,len(seg1_veh_set)):
            seg1_veh_set[i].contract_time = self.t+opt_time_table[i]*self.ts

    def get_time_table(self,seg1_veh_set,seg2_veh_set,order,ub=None):
        if ub is None:
            ub = np.ones(len(seg1_veh_set),dtype=int)*10000
        time_table = np.zeros(len(seg1_veh_set),dtype=int)
        t=0
        Failure = False
        for i in range(0,len(order)):
            idx = order[i]
            veh = seg1_veh_set[idx]
            feasible = False
            if (time_table>ub).any():
                Failure = True
                break
            while not feasible:
                feasible = True
                for veh1 in seg2_veh_set:
                    if not feasible:
                        break
                    while check_compatibility(veh.crossing_traj, veh1.crossing_traj,t01=t,t02=0):
                        feasible = False
                        t+=1
                        if t>1000:
                            pdb.set_trace()
                for j in range(0,i):
                    if not feasible:
                        break
                    veh1 = seg1_veh_set[order[j]]

                    while check_compatibility(veh.crossing_traj, veh1.crossing_traj,t01=t,t02=time_table[order[j]]):
                        feasible = False
                        t+=1
                        if t>1000:
                            pdb.set_trace()


            time_table[idx] = t
        if not Failure:
            return time_table
        else:
            return None
    # def intersection_coordination(self):

    def step(self):
        self.t+=self.ts
        new_queued_veh = False
        queued_veh_left = False
        for veh in self.veh_set:
            if veh.segment==0 and veh.dir=='N' and veh.state[1]>params.bdry[2]-1.8 and veh.state[2]<0.1:
                veh.segment = 1
                new_queued_veh = True
                tt,xx=self.get_traj_pred(veh)
                veh.crossing_traj = xx
            elif veh.segment==0 and veh.dir=='S' and veh.state[1]<params.bdry[3]+1.8 and veh.state[2]<0.1:
                veh.segment = 1
                new_queued_veh = True
                tt,xx=self.get_traj_pred(veh)
                veh.crossing_traj = xx
            elif veh.segment==0 and veh.dir=='E' and veh.state[0]>params.bdry[0]-1.8 and veh.state[2]<0.1:
                veh.segment = 1
                new_queued_veh = True
                tt,xx=self.get_traj_pred(veh)
                veh.crossing_traj = xx
            elif veh.segment==0 and veh.dir=='W' and veh.state[0]<params.bdry[1]+1.8 and veh.state[2]<0.1:
                veh.segment = 1
                new_queued_veh = True
                tt,xx=self.get_traj_pred(veh)
                veh.crossing_traj = xx
            if veh.segment==2:
                tt,xx=self.get_traj_pred(veh)
                veh.crossing_traj = xx
                if veh.dir=='N':
                    if veh.goal=='F' and veh.state[1]> params.bdry[3]:
                        veh.segment = 3
                    elif veh.goal=='L' and veh.state[0]< params.bdry[0]:
                        veh.segment = 3
                        veh.dir = 'W'
                    elif veh.goal=='R' and veh.state[0]> params.bdry[1]:
                        veh.segment = 3
                        veh.dir = 'E'
                elif veh.dir=='S':
                    if veh.goal=='F' and veh.state[1]<params.bdry[2]:
                        veh.segment = 3
                    elif veh.goal=='L' and veh.state[0]> params.bdry[1]:
                        veh.segment = 3
                        veh.dir = 'E'
                    elif veh.goal=='R' and veh.state[0]< params.bdry[0]:
                        veh.segment = 3
                        veh.dir = 'W'
                elif veh.dir=='E':
                    if veh.goal=='F' and veh.state[0]>params.bdry[1]:
                        veh.segment = 3
                    elif veh.goal=='L' and veh.state[1]> params.bdry[3]:
                        veh.segment = 3
                        veh.dir = 'N'
                    elif veh.goal=='R' and veh.state[1]< params.bdry[2]:
                        veh.segment = 3
                        veh.dir = 'S'
                elif veh.dir=='W':
                    if veh.goal=='F' and veh.state[0]<params.bdry[0]:
                        veh.segment = 3
                    elif veh.goal=='L' and veh.state[1]< params.bdry[2]:
                        veh.segment = 3
                        veh.dir = 'S'
                    elif veh.goal=='R' and veh.state[1]> params.bdry[3]:
                        veh.segment = 3
                        veh.dir = 'N'

        if new_queued_veh:
            self.get_baseline_time()
            self.contract_controller()
        for veh in self.veh_set:
            if veh.segment==1 and self.t>=veh.baseline_time:
                veh.segment=2
                # queued_veh_left = True
        # if queued_veh_left or new_queued_veh:
        #
        # for veh in self.veh_set:
            if veh.segment==0 or veh.segment==1 or veh.segment==3 or veh.goal=='F':
                xf = self.get_xf(veh)
                u = straight_con(veh.state,veh.dir,veh.acc_range,veh.steer_range,xf)
            elif veh.segment==2 and veh.goal!='F':
                u = turning_con(veh.state,veh.dir,veh.goal,veh._length,veh.acc_range,veh.steer_range)
            # elif veh.segment==1:
            #     u = [-veh.state[2],0]
            veh.next(u,self.ts)
            if veh.segment==3:
                if veh.dir=='N' and veh.state[1]>params.Y0+params.Y1:
                    self.veh_set.remove(veh)
                elif veh.dir=='S' and veh.state[1]<0:
                    self.veh_set.remove(veh)
                elif veh.dir=='E' and veh.state[0]>params.X0+params.X1:
                    self.veh_set.remove(veh)
                elif veh.dir=='W' and veh.state[1]<0:
                    self.veh_set.remove(veh)

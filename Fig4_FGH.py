#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:41:40 2019

@author: mbeiran
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import modules4 as md

import f_manuscript as fm

#import lib_rnns as lr
import tools_MF as tm
from mpl_toolkits.mplot3d import Axes3D
import funcs_Sphere as fs
from matplotlib import cm
import cartopy.crs as ccrs

rank = 3

dt = 10#ms
tau = 100#ms

alpha = dt/tau
std_noise_rec = np.sqrt(2*alpha)*0.1

input_size = 4
hidden_size = 1000
output_size = 1
size_f = np.sqrt(10)

#initial connectivity
sigma_mn = 0.85

trials_train = 500
trials_test = 100

R_on  = 1000//dt#500//dt

tss = np.array((800, 1550))
tss2 = np.array(( 1550,))#np.array(( 800,  1050, 1300, 1550))
tss22 = np.array(( 1550,))#2*np.array(( 800,  1050, 1300, 1550))

tss3 = np.linspace(800, 3000, 20)
tss32 = 1.5*np.linspace(800, 3000, 20)

N_steps = 5


Tss  = fs.gen_intervals(tss,N_steps)
Tss2 = fs.gen_intervals(tss2,N_steps)

Nt = 1100 
time = np.arange(Nt)

Nt2 = 1300
time2 = np.arange(Nt2)


# Parameters of task
SR_on = 60
factor = 1
dela = 150

# Colors
cls2 = fs.set_plot()

cls2[1,:] = cls2[2,:]
cls2[2,:] = cls2[4,:]
cls2[3,:] = cls2[5,:]

initial_h0 = False


cls = np.zeros((7,3))
cl11 = np.array((102, 153, 255))/255.
cl12 = np.array((53, 153, 53))/255.

cl21 = np.array((255, 204, 51))/255.
cl22 = np.array((204, 0, 0))/255.

cls[0,:] = 0.4*np.ones((3,))

cls[1,:] = cl11
cls[2,:] = 0.5*cl11+0.5*cl12
cls[3,:] = cl12

cls[4,:] = cl21
cls[5,:] = 0.5*cl21+0.5*cl22
cls[6,:] = cl22

CLL = np.zeros((8,3))
CLL[0,:] = np.array((150, 53, 47))/256.
CLL[1,:] = np.array((207, 84, 57))/256.
CLL[2,:] = np.array((212, 104, 81))/256.
CLL[3,:] = np.array((220, 135, 64))/256.

CLL[4,:] = np.array((78, 104, 172))/256.
CLL[5,:] = np.array((85, 131, 198))/256.
CLL[6,:] = np.array((102, 185, 224))/256.
CLL[7,:] = np.array((150, 198, 219))/256.
CLL2 = CLL

CLL[0,:] = np.array((203, 80, 71))/256.

CLL[3,:] = np.array((53, 153, 53))/256.
CLL[1,:] = 0.75*CLL[0,:] + 0.25*CLL[3,:]
CLL[2,:] = 0.25*CLL[0,:] + 0.75*CLL[3,:]

CLL[4,:] = np.array((255, 204, 51))/256.
CLL[7,:] = np.array((71, 89, 156))/256.
CLL[5,:] = 0.75*CLL[4,:] + 0.25*CLL[7,:]
CLL[6,:] = 0.1*CLL[4,:] + 0.9*CLL[7,:]

#%%
#12 and 15 is quite good

only_perf = True
T0s_lr2 = []
T0s_lr2_sD = []
T0s_lr2_lD = []

T0s_lr22 = []
T0s_lr22_sD = []
T0s_lr22_lD = []

T0s_fr2 = []
T0s_fr2_sD = []
T0s_fr2_lD = []

T0s_fr22 = []
T0s_fr22_sD = []
T0s_fr22_lD = []       
 

#%%
# Load networks
tR = 1
net_low_all = md.OptimizedLowRankRNN(input_size, hidden_size, output_size, std_noise_rec, alpha,
                             rank=rank, train_wi = True, train_wo = True, train_h0=True, train_wrec = True)
net_low_all.load_state_dict(torch.load("TrainedNets/MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_2int_traininpall.pt", map_location='cpu'))

net_low_fr = md.FullRankRNN(input_size, hidden_size, output_size, 0.1*std_noise_rec, alpha, 
                                       train_wi = True, train_wo = True, train_h0=True)
net_low_fr.load_state_dict(torch.load("TrainedNets/MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_fr.pt", map_location='cpu'))

#%%

ntr = 50

M_pos, N_pos, corr_pos, I_pos, O_pos, J_pos = fs.get_SVDweights_CSG(net_low_all, rank=rank)

fig, ax, Traj_fs, Time_fs, iT_fs, time_fs = fs.run_randtraj_fs_all(M_pos*np.sqrt(hidden_size), N_pos*np.sqrt(hidden_size), T = 200, dt = 0.1, trajs=ntr)
R = np.mean(np.sqrt(Traj_fs[0,-100:,:]**2+Traj_fs[1,-100:,:]**2+Traj_fs[2,-100:,:]**2))
ths = np.linspace(0, 2.1*np.pi)
#plt.plot(R*np.cos(ths), 0*ths, R*np.sin(ths), c='k', lw=0.5)
#plt.plot(R*np.cos(ths),  R*np.sin(ths), 0*ths, c='k', lw=0.5)
#plt.plot(0*np.cos(ths),R*np.cos(ths),  R*np.sin(ths), c='k', lw=0.5)

# plt.savefig('Plots/MWG2_pretrained_random_tr'+str(tR)+'_June.pdf')


#%%
rs = np.linspace(0.5, 2.2, 100)
theta = np.linspace(-np.pi/2, 1.5*np.pi, 100)
phi = np.linspace(0,np.pi, 100)

M, N, corr_pos, I_pos, O_pos, J_pos = fs.get_SVDweights_CSG(net_low_all, rank=rank)        
sigma_mn = np.dot(M.T,N)
sigma_m = np.diag(np.dot(M.T,M))


vf_mf, Q_mf, vf_fs, Q_fs = fs.give_fields(rs, theta, phi, M*np.sqrt(hidden_size), N*np.sqrt(hidden_size), sigma_mn, sigma_m, verbose=True)
Q_man, R_man, U_man, V_man, Q_man_fs, R_man_fs, U_man_fs, V_man_fs = fs.give_manif(theta, phi, rs, Q_mf, vf_mf, Q_fs, vf_fs)

s = 0.1
vf_I_fs, Q_I_fs = fs.give_fields_Inp(rs, theta, phi, M*np.sqrt(hidden_size), N*np.sqrt(hidden_size), I_pos[3,:]*s, verbose=True)
Q_I_man, R_I_man, U_I_man, V_I_man, Q_I_man_fs, R_I_man_fs, U_I_man_fs, V_I_man_fs = fs.give_manif(theta, phi, rs,  Q_I_fs, vf_I_fs, Q_I_fs, vf_I_fs)


#%%
ntr = 40
np.random.seed(21)

M_pos, N_pos, corr_pos, I_pos, O_pos, J_pos = fs.get_SVDweights_CSG(net_low_all, rank=rank)

fig, ax, Traj_fs, Time_fs, iT_fs, time_fs = fs.run_randtraj_fs_all(M_pos*np.sqrt(hidden_size), N_pos*np.sqrt(hidden_size), T = 150, dt = 0.1, trajs=ntr)
R = np.mean(np.sqrt(Traj_fs[0,-100:,:]**2+Traj_fs[1,-100:,:]**2+Traj_fs[2,-100:,:]**2))
ths = np.linspace(0, 2.1*np.pi)

for it, th in enumerate(theta):   
    X = R_man[it, :]*np.sin(phi)*np.cos(th)
    Y = R_man[it, :]*np.sin(phi)*np.sin(th)
    Z = R_man[it, :]*np.cos(phi)
    ax.plot(X,Y,Z,c='k', alpha=0.1)


for ip, p in enumerate(phi):   
    X = R_man[:, ip]*np.sin(p)*np.cos(theta)
    Y = R_man[:, ip]*np.sin(p)*np.sin(theta)
    Z = R_man[:, ip]*np.cos(p)
    ax.plot(X,Y,Z,c='k', alpha=0.1)

ax.set_xticks([-1, 1])
ax.set_yticks([-1, 1])
ax.set_zticks([-1, 1])
# plt.savefig('Plots/MWG2_noInp_Manifold.pdf')
# plt.savefig('Plots/FM_MWG_dist2manif_proof1.pdf') 
#%%
fig = plt.figure()
ax = fig.add_subplot(111)
SS = 40
T0 = 1000
DT = 100
mean_= 0

iTF = 500

t_list = []
for itr in range(ntr):
    plt.plot(time_fs[:iTF]*DT, Traj_fs[0,:iTF,itr], color=0.6*np.ones(3))   
    plt.scatter(DT*time_fs[0], Traj_fs[0,0,itr], s = 30, marker='^', edgecolor='k', color=0.7*np.ones(3), zorder=4)
    plt.scatter(DT*time_fs[iTF-1], Traj_fs[0,iTF-1,itr], s = 50,  edgecolor='k', color=0.7*np.ones(3), zorder=4)
    calc = Traj_fs[0,:iTF,itr] - Traj_fs[0,-1,itr]#/rkaps[0,1,itr]
    calc = calc/calc[0]
    ix_ = np.argmin(np.abs(calc-0.5))
    mean_ += DT*time_fs[ix_]/ntr
    t_list.append(DT*time_fs[ix_])
    #print(DT*time[ix_])
    #plt.plot(calc)
    

plt.plot([mean_, mean_], [-1.5, 1.5], c='C0', lw=4, alpha=0.3)
ax.set_ylim([-1.5, 1.5])
ax.set_xlim([-200, 5200])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('time (ms)')
ax.set_ylabel('read out')
print(mean_)
# plt.savefig('Plots/FM_MWG_dist2manif_proof2.pdf') 

#%%

fig = plt.figure(figsize= [2.2*1.5, 0.3*2*1.5])
ax = fig.add_subplot(111)
plt.hist(t_list, 30, alpha=0.3, color='k')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylabel('')
ax.set_yticks([])
ax.set_xlim([-200, 5200])
ax.set_xticks([0, 2000, 4000])
# ax.set_xticklabels('')
# plt.savefig('Plots/FM_MWG_dist2manif_proof2_top.pdf') 

#%%

def to_polar2(Traj):
    x = Traj[0,:,:]
    y = Traj[1,:,:]
    z = Traj[2,:,:]
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2) 
    elev = 0.5*np.pi-np.arctan(z/np.sqrt(XsqPlusYsq)) 
    maskE=  elev>3.141592
    elev[maskE] = 0
    az = np.arctan2(y,x)
    
    maskAz =  az<-np.pi/2
    az[maskAz] += 2*np.pi
    return(r, az, elev)

R_fs, Az_fs, Elev_fs = to_polar2(Traj_fs)



theta2 = np.linspace(-np.pi/2, 1.5*np.pi, 100)
phi2 = np.linspace(0,np.pi, 100)
dist = np.zeros_like(Traj_fs[0,:iTF,:])
Speed = np.zeros_like(Traj_fs[0,:iTF,:])
for it, ti in enumerate(time_fs[:iTF]):
    for itr in range(ntr):
        iTH = np.argmin(np.abs(theta2-Az_fs[it, itr]))
        iPH = np.argmin(np.abs(phi2-Elev_fs[it,itr]))
        dist[it, itr] = R_fs[it, itr]-R_man[iTH,iPH]
        Speed[it, itr] = Q_man[iTH,iPH]


#%%
# matplotlib.rcParams['pdf.fonttype'] = 3
fig = plt.figure()
ax = fig.add_subplot(111)
SS = 40
T0 = 1000
DT = 100
mean_= 0

iTF = 500
t_list = []
for itr in range(ntr):
    plt.plot(time_fs[:iTF]*DT, -dist[:,itr], color=0.6*np.ones(3))    
    calc = dist[:,itr] - dist[-1,itr]#/rkaps[0,1,itr]
    calc = calc/calc[0]
    ix_ = np.argmin(np.abs(calc-0.5))
    mean_ += DT*time_fs[ix_]/ntr
    plt.scatter(DT*time_fs[0], -(dist[0,itr]), s = 30, marker='^', edgecolor='k', color=0.7*np.ones(3), zorder=4)
    plt.scatter(DT*time_fs[iTF-1], -(dist[iTF-1,itr]), s = 50,  edgecolor='k', color=0.7*np.ones(3), zorder=4)
    #print(DT*time[ix_])
    #plt.plot(calc)
    t_list.append(DT*time_fs[ix_])
plt.plot([mean_, mean_], [-1.5, 1.5], c='C0', lw=4, alpha=0.3)
ax.set_ylim([-0.1, 1.2])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('time (ms)')
ax.set_ylabel('distance to manifold')
#plt.plot([np.mean(t_list), np.mean(t_list)], [0, 100], c='r', lw=2, alpha=0.2)
ax.set_xlim([-200, 5200])
print(mean_)
# plt.savefig('Plots/FM_MWG_dist2manif_proof3.pdf') 

#%%

fig = plt.figure(figsize= [2.2*1.5, 0.3*2*1.5])
ax = fig.add_subplot(111)
plt.hist(t_list, 5, alpha=0.3, color='k')
#plt.plot([np.mean(t_list), np.mean(t_list)], [0, 100], c='r', lw=2, alpha=0.2)
ax.set_xlim([-200, 5200])
ax.set_xticks([0, 2000, 4000])
ax.set_xticklabels(['', '', ''])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.set_ylabel('')
ax.set_yticks([])
# plt.savefig('Plots/FM_MWG_dist2manif_proof3_top.pdf') 

#%%
dela = 200
R_onN = 500

tss2_ = np.array(( 800, 1050, 1300, 1550))#np.array(( 800,  1050, 1300, 1550))
tss22_ = np.array(( 1550,))


Inps_lr, Trajs_lr, Inps_lr2, Trajs_lr2 = fm.plot_output_MWG(net_low_all, tss2_, tss22_, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=dela, give_inps=True,plot=False, plot_sev = np.array((1,1)), R_on = R_onN)

#%%
def to_polar(Traj):
    x = Traj[:,0,:]
    y = Traj[:,1,:]
    z = Traj[:,2,:]
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2) 
    elev = 0.5*np.pi-np.arctan(z/np.sqrt(XsqPlusYsq)) 
    maskE=  elev>3.141592
    elev[maskE] = 0
    az = np.arctan2(y,x)
    
    maskAz =  az<-np.pi/2
    az[maskAz] += 2*np.pi
    return(r, az, elev)
#%%
tss2 = tss2_
tss22 = tss22_
fig = plt.figure()
ax= fig.add_subplot(111, projection='3d', azim=103, elev=1)
for it, th in enumerate(theta):   
    X = R_man[it, :]*np.sin(phi)*np.cos(th)
    Y = R_man[it, :]*np.sin(phi)*np.sin(th)
    Z = R_man[it, :]*np.cos(phi)
    ax.plot(X,Y,Z,c='k', alpha=0.2)


for ip, p in enumerate(phi):   
    X = R_man[:, ip]*np.sin(p)*np.cos(theta)
    Y = R_man[:, ip]*np.sin(p)*np.sin(theta)
    Z = R_man[:, ip]*np.cos(p)
    ax.plot(X,Y,Z,c='k', alpha=0.2)

TrajR, TrajAz, TrajElev = to_polar(Trajs_lr)
X1 = 500
X2 = 1030
SS = 40
for xx in range(len(tss2)):
    X1_0 = 500-(tss2[xx]//10-tss2[-1]//10) # inp1
    X1_1 = 500-(tss2[xx]//10-tss2[-1]//10)+tss2[xx]//10 # inp2
    X1_2 = 500-(tss2[xx]//10-tss2[-1]//10)+tss2[xx]//10+200 # Set
    X1_3 = 500-(tss2[xx]//10-tss2[-1]//10)+2*tss2[xx]//10+200 # Go
    X = TrajR[:,xx]*np.sin(TrajElev[:,xx])*np.cos(TrajAz[:,xx])
    Y = TrajR[:,xx]*np.sin(TrajElev[:,xx])*np.sin(TrajAz[:,xx])
    Z = TrajR[:,xx]*np.cos(TrajElev[:,xx])
    ax.plot(X[X1:X1_3],Y[X1:X1_3],Z[X1:X1_3], color=CLL2[2*xx,:], lw=2.5)
    ax.scatter(X[X1_0],Y[X1_0],Z[X1_0], marker='^', color=CLL[2*xx,:], edgecolor='k', zorder=2, s=SS)
    ax.scatter(X[X1_1],Y[X1_1],Z[X1_1], color=CLL2[2*xx,:], edgecolor='k', zorder=3, s=SS)
    ax.scatter(X[X1_2],Y[X1_2],Z[X1_2], marker='^', color=CLL2[2*xx,:], edgecolor='k', zorder=4, s=SS)
    ax.scatter(X[X1_3],Y[X1_3],Z[X1_3], color=CLL2[2*xx,:], edgecolor='k', zorder=5, s=SS)
    
    #ax.scatter(X[0],Y[0],Z[0], color='k', edgecolor='w')
    
    
    
ax.set_xticks([-1,0, 1])
ax.set_yticks([-1,0, 1])
ax.set_zticks([-1,0, 1])
ax.set_xlabel('$\kappa_1$')
ax.set_ylabel('$\kappa_2$')
ax.set_zlabel('$\kappa_3$')
ax.dist=6
ax.axis('off')
#plt.savefig('Plots/MWG2_noInp_ManifoldTrajs.pdf')

#%%
tss2 = tss2_
tss22 = tss22_
fig = plt.figure()
ax= fig.add_subplot(111, projection='3d', azim=103, elev=1)
for it, th in enumerate(theta):   
    X = R_man[it, :]*np.sin(phi)*np.cos(th)
    Y = R_man[it, :]*np.sin(phi)*np.sin(th)
    Z = R_man[it, :]*np.cos(phi)
    ax.plot(X,Y,Z,c='k', alpha=0.2)


for ip, p in enumerate(phi):   
    X = R_man[:, ip]*np.sin(p)*np.cos(theta)
    Y = R_man[:, ip]*np.sin(p)*np.sin(theta)
    Z = R_man[:, ip]*np.cos(p)
    ax.plot(X,Y,Z,c='k', alpha=0.2)

TrajR, TrajAz, TrajElev = to_polar(Trajs_lr)
X1 = 500
X2 = 1030
SS = 40
# for xx in range(len(tss2)):
#     X1_0 = 500-(tss2[xx]//10-tss2[-1]//10) # inp1
#     X1_1 = 500-(tss2[xx]//10-tss2[-1]//10)+tss2[xx]//10 # inp2
#     X1_2 = 500-(tss2[xx]//10-tss2[-1]//10)+tss2[xx]//10+200 # Set
#     X1_3 = 500-(tss2[xx]//10-tss2[-1]//10)+2*tss2[xx]//10+200 # Go
#     X = TrajR[:,xx]*np.sin(TrajElev[:,xx])*np.cos(TrajAz[:,xx])
#     Y = TrajR[:,xx]*np.sin(TrajElev[:,xx])*np.sin(TrajAz[:,xx])
#     Z = TrajR[:,xx]*np.cos(TrajElev[:,xx])
#     ax.plot(X[X1:X1_3],Y[X1:X1_3],Z[X1:X1_3], color=CLL2[2*xx,:], lw=2.5)
#     ax.scatter(X[X1_0],Y[X1_0],Z[X1_0], marker='^', color=CLL[2*xx,:], edgecolor='k', zorder=2, s=SS)
#     ax.scatter(X[X1_1],Y[X1_1],Z[X1_1], color=CLL2[2*xx,:], edgecolor='k', zorder=3, s=SS)
#     ax.scatter(X[X1_2],Y[X1_2],Z[X1_2], marker='^', color=CLL2[2*xx,:], edgecolor='k', zorder=4, s=SS)
#     ax.scatter(X[X1_3],Y[X1_3],Z[X1_3], color=CLL2[2*xx,:], edgecolor='k', zorder=5, s=SS)
    
    #ax.scatter(X[0],Y[0],Z[0], color='k', edgecolor='w')
    
    
    
ax.set_xticks([-1,0, 1])
ax.set_yticks([-1,0, 1])
ax.set_zticks([-1,0, 1])
ax.set_xlabel('$\kappa_1$')
ax.set_ylabel('$\kappa_2$')
ax.set_zlabel('$\kappa_3$')
ax.dist=6
ax.axis('off')
plt.savefig('Plots/MWG2_noInp_ManifoldTrajs_0.pdf')



#%%
fig = plt.figure()
ax= fig.add_subplot(111, projection='3d', azim=103, elev=1)
for it, th in enumerate(theta):   
    X = R_man[it, :]*np.sin(phi)*np.cos(th)
    Y = R_man[it, :]*np.sin(phi)*np.sin(th)
    Z = R_man[it, :]*np.cos(phi)
    ax.plot(X,Y,Z,c='C0', alpha=0.2)
    X = R_I_man[it, :]*np.sin(phi)*np.cos(th)
    Y = R_I_man[it, :]*np.sin(phi)*np.sin(th)
    Z = R_I_man[it, :]*np.cos(phi)
    ax.plot(X,Y,Z,c='C1', alpha=0.2)


for ip, p in enumerate(phi):   
    X = R_man[:, ip]*np.sin(p)*np.cos(theta)
    Y = R_man[:, ip]*np.sin(p)*np.sin(theta)
    Z = R_man[:, ip]*np.cos(p)
    
    ax.plot(X,Y,Z,c='C0', alpha=0.2)
    
    X = R_I_man[:, ip]*np.sin(p)*np.cos(theta)
    Y = R_I_man[:, ip]*np.sin(p)*np.sin(theta)
    Z = R_I_man[:, ip]*np.cos(p)
    
    ax.plot(X,Y,Z,c='C1', alpha=0.2)
        
ax.set_xticks([-1,0, 1])
ax.set_yticks([-1,0, 1])
ax.set_zticks([-1,0, 1])
ax.set_xlabel('$\kappa_1$')
ax.set_ylabel('$\kappa_2$')
ax.set_zlabel('$\kappa_3$')
ax.dist=6
ax.axis('off')
plt.savefig('Fig4_G.pdf')


#%%
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)


TrajR, TrajAz, TrajElev = to_polar(Trajs_lr)

def find_R(TrajAz, TrajElev, R_man, tss2):
    TrajRM = np.zeros_like(TrajAz)
    for xx in range(len(tss2)):
        for it in range(np.shape(TrajR)[0]):
            Elev = TrajElev[it,xx]
            Az = TrajAz[it,xx]
            iP = np.argmin(np.abs(Elev-phi))
            iZ = np.argmin(np.abs(Az  - theta))
            TrajRM[it,xx] = R_man[iZ, iP]
    return(TrajRM)

TrajRM = find_R(TrajAz, TrajElev, R_man, tss2)
            
SS = 40

for xx in range(len(tss2)):
    X1 = 500
    X1_0 = 500-(tss2[xx]//10-tss2[-1]//10) # inp1
    X1_1 = 500-(tss2[xx]//10-tss2[-1]//10)+tss2[xx]//10 # inp2
    X1_2 = 500-(tss2[xx]//10-tss2[-1]//10)+tss2[xx]//10+200 # Set
    X1_3 = 500-(tss2[xx]//10-tss2[-1]//10)+2*tss2[xx]//10+200 # Go
    
    X2 = 1030
    T0_ = time[X1]+350
    #ax1.plot(TrajR[X1:X2,xx], '--', color=CLL2[2*xx,:], lw=3)
    #ax1.plot(TrajRM[X1:X2,xx], color=CLL2[2*xx,:], lw=3)
    ax1.plot(dt*(time[X1:X2]-T0_), TrajR[X1:X2,xx]-TrajRM[X1:X2,xx], color=CLL2[2*xx,:], lw=3)
    ax2.plot(dt*(time[X1:X2]-T0_), TrajElev[X1:X2,xx], color=CLL2[2*xx,:], lw=3)
    ax3.plot(dt*(time[X1:X2]-T0_), TrajAz[X1:X2,xx], color=CLL2[2*xx,:], lw=3)
    ax1.scatter(dt*(time[X1_0]-T0_), TrajR[X1_0,xx]-TrajRM[X1_0,xx], s=SS, edgecolor='k', color=CLL2[2*xx,:],zorder=3)
    ax1.scatter(dt*(time[X1_1]-T0_), TrajR[X1_1,xx]-TrajRM[X1_1,xx], s=SS, edgecolor='k', color=CLL2[2*xx,:],zorder=3)
    ax1.scatter(dt*(time[X1_2]-T0_), TrajR[X1_2,xx]-TrajRM[X1_2,xx], s=SS, edgecolor='k', color=CLL2[2*xx,:],zorder=3)
    ax1.scatter(dt*(time[X1_3]-T0_), TrajR[X1_3,xx]-TrajRM[X1_3,xx], s=SS, edgecolor='k', color=CLL2[2*xx,:],zorder=3)
    
    ax2.scatter(dt*(time[X1_0]-T0_), TrajElev[X1_0,xx], s=SS, edgecolor='k', color=CLL2[2*xx,:],zorder=3)
    ax2.scatter(dt*(time[X1_1]-T0_), TrajElev[X1_1,xx], s=SS, edgecolor='k', color=CLL2[2*xx,:],zorder=3)
    ax2.scatter(dt*(time[X1_2]-T0_), TrajElev[X1_2,xx], s=SS, edgecolor='k', color=CLL2[2*xx,:],zorder=3)
    ax2.scatter(dt*(time[X1_3]-T0_), TrajElev[X1_3,xx], s=SS, edgecolor='k', color=CLL2[2*xx,:],zorder=3)    
    
    ax3.scatter(dt*(time[X1_0]-T0_), TrajAz[X1_0,xx], s=SS, edgecolor='k', color=CLL2[2*xx,:],zorder=3)
    ax3.scatter(dt*(time[X1_1]-T0_), TrajAz[X1_1,xx], s=SS, edgecolor='k', color=CLL2[2*xx,:],zorder=3)
    ax3.scatter(dt*(time[X1_2]-T0_), TrajAz[X1_2,xx], s=SS, edgecolor='k', color=CLL2[2*xx,:],zorder=3)
    ax3.scatter(dt*(time[X1_3]-T0_), TrajAz[X1_3,xx], s=SS, edgecolor='k', color=CLL2[2*xx,:],zorder=3)    
    
ax1.set_ylabel('distance')
ax2.set_ylabel('elevation')
ax3.set_ylabel('azimuth')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.yaxis.set_ticks_position('left')
ax3.xaxis.set_ticks_position('bottom')

ax3.set_xlabel('time after Set (ms)')
ax2.set_xticks([-2000, 0, 2000])
ax1.set_xticks([-2000, 0, 2000])
ax2.set_xticklabels('')
ax1.set_xticklabels('')

#plt.savefig('Plots/MWG2_noInp_ManifoldCoords.pdf')

#%%
fig = plt.figure()
ax1 = fig.add_subplot(111)
def to_polar(Traj):
    x = Traj[:,0,:]
    y = Traj[:,1,:]
    z = Traj[:,2,:]
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2) 
    elev = 0.5*np.pi-np.arctan(z/np.sqrt(XsqPlusYsq)) 
    maskE=  elev>3.141592
    elev[maskE] = 0
    az = np.arctan2(y,x)
    maskAz =  az<-np.pi/2
    az[maskAz] += 2*np.pi
    return(r, az, elev)

TrajR, TrajAz, TrajElev = to_polar(Trajs_lr)

def find_R(TrajAz, TrajElev, R_man, tss2):
    TrajRM = np.zeros_like(TrajAz)
    for xx in range(len(tss2)):
        for it in range(np.shape(TrajR)[0]):
            Elev = TrajElev[it,xx]
            Az = TrajAz[it,xx]
            iP = np.argmin(np.abs(Elev-phi))
            iZ = np.argmin(np.abs(Az  - theta))
            TrajRM[it,xx] = R_man[iZ, iP]
    return(TrajRM)

TrajRM = find_R(TrajAz, TrajElev, R_man, tss2)
            
SS = 40

for xx in range(len(tss2)):
    X1 = 500
    X1_0 = 500-(tss2[xx]//10-tss2[-1]//10) # inp1
    X1_1 = 500-(tss2[xx]//10-tss2[-1]//10)+tss2[xx]//10 # inp2
    X1_2 = 500-(tss2[xx]//10-tss2[-1]//10)+tss2[xx]//10+200 # Set
    X1_3 = 500-(tss2[xx]//10-tss2[-1]//10)+2*tss2[xx]//10+200 # Go
    
    X2 = 1030
    T0_ = time[X1]+350
    #ax1.plot(TrajR[X1:X2,xx], '--', color=CLL2[2*xx,:], lw=3)
    #ax1.plot(TrajRM[X1:X2,xx], color=CLL2[2*xx,:], lw=3)
    ax1.plot(dt*(time[X1:X1_3]-T0_), TrajR[X1:X1_3,xx]-TrajRM[X1:X1_3,xx], color=CLL2[2*xx,:], lw=3)
    ax1.scatter(dt*(time[X1_0]-T0_), TrajR[X1_0,xx]-TrajRM[X1_0,xx], s=SS, marker='^', edgecolor='k', color=CLL2[2*xx,:],zorder=3)
    ax1.scatter(dt*(time[X1_1]-T0_), TrajR[X1_1,xx]-TrajRM[X1_1,xx], s=SS, edgecolor='k', color=CLL2[2*xx,:],zorder=3)
    ax1.scatter(dt*(time[X1_2]-T0_), TrajR[X1_2,xx]-TrajRM[X1_2,xx], s=SS, marker='^', edgecolor='k', color=CLL2[2*xx,:],zorder=3)
    ax1.scatter(dt*(time[X1_3]-T0_), TrajR[X1_3,xx]-TrajRM[X1_3,xx], s=SS, edgecolor='k', color=CLL2[2*xx,:],zorder=3)
    
ax1.fill_between([-3700, 2200],[-0.1, -0.1], [0.1, 0.1], alpha=0.1, color='k')
ax1.set_xlim([-3600, 2100])
ax1.set_ylabel('distance to manifold')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')

ax1.set_xlabel('time after Set (ms)')
ax1.set_xticks([-2000, 0, 2000])
ax1.set_ylim([-0.6, 0.6])
plt.savefig('Fig4_Finset.pdf')





#%%
def remove_noise(Inps_lr, Inps_lr2, comps = 20, tss2=tss2, tss22=tss22):
    # Remove noise
    X = np.dstack((Inps_lr, Inps_lr2))  
    X2 = np.moveaxis(X, 1, 0)
    Xp = np.reshape(X2, (np.shape(X2)[0], np.shape(X2)[1]*np.shape(X2)[2]))
    C = np.dot(Xp, Xp.T)
    
    uh, vh = np.linalg.eigh(C) 
    uh=uh[::-1]
    vh=vh[:, ::-1]
    
    sel = comps
    vS = vh[:,0:sel]
    Inps_lr_cl = np.zeros_like(Inps_lr)
    Inps_lr2_cl = np.zeros_like(Inps_lr2)
    
    for ix, xx in enumerate(tss2):
        Mat = (Inps_lr[:,:,ix].dot(vS)).dot(vS.T)
        Inps_lr_cl[:,:,ix] =  Mat
        
    for ix, xx in enumerate(tss22):
        Mat = (Inps_lr2[:,:,ix].dot(vS)).dot(vS.T)
        Inps_lr2_cl[:,:,ix] =  Mat
        
    # Calculate speed
    Sp = np.diff(Inps_lr_cl,axis=0)/dt
    Sp2 = np.diff(Inps_lr2_cl,axis=0)/dt
    
    Sp_norm =  np.sqrt(np.sum(Sp**2,1))
    Sp_norm2 =  np.sqrt(np.sum(Sp2**2,1))

    return(Inps_lr_cl, Inps_lr2_cl, Sp_norm, Sp_norm2)


#%%
# fig = plt.figure()
# ax1 = fig.add_subplot(111)

TrajR, TrajAz, TrajElev = to_polar(Trajs_lr)
TrajRM = find_R(TrajAz, TrajElev, R_man, tss2)

TrajR_I, TrajAz_I, TrajElev_I = to_polar(Trajs_lr2)
TrajRM_I = find_R(TrajAz_I, TrajElev_I, R_I_man, tss22_)
            
Inps_lr_cl, Inps_lr2_cl, Sp_norm, Sp_norm2 = remove_noise(Inps_lr[:,:,[-1,]], Inps_lr2[:,:,[0,]], tss2=[tss2[-1],], tss22=[tss22[0],])

#%%
fig = plt.figure()
#ax = fig.add_subplot(111)
ax = plt.axes(projection=ccrs.Robinson(central_longitude=149))
ax.gridlines(alpha=0.7)
lon = theta*180/np.pi+90
lat = (phi*180/np.pi)-90
Lon, Lat = np.meshgrid(lon , lat)
cb = ax.pcolormesh(Lon, -Lat, np.log10(Q_man_fs.T), cmap='gray', alpha=1., shading='auto',transform=ccrs.PlateCarree(), vmin = -2.5, vmax=-0.7)


#cbar = fig.colorbar(cb, ax=ax)
#cbar.set_ticks([-1, -2])
ax.streamplot(Lon, -Lat, U_man_fs.T, -V_man_fs.T, color='w', linewidth=1, density=0.6,transform=ccrs.PlateCarree())

for xx in range(len(tss2)):
    if xx>-1:#==len(tss2)-1:
        nt_az = TrajAz[:,xx]*180/np.pi + 90
        nt_el = TrajElev[:,xx]*180/np.pi -90
        idcs = np.arange(len(nt_az)-1)
        iXX = idcs[np.abs(np.diff(nt_az))>300]
        try:
            iXX=iXX[0]
        except:
            iXX = 0
        nt_az[iXX+1:] = nt_az[iXX+1:]+360
        
        st_Prod = 457
        end_Prod = 1100#457 + tss2[xx]//10
        #ax.plot(nt_az[st_Prod:end_Prod], -nt_el[st_Prod:end_Prod], color='w', lw=2, transform=ccrs.PlateCarree())
        
        #ax.plot(nt_az[st_Prod:end_Prod], -nt_el[st_Prod:end_Prod], color='w', lw=2.5, transform=ccrs.PlateCarree())
        ax.plot(nt_az[st_Prod:end_Prod], -nt_el[st_Prod:end_Prod], color=CLL2[xx*2,:], lw=2, transform=ccrs.PlateCarree())
        X1_0 = 500-(tss2[xx]//10-tss2[-1]//10) # inp1
        X1_1 = 500-(tss2[xx]//10-tss2[-1]//10)+tss2[xx]//10-1 # inp2
        X1_2 = 500-(tss2[xx]//10-tss2[-1]//10)+tss2[xx]//10+200 # Set
        X1_3 = 500-(tss2[xx]//10-tss2[-1]//10)+2*tss2[xx]//10+200 # Go
        X = TrajR[:,xx]*np.sin(TrajElev[:,xx])*np.cos(TrajAz[:,xx])
        Y = TrajR[:,xx]*np.sin(TrajElev[:,xx])*np.sin(TrajAz[:,xx])
        Z = TrajR[:,xx]*np.cos(TrajElev[:,xx])
        #ax.plot(X[X1:X1_3],Y[X1:X1_3],Z[X1:X1_3], color=CLL2[2*xx,:], lw=2.5)
        ax.scatter(nt_az[X1_0],-nt_el[X1_0], marker='^', color=CLL[2*xx,:], edgecolor='k', zorder=12, s=SS, transform=ccrs.PlateCarree())
        ax.scatter(nt_az[X1_1],-nt_el[X1_1], color=CLL2[2*xx,:], edgecolor='k', zorder=13, s=SS, transform=ccrs.PlateCarree())
        ax.scatter(nt_az[X1_2],-nt_el[X1_2], marker='^', color=CLL2[2*xx,:], edgecolor='k', zorder=14, s=SS, transform=ccrs.PlateCarree())
        ax.scatter(nt_az[X1_3],-nt_el[X1_3], color=CLL2[2*xx,:], edgecolor='k', zorder=15, s=SS, transform=ccrs.PlateCarree())
        
        
plt.savefig('Fig4_F.pdf')


#%%

TrajR, TrajAz, TrajElev = to_polar(Trajs_lr)


TrajR_I, TrajAz_I, TrajElev_I = to_polar(Trajs_lr2)

#%%
fig = plt.figure()
ax = plt.axes(projection=ccrs.Robinson(central_longitude=149))

ax.gridlines(alpha=0.7)

lon = theta*180/np.pi+90
lat = (phi*180/np.pi)-90
Lon, Lat = np.meshgrid(lon , lat)
# compute native map projection coordinates of lat/lon grid.
#x, y = map(Lon, Lat)
cb = ax.pcolormesh(Lon, -Lat, Q_man_fs.T-Q_I_man_fs.T, transform=ccrs.PlateCarree(), cmap = 'PiYG', vmin = -0.06, vmax=0.06)


for xx in range(len(tss2_)):
    if xx==len(tss2_)-1:
        nt_az = TrajAz[:,xx]*180/np.pi + 90
        nt_el = TrajElev[:,xx]*180/np.pi -90
        idcs = np.arange(len(nt_az)-1)
        iXX = idcs[np.abs(np.diff(nt_az))>300]
        try:
            iXX=iXX[0]
        except:
            iXX = 0
        nt_az[iXX+1:] = nt_az[iXX+1:]+360
        
        st_Prod = 457
        end_Prod = 1100#457 + tss2[xx]//10
        ax.plot(nt_az[st_Prod:end_Prod], -nt_el[st_Prod:end_Prod], color='w', lw=2.5, transform=ccrs.PlateCarree())
        ax.plot(nt_az[st_Prod:end_Prod], -nt_el[st_Prod:end_Prod], color=CLL[0,:], lw=2, transform=ccrs.PlateCarree())

for xx in range(len(tss22_)):
    if xx==0:
        nt_az = TrajAz_I[:,xx]*180/np.pi + 90
        nt_el = TrajElev_I[:,xx]*180/np.pi -90
        idcs = np.arange(len(nt_az)-1)
        iXX = idcs[np.abs(np.diff(nt_az))>300]
        try:
            iXX=iXX[0]
        except:
            iXX = 0
        nt_az[iXX+1:] = nt_az[iXX+1:]+360
        
        st_Prod = 457
        end_Prod = 1100#457 + tss2[xx]//10
        ax.plot(nt_az[st_Prod:end_Prod], -nt_el[st_Prod:end_Prod], color='w', lw=2.5, transform=ccrs.PlateCarree())
        ax.plot(nt_az[st_Prod:end_Prod], -nt_el[st_Prod:end_Prod], color=CLL[-1,:], lw=2, transform=ccrs.PlateCarree())


plt.savefig('Fig4_H.pdf')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:41:40 2019

@author: mbeiran
"""
import numpy as np
import matplotlib.pyplot as plt
import modules4 as md
import torch
#import lib_rnns as lr
#import tools_MF as tm
from mpl_toolkits.mplot3d import Axes3D
import funcs_Sphere as fs
from matplotlib import cm

rank = 3

dt = 10#ms
tau = 100#ms

alpha = dt/tau
std_noise_rec = np.sqrt(2*alpha)*0.1

input_size = 3
hidden_size = 1000
output_size = 1
size_f = np.sqrt(10)

#initial connectivity
sigma_mn = 0.85

trials_train = 500
trials_test = 100

R_on  = 1000//dt#500//dt

#%%
tss = np.array((800, 1550))
tss2 = np.array(( 800,  1050, 1300, 1550))
tss3 = np.linspace(500, 3000, 30)
N_steps = 5


Tss  = fs.gen_intervals(tss,N_steps)
Tss2 = fs.gen_intervals(tss2,N_steps)


#%%
Nt = 1000 #1000
time = np.arange(Nt)

# Parameters of task
SR_on = 60
factor = 1
dela = 150
repeat = 10#7 #20 number of examples

# Colors
cls2 = fs.set_plot()

cls2[1,:] = cls2[2,:]
cls2[2,:] = cls2[4,:]
cls2[3,:] = cls2[5,:]

initial_h0 = False

#%%
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

#New colors
cl11 = np.array((71, 89, 156))/255.#p.array((102, 153, 255))/255.
cl12 = np.array((53, 153, 53))/255.

cl21 = np.array((255, 204, 51))/255.
cl22 = np.array((203, 81, 71))/255.#np.array((204, 0, 0))/255.

# New colors April 2021
cls[3,:] = cl11#0.4*np.ones((3,))

cls[2,:] = cl21#0.5*cl11+0.5*cl12
cls[1,:] = cl12
cls[0,:] = cl22

        #%%
def plot_output_MWG(net_low_all, tss2, dt, rank=3, give_trajs=False, plot=True,
                    fr = False, t0s=False, gener=False, tss_ref= 0, dela = 150, only_perf=False):
    if fr==False:
        M_pre, N_pre, corr_pre, I_pre, O_pre, J_pre = fs.get_SVDweights_CSG(net_low_all, rank=rank)
    if plot:
        if only_perf:
            fig_width = 1.5*2.2 # width in inches
            fig_height = 0.8*1.5*2  # height in inches
            fig_size =  [fig_width,fig_height]
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(111)

        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
    CLL = cls
    if gener==True:
        evenly_spaced_interval = np.linspace(0, 1, len(tss2)+10)
        cls2 = [cm.viridis(x) for x in evenly_spaced_interval]
        CLL = cls2
    trials = 10
    Trajs = np.zeros((len(time), rank, len(tss2)))
    T0s_lr = []
    
    dtss = tss2[1]-tss2[0]
    for xx in range(len(tss2)):
        input_train, output_train, mask_train, ct_train, ct2_train, ct3_train = fs.create_inp_out_MWG(trials, Nt, tss2//dt, 
                                                                R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = True, delayF = dela, inp_size=3)
        outp, traj = net_low_all.forward(input_train, return_dynamics=True)
        outp = outp.detach().numpy()
        if fr==False:
            traj = traj.detach().numpy()
            
            mtraj  = np.mean(traj,0)
            k1_traj = M_pre[:,0].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,0]**2))
            k2_traj = M_pre[:,1].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,1]**2))
            k3_traj = M_pre[:,2].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,2]**2))
            Trajs[:,0,xx] = k1_traj[:-1]
            Trajs[:,1,xx] = k2_traj[:-1]
            Trajs[:,2,xx] = k3_traj[:-1]
        outp2 = np.copy(outp)
        
        outp3 = np.copy(outp2[:,1:,0])
         #outp3[np.diff(outp2[:,:,0])<0]=5.
        outp3[:,np.diff(low_pass(np.mean(outp2[:,:,0],0)))<0]=5.
        
        outp3[:, time[1:]*dt<4000] = 5. 
        outp2 = outp3
        if gener==False:
            tt0s = time[np.argmin(np.abs(outp2-0.35),1)]*dt-4000
        else:
            tt0s = time[np.argmin(np.abs(outp2-0.35),1)]*dt-4000-dela*dt
        T0s_lr.append(tt0s)
        avg_outp0 = np.mean(outp[:,:,0],0)#np.mean(outp3[:,:,0],0)
        if plot:
            if only_perf:
                if gener==False:
                    T0=4000
                else:
                    T0 = 5000
                if gener==False:
                    ax.plot(time*dt-T0, avg_outp0, color=CLL[xx,:], lw=2)    
                    ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color='k', alpha=0.5) #This is for the mask (in all four)
                    #ax.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color=CLL[xx,:], alpha=0.5)
                    #ax.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color=CLL[xx,:], alpha=0.5)
                else:
                    if np.min(np.abs(tss2[xx]-tss_ref))<0.5*dtss:
                        ax.plot(time*dt-T0, avg_outp0, color='k', lw=2)    
                        #ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color='k')
                        #ax.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k')
                        #ax.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color='k')
                    else:
                        ax.plot(time*dt-T0, avg_outp0, color=CLL[xx], lw=1., alpha=0.5)    
                        #ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color=CLL[xx])
                        #ax.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color=CLL[xx], alpha=0.5)
                        #ax.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color=CLL[xx], alpha=0.5)
                    
            else:
                if gener==False:
                    ax.plot(time*dt, avg_outp0, color=CLL[xx,:], lw=2)    
                    ax.plot(time*dt, output_train.detach().numpy()[0,:,0], '--', color=CLL[xx,:])
                    #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color=CLL[xx,:], alpha=0.5)
                    #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color=CLL[xx,:], alpha=0.5)
                else:
                    if np.min(np.abs(tss2[xx]-tss_ref))<0.5*dtss:
                        ax.plot(time*dt, avg_outp0, color='k', lw=2)    
                        #ax.plot(time*dt, output_train.detach().numpy()[0,:,0], '--', color='k')
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color='k')
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color='k')
                    else:
                        ax.plot(time*dt, avg_outp0, color=CLL[xx], lw=1., alpha=0.5)    
                        #ax.plot(time*dt, output_train.detach().numpy()[0,:,0], '--', color=CLL[xx])
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color=CLL[xx], alpha=0.5)
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color=CLL[xx], alpha=0.5)
                    
    if plot:                    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('time (ms)')
        plt.ylabel('read out')
        plt.ylim([-1, 1])
        plt.yticks([-0.5, 0.5])
    
    if only_perf:
        if gener==False:
            plt.xlim([-500, 2500])
        else:
            plt.xlim([-500, 4000])
        plt.ylim([-0.6, 0.6])
        
    if give_trajs==True:
        return(Trajs)
    if t0s==True:
        return(fig, ax, T0s_lr)
    else:
        return(fig, ax)

def plot_inputs_MWG(net_low_all, tss2, dt, rank=3, give_trajs=False, plot=True,
                    fr = False, t0s=False, gener=False, tss_ref= 0, dela = 150, only_perf=False):

    if plot:
        fig_width = 1.5*2.2 # width in inches
        fig_height = 0.8*1.5*2  # height in inches
        fig_size =  [fig_width,fig_height]
        fig = plt.figure(figsize=fig_size)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        
        
    CLL = cls
    if gener==True:
        evenly_spaced_interval = np.linspace(0, 1, len(tss2)+10)
        cls2 = [cm.viridis(x) for x in evenly_spaced_interval]
        CLL = cls2
    trials = 10

    
    dtss = tss2[1]-tss2[0]
    for xx in range(len(tss2)):
        input_train, output_train, mask_train, ct_train, ct2_train, ct3_train = fs.create_inp_out_MWG(trials, Nt, tss2//dt, 
                                                                R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = False, delayF = dela, inp_size=3)

        T0=2500
        if plot:

            if gener==False:
                ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k', alpha=0.5)
                ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color=CLL[xx,:], alpha=1, lw=2)
            else:
                if np.min(np.abs(tss2[xx]-tss_ref))<0.5*dtss:
                    ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k')
                    ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color='k', lw=2)
                else:
                    ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k', alpha=0.5)
                    ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color=CLL[xx], alpha=0.8, lw=1)
                

    if plot:                    
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')
        ax2.set_xlabel('time (ms)')
        ax1.set_ylabel('input 1')
        ax2.set_ylabel('input 2')
        ax1.set_ylim([-0.05, 1])
        ax2.set_ylim([-0.05, 1])
        if gener==False:
            ax2.set_xlim([-100, 1900])
            ax1.set_xlim([-100, 1900])
        else:
            ax2.set_xlim([-100, 3100])
            ax1.set_xlim([-100, 3100])
        ax1.set_xticks([0, 1000, 2000,3000])
        ax2.set_xticks([0,  1000, 2000, 3000])
        ax1.set_xticklabels(['','','',''])
        
    return(fig, ax1, ax2)

def filter_loss(loss, vals = 15):
    nloss = np.zeros_like(loss)
    nloss[0] = loss[0]
    i_s = 1
    count = 1
    for i in range(len(loss)-1):
        ip = i+1
        if ip//15 == ip/15.:
            #print(str(i_s) + '  - ' +str(i))
            nloss[count] = np.mean(loss[i_s:i])
            count += 1
            i_s = i
    nloss = nloss[0:count]
    return(nloss)

import math as m
def low_pass(x, ints=5):
    y = np.copy(x)
    for ix in range(len(x)):
        lmin = np.max((0, ix-ints))
        lmax = np.min((len(x), ix+ints))
        y[ix] = np.mean(x[lmin:lmax])
    return(y)

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
def give_vectors( sigma1, sigma2, sigma3, s_m1, s_m2, s_m3, s= 1, hidden_units = 1500, max_iter = 100, bn2 = 0.5):
    bigSigma = np.zeros((7,7)) #2*rank+input
    bigSigma[0,0] = s_m1
    bigSigma[1,1] = s_m2
    bigSigma[2,2] = s_m3
    bigSigma[3,3] = 1.
    bigSigma[4,4] = 1.
    bigSigma[5,5] = 1.
    
    bigSigma[6,6] = s**2
    bigSigma[0,3] = sigma1
    bigSigma[3,0] = sigma1
    
    bigSigma[1,4] = sigma2
    bigSigma[4,1] = sigma2
    
    bigSigma[2,5] = sigma3
    bigSigma[5,2] = sigma3
    
    bigSigma[3,4] = s*bn2
    bigSigma[4,3] = s*bn2
    
    stop = False
    ite=0
    while stop==False and ite<max_iter:
        if np.min(np.linalg.eigvals(bigSigma))<0:
            bigSigma[2,2] = 1.1*bigSigma[2,2]
            bigSigma[3,3] = 1.1*bigSigma[3,3]
            ite +=1
        else:
            bigSigma[2,2] = 1.1*bigSigma[2,2]
            bigSigma[3,3] = 1.1*bigSigma[3,3]
            stop =True
            
    mean = np.zeros(7)
    error0 = 10.
    for K in range(100):
        X = np.random.multivariate_normal(mean, bigSigma, hidden_units)
        
        empSig = np.dot(X.T,X)/hidden_units
        error = np.std(empSig-bigSigma)
        if error<error0:
            error0=error
            X_save = X
    X = X_save
    empSig = np.dot(X.T,X)/hidden_units
        
    M = X[:,0:3]
    N = X[:,3:6]
    I = X[:,6]
    return(M, N, I)

                


sigma1 = 0.8
sigma2 = 0.8
sigma3 = 0.8

s_m1 = 1.
s_m2 = 1.
s_m3 = 1.

Mnaive, Nnaive, Inaive = give_vectors( sigma1, sigma2, sigma3, s_m1, s_m2, s_m3, s= 1, hidden_units = hidden_size, max_iter = 100, bn2 = 0.5)

Is_naive = np.random.randn()*Mnaive[:,0]+np.random.randn()*Mnaive[:,1]+np.random.randn()*Mnaive[:,2]
Is_naive1 = np.random.randn()*Mnaive[:,0]+np.random.randn()*Mnaive[:,1]+np.random.randn()*Mnaive[:,2]
Is_naive2 = np.random.randn()*Mnaive[:,0]+np.random.randn()*Mnaive[:,1]+np.random.randn()*Mnaive[:,2]
Is_naive3 = np.random.randn()*Mnaive[:,0]+np.random.randn()*Mnaive[:,1]+np.random.randn()*Mnaive[:,2]

O_naive = np.random.randn()*Mnaive[:,0]+np.random.randn()*Mnaive[:,1]+np.random.randn()*Mnaive[:,2]
O_naive = O_naive[:,np.newaxis]

I_naive = np.vstack((Is_naive1, Is_naive2, Is_naive3))


#%%
trials = [10,] #10, 18


train_ = False
repeat = 10#20
only_perf = True
T0s_lr2 = []
T0s_fr2 = []
        
Evs_all = []
Evs_fr_all = []
P_all = []
P_fr_all= []
P_all2 = []
P_fr_all2= []    

for itR, tR in enumerate(trials): #10 #4
    if itR==0:
        A = np.load('TrainedNets/net_MWG'+str(tR+1)+'.npz')
        M = A['arr_0'] 
        N = A['arr_1'] 
        Is = A['arr_2']  
        Wo = A['arr_3']
        cond0 = A['arr_4']
        corrWo = 1
        Wo = Wo/hidden_size
        if len(np.shape(Wo))==1:
            Wo=Wo[:,np.newaxis]
        flow = M.dot(N.T).dot(np.tanh(cond0))-cond0
        
        dtype = torch.FloatTensor  
        mrec_i = M
        nrec_i = N
        mrec_I = torch.from_numpy(mrec_i).type(dtype)
        nrec_I = torch.from_numpy(nrec_i).type(dtype)
        inp_I = torch.from_numpy(Is.T).type(dtype)
        out_I = torch.from_numpy(Wo).type(dtype)
        h0_i = torch.from_numpy(cond0).type(dtype)
        
        mrec_naive_i = Mnaive/np.sqrt(hidden_size)
        nrec_naive_i = Nnaive/np.sqrt(hidden_size)
        mrec_naive_I = torch.from_numpy(mrec_i).type(dtype)
        nrec_naive_I = torch.from_numpy(nrec_i).type(dtype)
        inp_naive_I = torch.from_numpy(I_naive).type(dtype)
        out_naive_I = torch.from_numpy(O_naive/hidden_size).type(dtype)
    
        net_low_all = md.OptimizedLowRankRNN(input_size, hidden_size, output_size, std_noise_rec, alpha,
                                         rank=rank, train_wi = True, train_wo = True, train_h0=True, train_wrec = True, wi_init=inp_naive_I, 
                                         wo_init=out_naive_I, m_init=mrec_naive_I, n_init=nrec_naive_I)
        
        net_low_fr = md.FullRankRNN(input_size, hidden_size, output_size, 0.1*std_noise_rec, alpha, 
                                                   train_wi = True, train_wo = True, train_h0=True)
    print('Repeat '+str(tR))
    print('Training all')
    net_low_all.load_state_dict(torch.load("TrainedNets/MWG_rank"+str(rank)+"_rep_"+str(tR)+"_2int_trainall_June.pt", map_location='cpu'))
    fig, ax, t0s_lr = plot_output_MWG(net_low_all, tss2, dt, t0s=True,only_perf=True)
    #plt.savefig('Plots/MWG2_predesigned_trainedInpsAll_'+str(tR)+'_figure_June.pdf')  
    plt.show()
    
    print('Training full rank')
    net_low_fr.load_state_dict(torch.load("TrainedNets/MWG_rank"+str(rank)+"_rep_"+str(tR)+"_fr.pt", map_location='cpu'))
    fig, ax, t0s_fr = plot_output_MWG(net_low_fr, tss2, dt, fr=True, t0s=True, only_perf=True)
    #plt.savefig('Plots/MWG2_predesigned_trainedFullR_'+str(tR)+'_figure_June.pdf')  
    plt.show()
    #%
    fig, ax = fs.plot_output_MWG2(net_low_all, net_low_fr, tss2, dt, cls, time, rank=3, give_trajs=False, plot=True,
                    fr = False, t0s=False, gener=False, tss_ref= 0, dela = 150)
    #plt.savefig('Plots/FM_Fig1_D_June.pdf')  
    #%    
    net_low_all_nonoise = md.OptimizedLowRankRNN(input_size, hidden_size, output_size, 0.*std_noise_rec, alpha,
                                     rank=rank, train_wi = True, train_wo = True, train_h0=True, train_wrec = True, m_init=net_low_all.m, n_init = net_low_all.n, 
                                     wo_init=net_low_all.wo, h0_init = net_low_all.h0, wi_init=net_low_all.wi)
    
    net_low_fr_nonoise = md.FullRankRNN(input_size, hidden_size, output_size, 0*std_noise_rec, alpha, 
                                               train_wi = True, train_wo = True, train_h0=True, wrec_init=net_low_fr.wrec, 
                                     wo_init=net_low_fr.wo, h0_init = net_low_fr.h0, wi_init=net_low_fr.wi)
    
    traj, traj_fr = fs.give_traj_MWG2(net_low_all_nonoise, net_low_fr_nonoise, tss2, dt, cls, time, rank=3, give_trajs=False, plot=True,
                    fr = False, t0s=False, gener=False, tss_ref= 0, dela = 150)
    #%
    # neuron is the last index
    C=0 
    Sta = 422
    End = Sta+230
    for i in range(np.shape(traj)[0]):
        TT = traj[i,Sta:End,:]
        C += np.dot(TT.T,TT)
    C_fr=0
    for i in range(np.shape(traj)[0]):
        TT = traj_fr[i,Sta:End,:]
        C_fr += np.dot(TT.T,TT)
    
    cls2 = fs.set_plot()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ev = np.linalg.eigvalsh(C)
    ev = 100*ev[::-1]/np.sum(ev)
    P = np.sum(ev)**2/np.sum(ev**2)
    ev_fr = np.linalg.eigvalsh(C_fr)
    ev_fr = 100*ev_fr[::-1]/np.sum(ev_fr)
    P_fr = np.sum(ev_fr)**2/np.sum(ev_fr**2)
    
    
            # neuron is the last index
    C2=0 
    Sta = 100
    End = 422+100
    for i in range(np.shape(traj)[0]):
        TT = traj[i,Sta:End,:]#-np.mean(traj[i,Sta:End,:],0)
        C2 += np.dot(TT.T,TT)
    C_fr2=0
    for i in range(np.shape(traj)[0]):
        TT = traj_fr[i,Sta:End,:]#-np.mean(traj_fr[i,Sta:End,:],0)
        C_fr2 += np.dot(TT.T,TT)
    
    cls2 = fs.set_plot()
    ev2 = np.linalg.eigvalsh(C2)
    ev2 = 100*ev2[::-1]/np.sum(ev2)
    P2 = np.sum(ev2)**2/np.sum(ev2**2)
    ev_fr2 = np.linalg.eigvalsh(C_fr2)
    ev_fr2 = 100*ev_fr2[::-1]/np.sum(ev_fr2)
    P_fr2 = np.sum(ev_fr2)**2/np.sum(ev_fr2**2)
    
    
    xa1 = np.arange(1000)+1
    plt.plot(xa1, np.cumsum(ev_fr), '--', c='k', label='full rank', lw=2)
    plt.scatter(xa1, np.cumsum(ev_fr), facecolor=0.5*np.ones(3), edgecolor='k', s=40, zorder=3)

    plt.plot(xa1, np.cumsum(ev), c='k', label='rank three', lw=2)
    plt.scatter(xa1, np.cumsum(ev), c='k', s=30)
    

    plt.xlim([0.5,7.5])
    xa = np.linspace(1, 1000)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.ylabel('cumulative variance (%)')
    plt.xlabel('# principal component')
    plt.xticks([1, 2, 3, 4, 5, 6, 7])
    plt.legend()
    
            
    Evs_all.append(ev) 
    Evs_fr_all.append(ev_fr) 
    P_all.append(P) 
    P_fr_all.append(P_fr)
    P_all2.append(P2) 
    P_fr_all2.append(P_fr2)    
    
    
    #%
    
    c1 = 0.9*np.ones(3)
    c2 = 0.5*np.ones(3)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(tss2, tss2, '--k')
    for xx in range(len(tss2)):
        plt.scatter(tss2[xx], tss2[xx], marker='s',s= 80, color=cls[xx,:], edgecolor='k', zorder=4)
    for xx in range(len(tss2)):
        if xx==0:
            plt.scatter(tss2[xx]*np.ones(len(t0s_fr[xx])), t0s_fr[xx]/0.85, s=30, color=c1, edgecolor=c2, label='full rank')
            plt.scatter(tss2[xx]*np.ones(len(t0s_lr[xx])), t0s_lr[xx]/0.85, s=30, color=c2, edgecolor='k', label='rank three')
        else:
            plt.scatter(tss2[xx]*np.ones(len(t0s_lr[xx])), t0s_lr[xx]/0.85, s=30, color=c2, edgecolor='k')
            plt.scatter(tss2[xx]*np.ones(len(t0s_fr[xx])), t0s_fr[xx]/0.85, s=30,edgecolor=c2,color=c1)                
    ax.set_xlabel('$t_s$ (ms)')
    ax.set_ylabel('$t_p$ (ms)')
    ax.set_xticks([1000, 1250, 1500])
    ax.set_yticks([1000, 1250, 1500])
    
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.plot(tss2, tss2, '--k')
    plt.legend()
    #plt.savefig('Plots/MWG2_performance_lrvsfr_tr'+str(tR)+'_June.pdf')  
    #plt.savefig('Plots/FM_Fig1_G_June.pdf')  
    plt.show()
    
    #%
    print('Generalization low-rank')
    #Generalization low rank
    do_delay =False
    net_low_all_lessnoise = md.OptimizedLowRankRNN(input_size, hidden_size, output_size, 0.5*std_noise_rec, alpha,
                                     rank=rank, train_wi = True, train_wo = True, train_h0=True, train_wrec = True, m_init=net_low_all.m, n_init = net_low_all.n, 
                                     wo_init=net_low_all.wo, h0_init = net_low_all.h0, wi_init=net_low_all.wi)
    
    
    fig, ax, t0s_lr2 = plot_output_MWG(net_low_all_lessnoise, tss3, dt, t0s=True, gener=True, tss_ref = tss2, dela=100)
    #plt.savefig('Plots/MWG2_predesigned_trainedInpsAll_'+str(tR)+'_gener_June.pdf')  
    plt.show()
    T0s_lr2.append(t0s_lr2)

    
    # Generalization full rank
    print('generalization full-rank')
    fig, ax, t0s_fr2 = plot_output_MWG(net_low_fr, tss3, dt, fr=True, t0s=True, gener=True, tss_ref = tss2, dela=100)
    #plt.savefig('Plots/MWG2_predesigned_trainedFullR_'+str(tR)+'_gener_June.pdf')  
    plt.show()
    T0s_fr2.append(t0s_fr2)


    #%
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(tss3, tss3, '--k')
    for xx in range(len(tss3)):
        if xx==0:
            plt.scatter(tss3[xx]*np.ones(len(t0s_lr2[xx])), t0s_lr2[xx]/0.85, c='C0', label='low rank')
            plt.scatter(tss3[xx]*np.ones(len(t0s_fr2[xx])), t0s_fr2[xx]/0.85,  c='C1', label='full rank')
        else:
            plt.scatter(tss3[xx]*np.ones(len(t0s_lr2[xx])), t0s_lr2[xx]/0.85,  c='C0')
            plt.scatter(tss3[xx]*np.ones(len(t0s_fr2[xx])), t0s_fr2[xx]/0.85,  c='C1')     
    plt.scatter(tss2, tss2, s=60, c='k')
    ax.set_xlabel('$t_s$ (ms)')
    ax.set_ylabel('$t_p$ (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.plot(tss2, tss2, '--k')
    plt.ylim([600,2500])
    plt.xlim([600,3100])
    
    plt.legend()
    #plt.savefig('Plots/MWG2_performance_lrvsfr_tr'+str(tR)+'_gener_June.pdf')   
    plt.show()
    #%
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(tss3, tss3, '--k')
    for xx in range(len(tss3)):
        if xx==0:
            plt.scatter(tss3[xx]*np.ones(len(t0s_fr2[xx])), t0s_fr2[xx]/0.85, color=c1, edgecolor=c2,  s=30, label='full rank')
            plt.scatter(tss3[xx]*np.ones(len(t0s_lr2[xx])), t0s_lr2[xx]/0.85, color=c2, edgecolor='k', s=30, label='rank three')
        else:
            plt.scatter(tss3[xx]*np.ones(len(t0s_lr2[xx])), t0s_lr2[xx]/0.85, color=c2, edgecolor='k', s=30)
            plt.scatter(tss3[xx]*np.ones(len(t0s_fr2[xx])), t0s_fr2[xx]/0.85, color=c1, edgecolor=c2, s=30)     
    plt.scatter(tss2, tss2, s=60, c='k')
    ax.set_xlabel('$t_s$ (ms)')
    ax.set_ylabel('$t_p$ (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.plot(tss2, tss2, '--k')
    plt.ylim([600,2400])
    plt.xlim([600,3100])
        
    for xx in range(len(tss2)):
        plt.scatter(tss2[xx], tss2[xx], marker='s',s= 80, color=cls[xx,:], edgecolor='k', zorder=4)
        
    plt.legend(loc=2)
    #plt.savefig('Plots/FM_Fig1_H_June.pdf')   
    plt.show()

 

#%%
FF = 0.7
fig_width = FF*0.7*1.5*2.2 # width in inches
fig_height = FF*1.5*2  # height in inches
figsize2 = [fig_width, fig_height]
fig = plt.figure(figsize=figsize2)
ax = fig.add_subplot(111)

# for tr in range(trials):
#     plt.plot([0, 1], [P_all[tr], P_fr_all[tr]])
    

plt.plot([0,0],[np.mean(P_all)-np.std(P_all), np.mean(P_all)+np.std(P_all)], lw=2,c='k')
plt.scatter([0],np.mean(P_all), s=70, color=0.2*np.ones(3), edgecolor='k', zorder=4)

plt.plot([1,1],[np.mean(P_fr_all)-np.std(P_fr_all), np.mean(P_fr_all)+np.std(P_fr_all)], lw=2,c='k')
plt.scatter([1],np.mean(P_fr_all), s=70, color=0.65*np.ones(3), edgecolor='k', zorder=4)


plt.xticks([0, 1])
ax.set_xticklabels(['rank three', 'full rank'])
ax.set_ylabel('participation ratio', fontsize=12)
plt.xticks(rotation = 25)
plt.yticks([1.5, 2., 2.5])
plt.ylim([1.1, 2.5])
plt.xlim([-0.3, 1.3])

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('PR_MWG.pdf', transparent=True)
plt.show()

#%%
FF = 0.7
fig_width = FF*0.7*1.5*2.2 # width in inches
fig_height = FF*1.5*2  # height in inches
figsize2 = [fig_width, fig_height]
fig = plt.figure(figsize=figsize2)
ax = fig.add_subplot(111)

# for tr in range(trials):
#     plt.plot([0, 1], [P_all[tr], P_fr_all[tr]])
    

plt.plot([0,0],[np.mean(P_all2)-np.std(P_all2), np.mean(P_all2)+np.std(P_all2)], lw=2,c='k')
plt.scatter([0],np.mean(P_all2), s=70, color=0.2*np.ones(3), edgecolor='k', zorder=4)

plt.plot([1,1],[np.mean(P_fr_all2[1:])-np.std(P_fr_all2[1:]), np.mean(P_fr_all2[1:])+np.std(P_fr_all2[1:])], lw=2,c='k')
plt.scatter([1],np.mean(P_fr_all2[1:]), s=70, color=0.65*np.ones(3), edgecolor='k', zorder=4)


plt.xticks([0, 1])
ax.set_xticklabels(['rank three', 'full rank'])
ax.set_ylabel('participation ratio', fontsize=12)
plt.xticks(rotation = 25)
#plt.yticks([1.5, 2., 2.5])
#plt.ylim([1.1, 2.5])
plt.xlim([-0.3, 1.3])
plt.yticks([1.5, 2., 2.5, 3.], labels=['','2', '', '3'])
plt.ylim([1.35, 3.1])

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('PR_MWG_2.pdf', transparent=True)
plt.show()
                        
#%%

def mean_outlier(t0s, val = 0.5):
    mean_sol = np.zeros(len(t0s))
    for xx in range(len(t0s)):
        vals = 1.1*np.array(t0s[xx])
        if np.std(vals)>val*np.mean(vals):
            vals[vals>np.mean(vals)] = np.nan
            
        mean_sol[xx] = np.nanmean(vals)
    return(mean_sol)

fig = plt.figure()
ax = fig.add_subplot(111)
m = 0
m_fr = 0
sm = 0
sm_fr =0
for itr, tr in enumerate(trials):
    T0s_fr = T0s_fr2[itr]
    T0s_lr = T0s_lr2[itr]
    
    c1 = 0.9*np.ones(3)
    c2 = 0.5*np.ones(3)

    #plt.plot(amps4,  mean_outlier(T0s_lr), c='k')
    m  += mean_outlier(T0s_lr, val=0.2)
    sm += (mean_outlier(T0s_lr, val=0.2))**2
    m_fr  += mean_outlier(T0s_fr)
    sm_fr  += (mean_outlier(T0s_fr))**2
    
    
m = m/len(trials)
m_fr = m_fr/len(trials)
sm = np.sqrt(sm/len(trials) - m**2)
sm_fr = np.sqrt(sm_fr/len(trials) - m_fr**2)

plt.plot(tss3,  m_fr, '--',lw=2, c='k', label='full rank')
plt.fill_between(tss3, m_fr-sm_fr, m_fr+sm_fr, color=0.7*np.ones(3))
plt.plot(tss3,  m, lw=3, c='k', label='rank three')
plt.fill_between(tss3, m-sm/np.sqrt(len(trials)), m+sm/np.sqrt(len(trials)), color=0.4*np.ones(3))


    
   
for xx in range(len(tss2)):
    plt.scatter(tss2[xx], tss2[xx], marker='s',s= 80, color=cls[xx,:], edgecolor='k', zorder=4)
plt.plot(tss3, tss3,'--k', lw=0.8)
plt.ylim([500, 2500])
plt.xlim([800, 2500])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.xlabel(r'cue amplitude')
plt.ylabel(r'$t_p$ (ms)')

plt.legend()
#plt.savefig('Performance_sev_MWG.pdf')
plt.show()

#%%

def mean_outlier2(t0s, val = 0.5):
    mean_sol = np.zeros(len(t0s))
    std_sol = np.zeros(len(t0s))
    
    for xx in range(len(t0s)):
        vals = np.array(t0s[xx])/0.85
        if np.std(vals)>val*np.mean(vals):
            vals[vals>np.mean(vals)] = np.nan
            
        mean_sol[xx] = np.nanmean(vals)
        std_sol[xx] = np.std(vals)
        
    return(mean_sol, std_sol  )

for itr, tr in enumerate(trials):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = 0
    m_fr = 0
    sm = 0
    sm_fr =0

    T0s_fr = T0s_fr2[itr]
    T0s_lr = T0s_lr2[itr]
    
    c1 = 0.9*np.ones(3)
    c2 = 0.5*np.ones(3)

    #plt.plot(amps4,  mean_outlier(T0s_lr), c='k')
    m, sm  = mean_outlier2(T0s_lr, val=0.2)
    
    m_fr, sm_fr  = mean_outlier2(T0s_fr, val=0.2)
    
    
    


    # plt.plot(tss3,  m_fr, '--',lw=2, c='k', label='full rank')
    # plt.fill_between(tss3, m_fr-sm_fr, m_fr+sm_fr, color=0.7*np.ones(3))
    plt.plot(tss3,  m, lw=3, c='k', label='rank three')
    plt.fill_between(tss3, m-sm/np.sqrt(len(trials)), m+sm/np.sqrt(len(trials)), color=0.4*np.ones(3))
    

    
   
    for xx in range(len(tss2)):
        plt.scatter(tss2[xx], tss2[xx], marker='s',s= 80, color=cls[xx,:], edgecolor='k', zorder=4)
    plt.plot(tss3, tss3,'--k', lw=0.8)
    plt.ylim([500, 2500])
    plt.xlim([500, 2500])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel(r'input interval (ms)')
    plt.ylabel(r'output interval (ms)')
    
    #plt.legend()
    #plt.savefig('Performance_sev_MWG_net_lr_'+str(tr)+'.pdf')
    plt.show()
    
    #%
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = 0
    m_fr = 0
    sm = 0
    sm_fr =0

    T0s_fr = T0s_fr2[itr]
    T0s_lr = T0s_lr2[itr]
    
    c1 = 0.9*np.ones(3)
    c2 = 0.5*np.ones(3)

    #plt.plot(amps4,  mean_outlier(T0s_lr), c='k')
    m, sm  = mean_outlier2(T0s_lr, val=0.2)
    
    m_fr, sm_fr  = mean_outlier2(T0s_fr, val=0.2)
    
    plt.plot(tss3,  m_fr, '--',lw=2, c='k', label='full rank')
    plt.fill_between(tss3, m_fr-sm_fr, m_fr+sm_fr, color=0.7*np.ones(3))
    #plt.plot(tss3,  m, lw=3, c='k', label='rank three')
    #plt.fill_between(tss3, m-sm/np.sqrt(len(trials)), m+sm/np.sqrt(len(trials)), color=0.4*np.ones(3))
    

    for xx in range(len(tss2)):
        plt.scatter(tss2[xx], tss2[xx], marker='s',s= 80, color=cls[xx,:], edgecolor='k', zorder=4)
    plt.plot(tss3, tss3,'--k', lw=0.8)
    plt.ylim([500, 2500])
    plt.xlim([500, 2500])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel(r'input interval (ms)')
    plt.ylabel(r'output interval (ms)')
    
    #plt.legend()
    #plt.savefig('Performance_sev_MWG_net_fr_'+str(tr)+'.pdf')
    plt.show()



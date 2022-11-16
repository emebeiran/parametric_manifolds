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

#import lib_rnns as lr
import tools_MF as tm
from mpl_toolkits.mplot3d import Axes3D
import funcs_Sphere as fs
from matplotlib import cm

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


#%%
tss = np.array((800, 1550))
tss2 = np.array(( 800,  1050, 1300, 1550))
tss22 = 2*np.array(( 800,  1050, 1300, 1550))

tss3 = np.linspace(800, 3000, 20)
tss32 = 1.5*np.linspace(800, 3000, 20)

N_steps = 5


Tss  = fs.gen_intervals(tss,N_steps)
Tss2 = fs.gen_intervals(tss2,N_steps)


#%%
Nt = 1100 
time = np.arange(Nt)

Nt2 = 1300
time2 = np.arange(Nt2)

#%%
# Parameters of task
SR_on = 60
factor = 1
dela = 150
repeat = 1#7 #20 number of examples

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
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

        #%%
        
        # hellooo
def plot_output_MWG(net_low_all, tss2, tss22, dt, time, rank=3, give_trajs=False, plot=True,
                    fr = False, t0s=False, gener=False, tss_ref= 0, tss_ref2=0, dela = 150, give_inps=False,
                    plot_sev = np.ones(2), s1 = 0., s2 = 0.1, rem_prod=False, rem_meas = False, non_avg=False, R_on= False):
    if fr==False:
        M_pre, N_pre, corr_pre, I_pre, O_pre, J_pre = fs.get_SVDweights_CSG(net_low_all, rank=rank)
    if plot:
        fig_width = 1.5*2.2 # width in inches
        fig_height = 0.8*1.5*2  # height in inches
        fig_size =  [fig_width,fig_height]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        
    CLL = CLL2
    if gener==True:
        evenly_spaced_interval = np.linspace(0, 1, len(tss2)+10)
        cls2 = [cm.viridis(x) for x in evenly_spaced_interval]
        CLL = cls2
    trials = 10
    Trajs = np.zeros((len(time), rank, len(tss2)))
    if non_avg == False:
        Inps  = np.zeros((len(time), net_low_all.hidden_size, len(tss2)))
    else:
        Inps  = np.zeros((trials, len(time), net_low_all.hidden_size, len(tss2)))
    
    if sum(plot_sev)>1:
        Trajs2 = np.zeros((len(time), rank, len(tss2)))
        if non_avg == False:
            Inps2  = np.zeros((len(time), net_low_all.hidden_size, len(tss2)))
        else:
            Inps2  = np.zeros((trials, len(time), net_low_all.hidden_size, len(tss2)))
        
    T0s_lr = []
    T0s_lr2 = []
    
    if R_on==False:
        R_on = 1000//dt
    dtss = tss2[1]-tss2[0]
    Nt = len(time)
    T0 = 6000
    if plot_sev[0] == 1.: 
        for xx in range(len(tss2)):
            input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG2(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delayF = dela, inp_size=input_size, 
                                                                                 perc = 0., perc1 = 0., just2=0, s1 = s1, s2 = s2, align_set=True )
            if rem_prod:
                input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG3(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delayF = dela, inp_size=input_size, 
                                                                                 perc = 0., perc1 = 0., just2=0, s1 = s1, s2 = s2, align_set=True, rem_prod=True)                
            if rem_meas:
                input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG3(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delayF = dela, inp_size=input_size, 
                                                                                 perc = 0., perc1 = 0., just2=0, s1 = s1, s2 = s2, align_set=True, rem_meas=True )                               
            outp, traj = net_low_all.forward(input_train, return_dynamics=True)
            outp = outp.detach().numpy()
            traj = traj.detach().numpy()
            #if non_avg==False:
            mtraj  = np.mean(traj,0)
            # else:
            #     mtraj = traj
            if np.shape(mtraj)[0]>Nt:
                mtraj2 = mtraj[:-1,:]
                traj = traj[:,:-1,:]
            else:
                mtraj2 = mtraj
            if fr==False:

                k1_traj = M_pre[:,0].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,0]**2))
                k2_traj = M_pre[:,1].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,1]**2))
                k3_traj = M_pre[:,2].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,2]**2))
                Trajs[:,0,xx] = k1_traj[:-1]
                Trajs[:,1,xx] = k2_traj[:-1]
                Trajs[:,2,xx] = k3_traj[:-1]
            if give_inps==True:
                if non_avg==False:
                    Inps[:,:,xx] = mtraj2
                else:
                    Inps[:,:,:,xx] = traj
                
            outp2 = np.copy(outp)
            
            outp3 = np.copy(outp2[:,1:,0])
            outp3[:,np.diff(low_pass(np.mean(outp2[:,:,0],0)))<0]=5.
            
            outp3[:, time[1:]*dt<4000] = 5. 
            outp2 = outp3
    
            tt0s = time[np.argmin(np.abs(outp2-0.35),1)]*dt-R_on*dt-np.max((np.max(tss2),np.max(tss22)))-dela*dt
            T0s_lr.append(tt0s)
            avg_outp0 = np.mean(outp[:,:,0],0)#np.mean(outp3[:,:,0],0)
            if plot:
                        
                if gener==False:
                    ax.plot(time*dt-T0, avg_outp0, color=CLL[xx,:], lw=2)    
                    ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color='k', alpha=0.5)
                    #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color=CLL[xx,:], alpha=0.5)
                    #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color=CLL[xx,:], alpha=0.5)
                else:
                    if np.min(np.abs(tss2[xx]-tss_ref))<0.5*dtss:
                        ax.plot(time*dt-T0, avg_outp0, color='k', lw=2)    
                        ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color='k', alpha=0.5)
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color='k')
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color='k')
                    
                    else:
                        ax.plot(time*dt-T0, avg_outp0, color=CLL[xx], lw=1.5)    
                        ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color='k', alpha=0.5)
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color=CLL[xx], alpha=0.5)
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color=CLL[xx], alpha=0.5)
                    
    # Doing long stimuli
    if plot_sev[1] == 1.:
        dtss = tss22[1]-tss22[0]
        for xx in range(len(tss22)):
            input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG2(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delayF = dela, inp_size=input_size, perc = 0.,
                                                                                 perc1 = 0., just2=1, align_set=True,  s1 = s1, s2 = s2)
            if rem_prod:
                input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG3(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delayF = dela, inp_size=input_size, 
                                                                                 perc = 0., perc1 = 0., just2=1, s1 = s1, s2 = s2, align_set=True, rem_prod=True)                
            if rem_meas:
                input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG3(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delayF = dela, inp_size=input_size, 
                                                                                 perc = 0., perc1 = 0., just2=1, s1 = s1, s2 = s2, align_set=True, rem_meas=True )  
            outp, traj = net_low_all.forward(input_train, return_dynamics=True)
            outp = outp.detach().numpy()
            
            traj = traj.detach().numpy()
            traj2 = traj
            
            mtraj  = np.mean(traj,0)
            if np.shape(mtraj)[0]>Nt:
                mtraj2 = mtraj[:-1,:]
                traj2 = traj2[:,:-1,:]
            else:
                mtraj2 = mtraj
            if fr==False:
                k1_traj = M_pre[:,0].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,0]**2))
                k2_traj = M_pre[:,1].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,1]**2))
                k3_traj = M_pre[:,2].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,2]**2))
                if sum(plot_sev)==1:
                    Trajs[:,0,xx] = k1_traj[:-1]
                    Trajs[:,1,xx] = k2_traj[:-1]
                    Trajs[:,2,xx] = k3_traj[:-1]
                else:
                    Trajs2[:,0,xx] = k1_traj[:-1]
                    Trajs2[:,1,xx] = k2_traj[:-1]
                    Trajs2[:,2,xx] = k3_traj[:-1]

            
            if give_inps==True:
                if sum(plot_sev)==1:
                    if non_avg==False:
                        Inps[:,:,xx] = mtraj2
                    else:
                        Inps[:,:,:,xx] = traj2
                else:
                    if non_avg ==False:
                        Inps2[:,:,xx] = mtraj2
                    else:
                        Inps2[:,:,:,xx] = traj2
            outp2 = np.copy(outp)
            
            outp3 = np.copy(outp2[:,1:,0])
            outp3[:,np.diff(low_pass(np.mean(outp2[:,:,0],0), ints=10))<0]=5.
            
            outp3[:, time[1:]*dt<R_on*dt+np.max((np.max(tss2),np.max(tss22)))+dela*dt] = 5. 
            outp3[:, time[1:]*dt<R_on*dt+np.max((np.max(tss2),np.max(tss22)))+dela*dt] = 5. 
            
            outp2 = outp3
    
            tt0s = time[np.argmin(np.abs(outp2-0.35),1)]*dt-R_on*dt-np.max((np.max(tss2),np.max(tss22)))-dela*dt
            T0s_lr2.append(tt0s)
    
            avg_outp0 = np.mean(outp[:,:,0],0) #np.mean(outp3[:,:,0],0)
            if plot:
                
                if gener==False:
                    
                    nC = CLL[xx+4,:]#lighten_color(CLL[xx,:], amount=0.5)
                    ax.plot(time*dt-T0, avg_outp0, color=nC, lw=2)    
                    ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color='k', alpha=0.5)
                    #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color=nC, alpha=0.5)
                    #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color=nC, alpha=0.5)
                else:
                    nC = CLL[xx]#lighten_color(CLL[xx], amount=0.5)
                    if np.min(np.abs(tss22[xx]-tss_ref2))<0.5*dtss:
                        ax.plot(time*dt-T0, avg_outp0, color='k', lw=2)    
                        ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color='k',alpha=0.5)
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color='k')
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color='k')
                    else:
                        ax.plot(time*dt-T0, avg_outp0, color=nC, lw=1.5)    
                        ax.plot(time*dt-T0, output_train.detach().numpy()[0,:,0], '--', color='k',alpha=0.5)
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,0], color=nC, alpha=0.5)
                        #ax.plot(time*dt, input_train.detach().numpy()[0,:,1], color=nC, alpha=0.5)
                
    if plot:                    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('time (ms)')
        plt.ylabel('read out')
        plt.ylim([-0.6, 0.6])
        plt.yticks([-0.5, 0, 0.5])
        #plt.xlim([-800, 3500])
    if give_trajs==True:
        return(Trajs)
    
    if give_inps==True and fr == True and sum(plot_sev)==1.:
        return(Inps)
    elif give_inps==True and fr == False and sum(plot_sev)==1.:
        return(Inps, Trajs)
    elif give_inps==True and fr == False and sum(plot_sev)==2.:
        return(Inps, Trajs, Inps2, Trajs2)
    elif give_inps==True and fr == True and sum(plot_sev)==2.:
        return(Inps, Inps2)
    elif t0s==True:
        return(fig, ax, T0s_lr, T0s_lr2)
    else:
        return(fig, ax)
    
    
def plot_output_MWG_Inp(net_low_all, tss2, tss22, dt, time, rank=3, give_trajs=False, plot=True,
                    t0s=False, gener=False, tss_ref= 0, tss_ref2=0, dela = 150,
                    plot_sev = np.ones(2), s1 = 0., s2 = 0.1):
    M_pre, N_pre, corr_pre, I_pre, O_pre, J_pre = fs.get_SVDweights_CSG(net_low_all, rank=rank)
    Ib = np.copy(I_pre[3,:])

    for r in range(rank):
        P = np.dot(M_pre[:,r,np.newaxis], M_pre[:,r,np.newaxis].T)/np.dot(M_pre[:,r].T, M_pre[:,r])
        Ib = Ib - P.dot(Ib)
    CLL = cls

    trials = 10
    Trajs = np.zeros((len(time), rank+1, len(tss2)))
    Trajs2 = np.zeros((len(time), rank+1, len(tss2)))
    
    T0s_lr = []
    T0s_lr2 = []
    
    R_on = 1000//dt
    dtss = tss2[1]-tss2[0]
    Nt = len(time)
    if plot_sev[0] == 1.: 
        for xx in range(len(tss2)):
            #input_train, output_train, mask_train, ct_train, ct2_train, ct3_train = fs.create_inp_out_MWG(trials, Nt, tss2//dt, 
            #                                                        R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = True, delayF = dela, inp_size=3)
            input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG2(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delay_min = dela, inp_size=input_size, 
                                                                                 perc = 0., perc1 = 0., just2=0, s1 = s1, s2 = s2, align_set=True )
            outp, traj = net_low_all.forward(input_train, return_dynamics=True)
            outp = outp.detach().numpy()

            traj = traj.detach().numpy()
            
            mtraj  = np.mean(traj,0)
            k1_traj = M_pre[:,0].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,0]**2))
            k2_traj = M_pre[:,1].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,1]**2))
            k3_traj = M_pre[:,2].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,2]**2))
            kI_traj = Ib.dot(mtraj.T)/np.sum(Ib**2)
            
            Trajs[:,0,xx] = k1_traj[:-1]
            Trajs[:,1,xx] = k2_traj[:-1]
            Trajs[:,2,xx] = k3_traj[:-1]
            Trajs[:,3,xx] = kI_traj[:-1]
                
                    
    # Doing long stimuli
    if plot_sev[1] == 1.:
        dtss = tss22[1]-tss22[0]
        for xx in range(len(tss22)):
            input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG2(trials, Nt, tss2//dt, tss22//dt,
                                                                                 R_on+dela, 1, just=xx, fact=factor, delay_min = dela, inp_size=input_size, perc = 0.,
                                                                                 perc1 = 0., just2=1, align_set=True,  s1 = s1, s2 = s2)
            outp, traj = net_low_all.forward(input_train, return_dynamics=True)
            outp = outp.detach().numpy()

            traj = traj.detach().numpy()
            
            mtraj  = np.mean(traj,0)
            k1_traj = M_pre[:,0].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,0]**2))
            k2_traj = M_pre[:,1].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,1]**2))
            k3_traj = M_pre[:,2].dot(mtraj.T)/(np.sqrt(hidden_size)*np.sum(M_pre[:,2]**2))
            kI_traj = Ib.dot(mtraj.T)/np.sum(Ib**2)
            Trajs2[:,0,xx] = k1_traj[:-1]
            Trajs2[:,1,xx] = k2_traj[:-1]
            Trajs2[:,2,xx] = k3_traj[:-1]
            Trajs2[:,3,xx] = kI_traj[:-1]

            
    return(Trajs, Trajs2)

def plot_inputs_MWG(net_low_all, tss2, tss22,  dt, rank=3, give_trajs=False, plot=True,
                    fr = False, t0s=False, gener=False, tss_ref= 0, tss_ref2 = 0, dela = 150, 
                    only_perf=False, dist1=True, dist2 =True, gener2 =False):

    if plot:
        fig_width = 1.5*2.2 # width in inches
        fig_height = 1.2*1.5*2  # height in inches
        fig_size =  [fig_width,fig_height]
        fig = plt.figure(figsize=fig_size)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        
    CLL = CLL2
    if gener==True or gener2==True:
        evenly_spaced_interval = np.linspace(0, 1, len(tss2)+10)
        cls2 = [cm.viridis(x) for x in evenly_spaced_interval]
        CLL = cls2
    trials = 10

    

    dtss = tss2[1]-tss2[0]
    if dist1:
        for xx in range(len(tss2)):
            input_train, output_train, mask_train, ct_train, ct2_train, ct3_train = fs.create_inp_out_MWG(trials, Nt, tss2//dt, 
                                                                    R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = False, delayF = dela, inp_size=3)
    
            T0=2500
            if plot:
    
                if gener==False and gener2==False:
                    ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k', alpha=0.5)
                    ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color=CLL[xx,:], alpha=1, lw=2)
                elif gener2==False:
                    if np.min(np.abs(tss2[xx]-tss_ref))<0.5*dtss:
                        ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k')
                        ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color='k', lw=2)
                    else:
                        ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k', alpha=0.5)
                        ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color=CLL[xx], alpha=0.8, lw=1)
                else:
                    ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k')
                    ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color=CLL[xx], alpha=0.8, lw=1)
        if gener2==False:
            ax3.plot(time*dt-T0, 0*np.ones_like(time), lw=4, color='C0', alpha=0.7)
        
    if dist2:
        for xx in range(len(tss22)):
            input_train, output_train, mask_train, ct_train, ct2_train, ct3_train = fs.create_inp_out_MWG(trials, Nt, tss22//dt, 
                                                                    R_on+dela, 1, just=xx, perc=0., perc1=0., fact=factor, align_set = False, delayF = dela, inp_size=3)
    
            T0=2500
            if plot:
                if gener==False and gener2==False:
                    ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k', alpha=0.5)
                    ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color=CLL[xx+4,:], alpha=1, lw=2)
                elif gener2==False:
                    if np.min(np.abs(tss2[xx]-tss_ref2))<0.5*dtss:
                        ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k')
                        ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color='k', lw=2)
                    else:
                        ax1.plot(time*dt-T0, input_train.detach().numpy()[0,:,0], color='k', alpha=0.5)
                        ax2.plot(time*dt-T0, input_train.detach().numpy()[0,:,1], color=CLL[xx+4], alpha=0.8, lw=1)
        if gener2==False:          
            ax3.plot(time*dt-T0, 0.1*np.ones_like(time), lw=4, color='k', alpha=0.3)
    if gener2:
        ax3.plot(time*dt-T0, 0.1*np.ones_like(time), lw=3, color='k')
        ax3.plot(time*dt-T0, 0.*np.ones_like(time), lw=3, color='C0')
        ax3.plot(time*dt-T0, 0.05*np.ones_like(time), lw=2, color='C1', alpha=0.6)
        ax3.plot(time*dt-T0, 0.15*np.ones_like(time), lw=2, color='C3', alpha=0.6)
        ax3.plot(time*dt-T0, 0.2*np.ones_like(time), lw=2, color='C4', alpha=0.6)
        
    if plot:                    
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
        ax3.set_xlabel('time (ms)')
        ax1.set_ylabel('input 1')
        ax2.set_ylabel('input 2')
        ax3.set_ylabel('input ctx')
        
        ax1.set_ylim([-0.05, 1])
        ax2.set_ylim([-0.05, 1])
        if gener2==False:
            ax3.set_ylim([-0.02, 0.12])
        else:
            ax3.set_ylim([-0.05, 0.25])
        
        if gener2==False:
            if gener==False:
                ax2.set_xlim([-100, 3500])
                ax1.set_xlim([-100, 3500])
                ax3.set_xlim([-100, 3500])
                
            else:
                ax2.set_xlim([-100, 5100])
                ax1.set_xlim([-100, 5100])
                ax3.set_xlim([-100, 5100])
        else:
            
            ax2.set_xlim([-400, 6400])
            ax1.set_xlim([-400, 6400])
            ax3.set_xlim([-400, 6400])
        if gener2==False:
            if gener==False:
                
                ax1.set_xticks([0, 1000, 2000,3000])
                ax2.set_xticks([0,  1000, 2000, 3000])
                ax3.set_xticks([0,  1000, 2000, 3000])
                
                ax1.set_xticklabels(['','','',''])
                ax2.set_xticklabels(['','','',''])
            else:
                ax1.set_xticks([0, 2000, 4000])
                ax2.set_xticks([0, 2000, 4000])
                ax3.set_xticks([0, 2000, 4000])
                
                ax1.set_xticklabels(['','',''])
                ax2.set_xticklabels(['','',''])
        else:
            ax1.set_xticks([0, 3000, 6000])
            ax2.set_xticks([0, 3000, 6000])
            ax3.set_xticks([0, 3000, 6000])
            
            ax1.set_xticklabels(['','',''])
            ax2.set_xticklabels(['','',''])            
            
        
        
    return(fig, ax1, ax2, ax3)

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
#12 and 15 is quite good
train_ = False
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
 
for tR in range(repeat):
    tR+=1
    A = np.load('TrainedNets/net_MWG'+str(tR+1)+'.npz')#A = np.load('net_MWG'+str(tR+1)+'.npz')
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
    Is2 = np.zeros((hidden_size, input_size))
    Is2[:,input_size-1] = N[:,-2]
    inp_I = torch.from_numpy(Is2.T).type(dtype)
    out_I = torch.from_numpy(Wo).type(dtype)
    h0_i = torch.from_numpy(cond0).type(dtype)

    print('Repeat '+str(tR))
   # md.OptimizedLowRankRNN(input_size, hidden_size, output_size, std_noise_rec, alpha,
   #                                  rank=rank, train_wi = True, train_wo = True, train_h0=True, train_wrec = False,
   #                                  wo_init=out_I, m_init=mrec_I, n_init=nrec_I, h0_init = h0_i)
     
    if train_ == True:
        net_low_inp0 = md.OptimizedLowRankRNN(input_size, hidden_size, output_size, std_noise_rec, alpha,
                                     rank=rank, train_wi = True, train_wo = True, train_h0=True, train_wrec = False,
                                     wo_init=out_I, m_init=mrec_I, n_init=nrec_I, h0_init = h0_i)
    
        net_low_fr0 = md.FullRankRNN(input_size, hidden_size, output_size, 0.1*std_noise_rec, alpha, 
                                               train_wi = True, train_wo = True, train_h0=True)  
        input_train, output_train, mask_train, ct_train, ct2_train, ct3_train, ct_ctxt = fs.create_inp_out_MWG2(trials_train, Nt, tss2//dt, tss22//dt,
                                                                             R_on, SR_on, fact=factor, delay_min = 100, inp_size=input_size, perc = 0.1, perc1 = 0.1)

        print('Only inputs')
        
        net_low_inp0.load_state_dict(torch.load("MWG_cluster_rank"+str(rank)+"_rep_"+str(tR)+"_2int_traininpall.pt", map_location='cpu'))
        
        Io = net_low_inp0.wi.detach().numpy()
        Mo = net_low_inp0.m.detach().numpy()
        No = net_low_inp0.n.detach().numpy()
        
        In= np.zeros((input_size, hidden_size))
        In[0:3,:] = Io
        In[input_size-1,:] = 0.01*No[:,0]
        In =  torch.from_numpy(In).type(dtype)
        
        net_low_inp = md.OptimizedLowRankRNN(input_size, hidden_size, output_size, std_noise_rec, alpha,
                                     rank=rank, train_wi = True, train_wo = True, train_h0=True, train_wrec = False, wi_init=In, 
                                     wo_init=net_low_inp0.wo, m_init=net_low_inp0.m, n_init=net_low_inp0.n, h0_init = net_low_inp0.h0)
        
        torch.save(net_low_inp.state_dict(), "MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_2int_traininp_0.pt")
        loss = md.train(net_low_inp, input_train, output_train[:,:,0:1], mask_train[:,:,0:1], n_epochs=120, plot_learning_curve=True, plot_gradient=True, 
              lr=5e-4, clip_gradient = 1., keep_best=False, cuda=True, save_loss=True)
        torch.save(net_low_inp.state_dict(), "MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_2int_traininp.pt")

        np.savez("MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_2int_traininp_loss", loss)
        
        print('first inputs, then all')
        net_low_all = md.OptimizedLowRankRNN(input_size, hidden_size, output_size, std_noise_rec, alpha,
                                     rank=rank, train_wi = True, train_wo = True, train_h0=True, train_wrec = True, wi_init=In, 
                                     wo_init=net_low_inp0.wo, m_init=net_low_inp0.m, n_init=net_low_inp0.n, h0_init = net_low_inp0.h0)
        net_low_all.load_state_dict(torch.load("MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_2int_traininp.pt", map_location='cpu'))
        torch.save(net_low_all.state_dict(), "MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_2int_traininpall_0.pt")
        loss = md.train(net_low_all, input_train, output_train[:,:,0:1], mask_train[:,:,0:1], n_epochs=120, plot_learning_curve=True, plot_gradient=True, 
              lr=2e-4, clip_gradient = 0.2, keep_best=False, cuda=True, save_loss=True)
        torch.save(net_low_all.state_dict(), "MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_2int_traininpall.pt")
        np.savez("MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_2int_traininpall_loss", loss)
        
        print('full rank')
        net_low_fr0.load_state_dict(torch.load("MWG_cluster_rank"+str(rank)+"_rep_"+str(tR)+"_fr.pt", map_location='cpu'))
        
        Io = net_low_fr0.wi.detach().numpy()
        
        In= np.zeros((input_size, hidden_size))
        In[0:3,:] = Io
        In[input_size-1,:] = 0.01*np.random.randn(hidden_size)
        In =  torch.from_numpy(In).type(dtype)            
   
    
        net_low_fr = md.FullRankRNN(input_size, hidden_size, output_size, 0.1*std_noise_rec, alpha, 
                                               train_wi = True, train_wo = True, train_h0=True, 
                                               wi_init = In, wo_init = net_low_fr0.wo, h0_init = net_low_fr0.h0, wrec_init = net_low_fr0.wrec)

        torch.save(net_low_fr.state_dict(), "MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_fr_0.pt")
        loss = md.train(net_low_fr, input_train, output_train[:,:,0:1], mask_train[:,:,0:1], n_epochs=120, plot_learning_curve=True, plot_gradient=True, 
              lr=2e-4, clip_gradient = 0.2, keep_best=False, cuda=True, save_loss=True)
        torch.save(net_low_fr.state_dict(), "MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_fr.pt")

        np.savez("MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_2int_trainFR_loss", loss)
    else:
        #net_low_all.load_state_dict(torch.load("CSG_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tr)+"_2int_gen.pt", map_location='cpu'))
        net_low_all = md.OptimizedLowRankRNN(input_size, hidden_size, output_size, std_noise_rec, alpha,
                                     rank=rank, train_wi = True, train_wo = True, train_h0=True, train_wrec = True)
        net_low_all.load_state_dict(torch.load("TrainedNets/MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_2int_traininp_0.pt", map_location='cpu'))


        print('Training inputs+all')
        net_low_all.load_state_dict(torch.load("TrainedNets/MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_2int_traininpall.pt", map_location='cpu'))
        fig, ax, t0s_lr, t0s_lr2 = plot_output_MWG(net_low_all, tss2, tss22, dt, time, t0s=True)
        #plt.savefig('Plots/MWG2_ctxt2_predesigned_trainedInpsAll_'+str(tR)+'.pdf')  
        plt.show()
        print('Training full rank')
        net_low_fr = md.FullRankRNN(input_size, hidden_size, output_size, 0.1*std_noise_rec, alpha, 
                                               train_wi = True, train_wo = True, train_h0=True)
        net_low_fr.load_state_dict(torch.load("TrainedNets/MWGCtxt2_rank"+str(rank)+"_rep_"+str(tR)+"_fr.pt", map_location='cpu'))
        fig, ax, t0s_fr, t0s_fr2 = plot_output_MWG(net_low_fr, tss2, tss22, dt, time, fr=True, t0s=True)
        #plt.savefig('Plots/MWG2_ctxt2_predesigned_trainedFullR_'+str(tR)+'.pdf')  
        plt.show()
        

        
      
        #%%
        c1 = 0.9*np.ones(3)
        c2 = 0.5*np.ones(3)

       
        fig_width = 1.5*2.2 # width in inches
        fig_height = 1.5*2  # height in inches
        fig_size =  [fig_width,fig_height]
                
        CLL_ = np.zeros((8,3))
        CLL_[0,:] = np.array((150, 53, 47))/256.
        CLL_[1,:] = np.array((207, 84, 57))/256.
        CLL_[2,:] = np.array((212, 104, 81))/256.
        CLL_[3,:] = np.array((220, 135, 64))/256.
        
        CLL_[4,:] = np.array((78, 104, 172))/256.
        CLL_[5,:] = np.array((85, 131, 198))/256.
        CLL_[6,:] = np.array((102, 185, 224))/256.
        CLL_[7,:] = np.array((150, 198, 219))/256.
        fig_width = 1.5*2.2*0.8 # width in inches
        fig_height = 1.5*2*0.7*0.8  # height in inches
        fig_size =  [fig_width,fig_height]
 

        #%%
        print('Generalization low-rank')
        do_delay= False
        #Generalization low rank
        tss3 = np.linspace(500, 4500, 20)
        tss32 = np.linspace(500, 4500, 20)


        fig, ax, t0s_lr2, t0s_lr22 = plot_output_MWG(net_low_all, tss3, tss32, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100)
        #plt.savefig('Plots/MWG2_Ctxt_predesigned_trainedInpsAll_'+str(tR)+'_gener.pdf')  
        plt.show()
        T0s_lr2.append(t0s_lr2)
        
        fig, ax, t0s_lr2f, t0s_lr22f = plot_output_MWG(net_low_all, tss3, tss32, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, plot_sev = np.array((1,0)))
        #plt.savefig('Plots/MWG2_Ctxt_predesigned_trainedInpsAll_'+str(tR)+'_generShort.pdf')  
        plt.show()
        
        
        fig, ax, t0s_lr2f, t0s_lr22f = plot_output_MWG(net_low_all, tss3, tss32, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, plot_sev = np.array((0,1)))
        #plt.savefig('Plots/MWG2_Ctxt_predesigned_trainedInpsAll_'+str(tR)+'_generLong.pdf')  
        plt.show()
        if do_delay:
            fig, ax, t0s_lr2_sD, t0s_lr22_sD = plot_output_MWG(net_low_all, tss3, tss32, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela = 50)
            plt.savefig('Plots/MWG2_Ctxt_predesigned_trainedInpsAll_'+str(tR)+'_gener_sD.pdf')  
            plt.show()
            T0s_lr2_sD.append(t0s_lr2_sD)
            
            fig, ax, t0s_lr2_lD, t0s_lr22_sD = plot_output_MWG(net_low_all, tss3, tss32, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela = 200)
            plt.savefig('Plots/MWG2_Ctxt_predesigned_trainedInpsAll_'+str(tR)+'_gener_lD.pdf')  
            plt.show()
            
            T0s_lr2_lD.append(t0s_lr2_lD)
        
        # Generalization full rank
        print('generalization full-rank')
        fig, ax, t0s_fr2, t0s_fr22 = plot_output_MWG(net_low_fr, tss3, tss32, dt, time2, fr=True, t0s=True, gener=True, tss_ref = tss2, tss_ref2 = tss22, dela=100)
        plt.savefig('Plots/MWG2_Ctxt_predesigned_trainedFullR_'+str(tR)+'_gener.pdf')  
        plt.show()
        T0s_fr2.append(t0s_fr2)
        fig, ax, t0s_fr2f, t0s_fr22f = plot_output_MWG(net_low_fr, tss3, tss32, dt, time2, fr=True, t0s=True, gener=True, tss_ref = tss2, tss_ref2 = tss22, dela=100, plot_sev = np.array((1,0)))
        plt.savefig('Plots/MWG2_Ctxt_predesigned_trainedFullR_'+str(tR)+'_generShort.pdf')  
        plt.show()
        fig, ax, t0s_fr2f, t0s_fr22f = plot_output_MWG(net_low_fr, tss3, tss32, dt, time2, fr=True, t0s=True, gener=True, tss_ref = tss2, tss_ref2 = tss22, dela=100, plot_sev = np.array((0,1)))
        #plt.savefig('Plots/MWG2_Ctxt_predesigned_trainedFullR_'+str(tR)+'_generLong.pdf')  
        plt.show()        
        if do_delay:
            
            fig, ax, t0s_fr2_sD, t0s_fr2_sD2 = plot_output_MWG(net_low_fr, tss3, tss32, dt, time2, fr=True, t0s=True, gener=True, tss_ref = tss2, dela=50)
            plt.savefig('Plots/MWGCtxt_predesigned_trainedFullR_'+str(tR)+'_gener_sD.pdf')  
            plt.show()
            T0s_fr2_sD.append(t0s_fr2_sD)
            
            fig, ax, t0s_fr2_lD, t0s_fr2_lD2 = plot_output_MWG(net_low_fr, tss3,  tss32, dt, time2, fr=True, t0s=True, gener=True, tss_ref = tss2, dela=200)
            plt.savefig('Plots/MWGCtxt_predesigned_trainedFullR_'+str(tR)+'_gener_lD.pdf')  
            plt.show()
            T0s_fr2_lD.append(t0s_fr2_lD)
        

        #%%
        
        c1 = 0.9*np.ones(3)
        c2 = 0.5*np.ones(3)
        
        c1_s = 0.5*(0.9*np.ones(3)+np.array((206,84,57))/256)
        c2_s = 0.5*(0.5*np.ones(3)+np.array((206,84,57))/256)
        
        
        c1_l = 0.5*(0.9*np.ones(3)+np.array((85,130,197))/256)
        c2_l = 0.5*(0.5*np.ones(3)+np.array((85,130,197))/256)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(tss3, tss3, '--k')
        plt.plot(tss32, tss32, '--k', alpha=0.5)
        
        perc = 0.85
        
        for xx in range(len(tss3)):
            if xx==0:
                plt.scatter(tss3[xx]*np.ones(len(t0s_lr2[xx])), t0s_lr2[xx]/0.85, color=c2_s, label='rank three',s=50, edgecolor='k',)
                plt.scatter(tss3[xx]*np.ones(len(t0s_fr2[xx])), t0s_fr2[xx]/0.85, color=c1_s, label='full rank',s=50, edgecolor=c2_s,)
            else:
                plt.scatter(tss3[xx]*np.ones(len(t0s_lr2[xx])), t0s_lr2[xx]/0.85, color=c2_s, s=30, edgecolor='k', alpha=0.8)
                plt.scatter(tss3[xx]*np.ones(len(t0s_fr2[xx])), t0s_fr2[xx]/0.85, color=c1_s, s=30, edgecolor=c2_s, alpha=0.5)                
            plt.scatter(tss32[xx]*np.ones(len(t0s_lr22[xx])), t0s_lr22[xx]/0.85, color=c2_l, s=30, edgecolor='k', alpha=0.8)
            plt.scatter(tss32[xx]*np.ones(len(t0s_fr22[xx])), t0s_fr22[xx]/0.85, color=c1_l, s=30, edgecolor=c2_l, alpha=0.5)  
        for xx in range(len(tss2)):
            
            plt.scatter(tss2[xx], tss2[xx], s=80, color=CLL[xx,:], edgecolor='k', zorder=4, marker='s')
            plt.scatter(tss22[xx], tss22[xx], s=80, color=CLL[xx+4,:], edgecolor='k', zorder=5, marker='s')
            
        plt.scatter(tss2, tss2, s=50, c='k')
        plt.scatter(tss22, tss22, s=50, c='k', alpha=0.5)
        plt.ylim([0, 4500])
        ax.set_xlabel('input interval (ms)')
        ax.set_ylabel('output interval (ms)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.plot(tss2, tss2, '--k')
        plt.legend()
        plt.savefig('Plots/FM_Fig5_C.pdf')   
        plt.show()
        
       

        
                    
                #%%
        
        c1 = 0.9*np.ones(3)
        c2 = 0.5*np.ones(3)
        
        c1_s = 0.5*(0.9*np.ones(3)+np.array((206,84,57))/256)
        c2_s = 0.5*(0.5*np.ones(3)+np.array((206,84,57))/256)
        
        
        c1_l = 0.5*(0.9*np.ones(3)+np.array((85,130,197))/256)
        c2_l = 0.5*(0.5*np.ones(3)+np.array((85,130,197))/256)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(tss3, tss3, '--k')
        plt.plot(tss32, tss32, '--k', alpha=0.5)
        
        perc = 0.85
        
        for xx in range(len(tss3)):
            if xx==0:
                plt.scatter(tss32[xx]*np.ones(len(t0s_lr22[xx])), t0s_lr22[xx]/0.85, color=c2_l, s=30, edgecolor='k', alpha=0.8, label='slow ctxt')
                plt.scatter(tss3[xx]*np.ones(len(t0s_lr2[xx])), t0s_lr2[xx]/0.85, color=c2_s, label='fast ctxt',s=30, edgecolor='k',)
                
                #plt.scatter(tss3[xx]*np.ones(len(t0s_fr2[xx])), t0s_fr2[xx]/0.85, color=c1_s, label='full rank',s=50, edgecolor=c2_s,)
            else:
                plt.scatter(tss3[xx]*np.ones(len(t0s_lr2[xx])), t0s_lr2[xx]/0.85, color=c2_s, s=30, edgecolor='k', alpha=0.8)
                #plt.scatter(tss3[xx]*np.ones(len(t0s_fr2[xx])), t0s_fr2[xx]/0.85, color=c1_s, s=30, edgecolor=c2_s, alpha=0.5)                
            plt.scatter(tss32[xx]*np.ones(len(t0s_lr22[xx])), t0s_lr22[xx]/0.85, color=c2_l, s=30, edgecolor='k', alpha=0.8)
            #plt.scatter(tss32[xx]*np.ones(len(t0s_fr22[xx])), t0s_fr22[xx]/0.85, color=c1_l, s=30, edgecolor=c2_l, alpha=0.5)  
        for xx in range(len(tss2)):
            
            plt.scatter(tss2[xx], tss2[xx], s=80, color=CLL[xx,:], edgecolor='k', zorder=4, marker='s')
            plt.scatter(tss22[xx], tss22[xx], s=80, color=CLL[xx+4,:], edgecolor='k', zorder=5, marker='s')
            
  
            
        plt.scatter(tss2, tss2, s=50, c='k')
        plt.scatter(tss22, tss22, s=50, c='k', alpha=0.5)
        plt.ylim([0, 4500])
        ax.set_xlabel('input interval (ms)')
        ax.set_ylabel('output interval (ms)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.plot(tss2, tss2, '--k')
        plt.legend()
        plt.savefig('Plots/FM_Fig5_C_1.pdf')   
        plt.show()
        #%%
        c1 = 0.9*np.ones(3)
        c2 = 0.5*np.ones(3)
        
        c1_s = 0.5*(0.9*np.ones(3)+np.array((206,84,57))/256)
        c2_s = 0.5*(0.5*np.ones(3)+np.array((206,84,57))/256)
        
        
        c1_l = 0.5*(0.9*np.ones(3)+np.array((85,130,197))/256)
        c2_l = 0.5*(0.5*np.ones(3)+np.array((85,130,197))/256)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(tss3, tss3, '--k')
        plt.plot(tss32, tss32, '--k', alpha=0.5)
        
        perc = 0.85
        range_B_lr = np.zeros((2,2))
        C2_l = np.array((68, 79, 125))/256
        C2_s = np.array((152, 52, 47))/256
        plt.plot(tss32, np.mean(t0s_lr22,-1)/0.85, c=C2_l, lw=2, label='slow ctxt')
        s = np.std(t0s_lr22,-1)/0.85
        m =  np.mean(t0s_lr22,-1)/0.85
        range_B_lr[:,1] = np.array((np.min(m), np.max(m)))
        plt.fill_between(tss32, m-s, m+s, color=C2_l, alpha=0.5)

        plt.plot(tss3, np.mean(t0s_lr2,-1)/0.85, c=C2_s, lw=2, label='fast ctxt')
        s = np.std(t0s_lr2,-1)/0.85
        m =  np.mean(t0s_lr2,-1)/0.85
        range_B_lr[:,0] = np.array((np.min(m), np.max(m)))
        plt.fill_between(tss3, m-s, m+s, color=C2_s, alpha=0.5)        
 
        for xx in range(len(tss2)):
            
            plt.scatter(tss2[xx], tss2[xx], s=80, color=CLL[xx,:], edgecolor='k', zorder=4, marker='s')
            plt.scatter(tss22[xx], tss22[xx], s=80, color=CLL[xx+4,:], edgecolor='k', zorder=5, marker='s')
            

        plt.ylim([0, 3500])
        ax.set_xlabel('input interval (ms)')
        ax.set_ylabel('output interval (ms)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.plot(tss2, tss2, '--k')
        plt.legend()
        plt.savefig('Plots/FM_Fig5_C_1_Aug.pdf')   
        plt.show()


#%%
        def remove_outliers(t0s):
            x = np.copy(t0s)
            for it in range(np.shape(t0s)[0]):
                x[it,t0s[it]<0] = np.nan
                x[it,t0s[it]>4000] = np.nan
                
                # m = np.nanmean(t0s[it])
                # s = np.nanstd(t0s[it])
                # if s>0.25*m:
                #     x[it,x[it,:]>m]=np.nan
                
                #mask = np.abs(x[it,:]-m)>2*s
                #x[it,mask] = np.nan
            return(x)
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(tss3, tss3, '--k')
        plt.plot(tss32, tss32, '--k', alpha=0.5)
        
        perc = 0.85
        
        ct0s_fr22 = remove_outliers(t0s_fr22)
        ct0s_fr2 = remove_outliers(t0s_fr2)
        range_B_fr = np.zeros((2,2))
        
        s = np.nanstd(ct0s_fr22,-1)/0.85
        m =  np.nanmean(ct0s_fr22,-1)/0.85
        plt.plot(tss32, m, c=C2_l, lw=2, label='slow ctxt')
        plt.fill_between(tss32, m-s, m+s, color=C2_l, alpha=0.5)
        
        range_B_fr[:,0] = np.array((np.min(m), np.max(m)))

        s = np.nanstd(ct0s_fr2,-1)/0.85
        m =  np.nanmean(ct0s_fr2,-1)/0.85
        plt.plot(tss3, m, c=C2_s, lw=2, label='slow ctxt')
        plt.fill_between(tss3, m-s, m+s, color=C2_s, alpha=0.5)
        range_B_fr[:,1] = np.array((np.min(m), np.max(m)))
        for xx in range(len(tss2)):
            
            plt.scatter(tss2[xx], tss2[xx], s=80, color=CLL[xx,:], edgecolor='k', zorder=4, marker='s')
            plt.scatter(tss22[xx], tss22[xx], s=80, color=CLL[xx+4,:], edgecolor='k', zorder=5, marker='s')
            

        plt.ylim([0, 3500])
        ax.set_xlabel('input interval (ms)')
        ax.set_ylabel('output interval (ms)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.plot(tss2, tss2, '--k')
        #plt.legend()
        plt.savefig('Plots/FM_Fig5_C_2_Aug.pdf')   
        plt.show()
        np.savez('range_B_fr', range_B_lr)
        
        # #%%
        # fig = plt.figure(figsize=[1.5*2.2*0.5*0.9, 1.5*2.*0.9])
        # ax = fig.add_subplot(111)
        # plt.plot([0,0], range_B_lr[:,0], c=C2_l, lw=6, alpha=0.5)
        # plt.plot([0.1,0.1],  range_B_lr[:,1], c=C2_s, lw=6, alpha=0.5)
        # ax.set_xticks([0, 0.1])
        # ax.set_xlim([-0.04,0.14])
        # ax.set_xlabel('context cue')
        # ax.set_ylabel('output int. range (ms)')
        #%%
        fig = plt.figure(figsize=[1.5*2.2*0.9, 1.5*2.*0.9*0.5])
        ax = fig.add_subplot(111)
        rang = range_B_lr
        plt.plot( rang[:,0], [0,0], c=C2_s, lw=6, alpha=0.7)
        plt.plot( [np.mean(rang[:,0]),np.mean(rang[:,0])], [-0.03,0.03], c='k', lw=4, alpha=0.8, zorder=4)
        print(np.mean(rang[:,0]))
        plt.plot( rang[:,1],[0.1,0.1],   c=C2_l, lw=6, alpha=0.7)
        plt.plot( [np.mean(rang[:,1]),np.mean(rang[:,1])], [-0.03+0.1,0.03+0.1], c='k', lw=4, alpha=0.8, zorder=4)
        print(np.mean(rang[:,1]))
        ax.set_yticks([0, 0.1])
        ax.set_ylim([-0.06,0.16])
        ax.set_xlim([600, 3400])
        ax.set_ylabel('Ctxt')
        ax.set_xlabel('output range (ms)')
        plt.savefig('Plots/FM_Fig5_C_1_Aug_Inset.pdf', transparent=True) 
        
                #%%
        fig = plt.figure(figsize=[1.5*2.2*0.9, 1.5*2.*0.9*0.5])
        ax = fig.add_subplot(111)
        rang = range_B_fr
        plt.plot( rang[:,0], [0,0], c=C2_s, lw=6, alpha=0.7)
        plt.plot( [np.mean(rang[:,0]),np.mean(rang[:,0])], [-0.03,0.03], c='k', lw=4, alpha=0.8, zorder=4)
        print(np.mean(rang[:,0]))
        plt.plot( rang[:,1],[0.1,0.1],   c=C2_l, lw=6, alpha=0.7)
        plt.plot( [np.mean(rang[:,1]),np.mean(rang[:,1])], [-0.03+0.1,0.03+0.1], c='k', lw=4, alpha=0.8, zorder=4)
        print(np.mean(rang[:,1]))
        ax.set_yticks([0, 0.1])
        ax.set_ylim([-0.06,0.16])
        ax.set_xlim([600, 3400])
        ax.set_ylabel('Ctxt')
        ax.set_xlabel('output range (ms)')
        plt.savefig('Plots/FM_Fig5_C_2_Aug_Inset.pdf', transparent=True) 
        
       
        
        #%%
        tss33 = 1.5*np.linspace(100, 4000, 20)
        fig, ax, t0s_lr2_m1, t0s_lr22_m1 = plot_output_MWG(net_low_all, tss3, tss33, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, s1 = 0., s2 = -0.05, plot_sev = np.array((0,1)))
        fig, ax, t0s_lr2_m2, t0s_lr22_m2 = plot_output_MWG(net_low_all, tss3, tss33, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, s1 = 0., s2 = -0.1, plot_sev = np.array((0,1)))
        fig, ax, t0s_lr2_m3, t0s_lr22_m3 = plot_output_MWG(net_low_all, tss3, tss33, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, s1 = 0., s2 = -0.15, plot_sev = np.array((0,1)))
        fig, ax, t0s_lr2_m4, t0s_lr22_m4 = plot_output_MWG(net_low_all, tss3, tss33, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, s1 = 0., s2 = -0.2, plot_sev = np.array((0,1)))
        
        fig, ax, t0s_lr2_0, t0s_lr22_0 = plot_output_MWG(net_low_all, tss3, tss33, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, s1 = 0., s2 = 0., plot_sev = np.array((0,1)))
        fig, ax, t0s_lr2_1, t0s_lr22_1 = plot_output_MWG(net_low_all, tss3, tss33, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, s1 = 0., s2 = 0.05, plot_sev = np.array((0,1)))
        fig, ax, t0s_lr2_2, t0s_lr22_2 = plot_output_MWG(net_low_all, tss3, tss33, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, s1 = 0., s2 = 0.1, plot_sev = np.array((0,1)))
        fig, ax, t0s_lr2_3, t0s_lr22_3 = plot_output_MWG(net_low_all, tss3, tss33, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, s1 = 0., s2 = 0.15, plot_sev = np.array((0,1)))
        fig, ax, t0s_lr2_4, t0s_lr22_4 = plot_output_MWG(net_low_all, tss3, tss33, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, s1 = 0., s2 = 0.2, plot_sev = np.array((0,1)))
        plt.show()
        
        
        fig, ax, t0s_lr2_01, t0s_lr22_01 = plot_output_MWG(net_low_all, tss3, tss33, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, s1 = 0., s2 = 0.025, plot_sev = np.array((0,1)))
        fig, ax, t0s_lr2_11, t0s_lr22_11 = plot_output_MWG(net_low_all, tss3, tss33, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, s1 = 0., s2 = 0.075, plot_sev = np.array((0,1)))
        plt.show()
        
        fig, ax, t0s_lr2_21, t0s_lr22_21 = plot_output_MWG(net_low_all, tss3, tss33, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, s1 = 0., s2 = 0.125, plot_sev = np.array((0,1)))
        fig, ax, t0s_lr2_31, t0s_lr22_31 = plot_output_MWG(net_low_all, tss3, tss33, dt, time2, t0s=True, gener=True, tss_ref = tss2,tss_ref2 = tss22, dela=100, s1 = 0., s2 = 0.175, plot_sev = np.array((0,1)))
        plt.show()
       
        
        #%%
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(tss3, tss3, '--k')
        plt.plot(tss32, tss32, '--k', alpha=0.5)
        
        evenly_spaced_interval = np.linspace(0, 1, 8)
        cls2 = [cm.gist_yarg(x) for x in evenly_spaced_interval]
        
        CL = cls2[3:]
        perc = 0.85
        Lw=2
        Lw2=2.5
        ssiz = 10
        LLw = 0.1
        Cgr = 0.1*np.ones(3)

        range_C1_lr = np.zeros((2,5))


        
        s = np.std(t0s_lr22_01,1)/perc
        m =  np.mean(t0s_lr22_01,-1)/perc
        plt.plot(tss33, m, c=CL[0], lw=2)
        plt.fill_between(tss33, m-s, m+s, color=CL[0], alpha=0.5)
        range_C1_lr[:,1] = np.array((np.min(m), np.max(m)))
        
        s = np.std(t0s_lr22_1,1)/perc
        m =  np.mean(t0s_lr22_1,-1)/perc
        plt.plot(tss33, m, c=CL[1], lw=2)
        plt.fill_between(tss33, m-s, m+s, color=CL[1], alpha=0.5)
        range_C1_lr[:,2] = np.array((np.min(m), np.max(m)))
                
        s = np.std(t0s_lr22_11,1)/perc
        m =  np.mean(t0s_lr22_11,-1)/perc
        plt.plot(tss33, m, c=CL[2], lw=2)
        plt.fill_between(tss33, m-s, m+s, color=CL[2], alpha=0.5)
        range_C1_lr[:,3] = np.array((np.min(m), np.max(m)))

        s = np.std(t0s_lr22_0,1)/perc
        m =  np.mean(t0s_lr22_0,-1)/perc
        plt.plot(tss33, m, c=C2_s, lw=2)
        plt.fill_between(tss33, m-s, m+s, color=C2_s, alpha=0.5)
        range_C1_lr[:,0] = np.array((np.min(m), np.max(m)))

        s = np.std(t0s_lr22_2,1)/perc
        m =  np.mean(t0s_lr22_2,-1)/perc
        plt.plot(tss33, m, c=C2_l, lw=2)
        plt.fill_between(tss33, m-s, m+s, color=C2_l, alpha=0.5)
        range_C1_lr[:,4] = np.array((np.min(m), np.max(m)))
                 
        for xx in range(len(tss2)):
            
            plt.scatter(tss2[xx], tss2[xx], s=80, color=CLL2[xx,:], edgecolor='k', zorder=4, marker='s')
            plt.scatter(tss22[xx], tss22[xx], s=80, color=CLL2[xx+4,:], edgecolor='k', zorder=5, marker='s')
        
        plt.xlim([0, 6090])
        plt.ylim([0, 5500])
        ax.set_xlabel('input interval (ms)')
        ax.set_ylabel('output interval (ms)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.plot(tss2, tss2, '--k')
        #plt.legend()
        plt.savefig('Plots/FM_Fig5_E_21_B_Aug.pdf')   
        plt.show()
        np.savez('range_C1_lr', range_C1_lr)
        
        #%%
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(tss3, tss3, '--k')
        plt.plot(tss32, tss32, '--k', alpha=0.5)
        
        evenly_spaced_interval = np.linspace(0, 1, 8)
        cls2 = [cm.gist_yarg(x) for x in evenly_spaced_interval]
        
        CL = cls2[3:]
        perc = 0.85
        Lw=2
        Lw2=2.5
        ssiz = 10
        LLw = 0.1
        Cgr = 0.1*np.ones(3)


        s = np.std(t0s_lr22_0,1)/perc
        m =  np.mean(t0s_lr22_0,-1)/perc
        plt.plot(tss33, m, c=C2_s, lw=2)
        plt.fill_between(tss33, m-s, m+s, color=C2_s, alpha=0.5)
        # range_C1_lr[:,0] = np.array((np.min(m), np.max(m)))

        s = np.std(t0s_lr22_2,1)/perc
        m =  np.mean(t0s_lr22_2,-1)/perc
        plt.plot(tss33, m, c=C2_l, lw=2)
        plt.fill_between(tss33, m-s, m+s, color=C2_l, alpha=0.5)
        # range_C1_lr[:,4] = np.array((np.min(m), np.max(m)))
                 
        for xx in range(len(tss2)):
            
            plt.scatter(tss2[xx], tss2[xx], s=80, color=CLL2[xx,:], edgecolor='k', zorder=4, marker='s')
            plt.scatter(tss22[xx], tss22[xx], s=80, color=CLL2[xx+4,:], edgecolor='k', zorder=5, marker='s')
        
        plt.xlim([0, 6090])
        plt.ylim([0, 5500])
        ax.set_xlabel('input interval (ms)')
        ax.set_ylabel('output interval (ms)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.plot(tss2, tss2, '--k')

        plt.savefig('Plots/FM_Fig5_E_21_B_Aug0.pdf')   
        plt.show()
        
        
                        #%%
        fig = plt.figure(figsize=[1.5*2.2, 1.5*2.*0.6])
        ax = fig.add_subplot(111)
        rang = range_C1_lr
        for iii in range(np.shape(rang)[1]):
            if iii>0 and iii<4:
                plt.plot( rang[:,iii], [0.1*iii,0.1*iii], c=CL[iii], lw=3, alpha=0.7)
            else:
                if iii==0:
                    plt.plot( rang[:,iii], [0.1*iii,0.1*iii], c=C2_s, lw=4, alpha=0.7)
                else:
                    plt.plot( rang[:,iii], [0.1*iii,0.1*iii], c=C2_l, lw=4, alpha=0.7)
            plt.plot( [np.mean(rang[:,iii]),np.mean(rang[:,iii])], [-0.03+0.1*iii,0.03+0.1*iii], c='k', lw=4, alpha=0.8, zorder=10)
            print(np.mean(rang[:,iii]))
        
        ax.set_yticks([-1, 0, 1.])
        ax.set_yticklabels([-0.2, 0,  0.2])
        ax.set_ylim([-1.25, 1.25])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        #ax.set_ylim([-0.05,0.15])
        ax.set_ylabel('Ctxt')
        ax.set_xlabel('output range (ms)')
        plt.savefig('Plots/FM_Fig5_D_Aug_Inset_1.pdf', transparent=True) 
  

        #%%
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(tss3, tss3, '--k')
        plt.plot(tss32, tss32, '--k', alpha=0.5)
        
        evenly_spaced_interval = np.linspace(0, 1, 8)
        cls2 = [cm.gist_yarg(x) for x in evenly_spaced_interval]
        
        CL = cls2[3:]
        perc = 0.85
        Lw=2
        Lw2=2.5
        ssiz = 10
        LLw = 0.1
        Cgr = 0.1*np.ones(3)
        # for xx in range(len(tss3)):
        #     plt.scatter(tss33[xx]*np.ones(len(t0s_lr2[xx])), t0s_lr22_0[xx]/0.85, color=c2_s, s=ssiz, lw=LLw, edgecolor=Cgr, alpha=1.)
        #     plt.scatter(tss33[xx]*np.ones(len(t0s_lr2[xx])), t0s_lr22_2[xx]/0.85, color=c2_l, s=ssiz, lw=LLw, edgecolor=Cgr, alpha=1.)

        
        range_C2_lr = np.zeros((2,6))



        s = np.std(t0s_lr22_21,1)/perc
        m =  np.mean(t0s_lr22_21,-1)/perc
        plt.plot(tss33, m, c=CL[0], lw=2)
        plt.fill_between(tss33, m-s, m+s, color=CL[0], alpha=0.5)
        range_C2_lr[:,1] = np.array((np.min(m), np.max(m)))

        s = np.std(t0s_lr22_3,1)/perc
        m =  np.mean(t0s_lr22_3,-1)/perc
        plt.plot(tss33, m, c=CL[1], lw=2)
        plt.fill_between(tss33, m-s, m+s, color=CL[1], alpha=0.5)
        range_C2_lr[:,2] = np.array((np.min(m), np.max(m)))
        
        s = np.std(t0s_lr22_31,1)/perc
        m =  np.mean(t0s_lr22_31,-1)/perc
        plt.plot(tss33, m, c=CL[2], lw=2)
        plt.fill_between(tss33, m-s, m+s, color=CL[2], alpha=0.5)
        range_C2_lr[:,3] = np.array((np.min(m), np.max(m)))
        
        s = np.std(t0s_lr22_4,1)/perc
        m =  np.mean(t0s_lr22_4,-1)/perc
        plt.plot(tss33, m, c=CL[3], lw=2)
        plt.fill_between(tss33, m-s, m+s, color=CL[3], alpha=0.5)
        range_C2_lr[:,4] = np.array((np.min(m), np.max(m)))
        
        s = np.std(t0s_lr22_0,1)/perc
        m =  np.mean(t0s_lr22_0,-1)/perc
        plt.plot(tss33, m, c=C2_s, lw=2)
        plt.fill_between(tss33, m-s, m+s, color=C2_s, alpha=0.5)
        
        range_C2_lr[:,0] = np.array((np.min(m), np.max(m)))
        
        s = np.std(t0s_lr22_2,1)/perc
        m =  np.mean(t0s_lr22_2,-1)/perc
        plt.plot(tss33, m, c=C2_l, lw=2)
        plt.fill_between(tss33, m-s, m+s, color=C2_l, alpha=0.5)
        range_C2_lr[:,5] = np.array((np.min(m), np.max(m)))
        

        for xx in range(len(tss2)):
            
            plt.scatter(tss2[xx], tss2[xx], s=80, color=CLL2[xx,:], edgecolor='k', zorder=4, marker='s')
            plt.scatter(tss22[xx], tss22[xx], s=80, color=CLL2[xx+4,:], edgecolor='k', zorder=5, marker='s')
        
        plt.xlim([0, 6090])
        plt.ylim([0, 5500])
        ax.set_xlabel('input interval (ms)')
        ax.set_ylabel('output interval (ms)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.plot(tss2, tss2, '--k')
        #plt.legend()
        plt.savefig('Plots/FM_Fig5_E_22_B_Aug.pdf')   
        plt.show()
        
        np.savez('range_C2_lr', range_C2_lr)
        
        
                        #%%
        fig = plt.figure(figsize=[1.5*2.2, 1.5*2.*0.6])
        ax = fig.add_subplot(111)
        rang = range_C2_lr
        for iii in range(np.shape(rang)[1]):
            if iii>0 and iii<np.shape(rang)[1]-1:
                plt.plot( rang[:,iii], [0.025*iii+0.1,0.025*iii+0.1], c=CL[iii], lw=3, alpha=0.7)
                plt.plot( [np.mean(rang[:,iii]),np.mean(rang[:,iii])], [-0.03*0.2+0.025*iii+0.1,0.03*0.2+0.025*iii+0.1], c='k', lw=4, alpha=0.8, zorder=4)
            else:
                if iii==0:
                    plt.plot( rang[:,iii], [0.1*iii,0.*iii], c=C2_s, lw=4, alpha=0.7)
                    plt.plot( [np.mean(rang[:,iii]),np.mean(rang[:,iii])], [-0.03*0.2+0.1*iii,0.03*0.2+0.1*iii], c='k', lw=4, alpha=0.8, zorder=4)
                else:
                    plt.plot( rang[:,iii], [0.1,0.1], c=C2_l, lw=4, alpha=0.7)
                    plt.plot( [np.mean(rang[:,iii]),np.mean(rang[:,iii])], [-0.03*0.2+0.1,0.03*0.2+0.1], c='k', lw=4, alpha=0.8, zorder=4) 
            
            print(np.mean(rang[:,iii]))
        
        ax.set_yticks([-0.2, 0, 0.2])
        ax.set_yticklabels(['-0.2', '0', '0.2'])
        
        ax.set_ylim([-0.25, 0.25])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        #ax.set_ylim([-0.05,0.15])
        ax.set_ylabel('Ctxt')
        ax.set_xlabel('output range (ms)')
        #plt.savefig('Plots/FM_Fig5_D_Aug_Inset_2.pdf', transparent=True) 
        


        #%%

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(tss3, tss3, '--k')
        plt.plot(tss32, tss32, '--k', alpha=0.5)
        
        evenly_spaced_interval = np.linspace(0, 1, 8)
        cls2 = [cm.gist_yarg(x) for x in evenly_spaced_interval]
        
        CL = cls2[3:]
        
        perc = 0.85
        Lw=2
        Lw2=2.2
        ssiz = 10
        LLw = 0.1
        
        
        

        range_C3_lr = np.zeros((2,6))
        
        s = np.std(t0s_lr22_m1,1)/perc
        m =  np.mean(t0s_lr22_m1,-1)/perc
        plt.plot(tss33, m, c=CL[0], lw=2)
        plt.fill_between(tss33, m-s, m+s, color=CL[0], alpha=0.5)
        range_C3_lr[:,1] = np.array((np.min(m), np.max(m)))
                
        s = np.std(t0s_lr22_m2,1)/perc
        m =  np.mean(t0s_lr22_m2,-1)/perc
        plt.plot(tss33, m, c=CL[1], lw=2)
        plt.fill_between(tss33, m-s, m+s, color=CL[1], alpha=0.5)
        range_C3_lr[:,2] = np.array((np.min(m), np.max(m)))
        
        s = np.std(t0s_lr22_m3,1)/perc
        m =  np.mean(t0s_lr22_m3,-1)/perc
        plt.plot(tss33, m, c=CL[2], lw=2)
        plt.fill_between(tss33, m-s, m+s, color=CL[2], alpha=0.5)
        range_C3_lr[:,3] = np.array((np.min(m), np.max(m)))
        
        s = np.std(t0s_lr22_m4,1)/perc
        m =  np.mean(t0s_lr22_m4,-1)/perc
        plt.plot(tss33, m, c=CL[3], lw=2)
        plt.fill_between(tss33, m-s, m+s, color=CL[3], alpha=0.5)
        range_C3_lr[:,4] = np.array((np.min(m), np.max(m)))
        
        s = np.std(t0s_lr22_0,1)/perc
        m =  np.mean(t0s_lr22_0,-1)/perc
        plt.plot(tss33, m, c=C2_s, lw=2)
        plt.fill_between(tss33, m-s, m+s, color=C2_s, alpha=0.5)
        range_C3_lr[:,0] = np.array((np.min(m), np.max(m)))
        

        s = np.std(t0s_lr22_2,1)/perc
        m =  np.mean(t0s_lr22_2,-1)/perc
        plt.plot(tss33, m, c=C2_l, lw=2)
        plt.fill_between(tss33, m-s, m+s, color=C2_l, alpha=0.5)
        range_C3_lr[:,5] = np.array((np.min(m), np.max(m)))
            #     plt.scatter(tss33[xx]*np.ones(len(t0s_lr2[xx])), t0s_lr22_m1[xx]/perc, color=CL[0], s=ssiz, lw=LLw, edgecolor='k', alpha=1.)
        for xx in range(len(tss2)):
            plt.scatter(tss2[xx], tss2[xx], s=80, color=CLL2[xx,:], edgecolor='k', zorder=4, marker='s')
            plt.scatter(tss22[xx], tss22[xx], s=80, color=CLL2[xx+4,:], edgecolor='k', zorder=5, marker='s')
        
        # plt.xlim([0, 3900])
        # plt.ylim([0, 3900])
        plt.xlim([0, 6090])
        plt.ylim([0, 5500])
        
        ax.set_xlabel('input interval (ms)')
        ax.set_ylabel('output interval (ms)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.plot(tss2, tss2, '--k')
        #plt.legend()
        #plt.savefig('Plots/FM_Fig5_F_2_Aug.pdf')   
        plt.show()
        #np.savez('range_C3_lr', range_C3_lr)
        


      
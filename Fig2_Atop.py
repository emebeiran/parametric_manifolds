#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:41:40 2019

@author: mbeiran
"""
import numpy as np
import modules4 as md
import matplotlib.pyplot as plt
import torch
#import lib_rnns as lr
#import tools_MF as tm
from mpl_toolkits.mplot3d import Axes3D
import funcs_Sphere as fs
from matplotlib import cm

repeat = 1 # number of example trained networks. Can go up to 20

rank = 2

dt = 10#ms
tau = 100#ms

alpha = dt/tau
std_noise_rec = np.sqrt(2*alpha)*0.1

input_size = 2
hidden_size = 1500
output_size = 1

#initial connectivity
sigma_mn = 0.85

trials_train = 500
trials_test = 100

Nt = 350#2001
Nt2 = 550
time = np.arange(Nt)
time2 = np.arange(Nt2)


R_on  = 1000//dt#500//dt

#%%

# =============================================================================
#   Initialize inputs, outputs and recurrent connectivity (same as training initialization)
# =============================================================================
def give_vectors( sigma1, sigma2, s_m1, s_m2, s= 1, hidden_units = 1500, max_iter = 100, bn2 = 0.5):
    bigSigma = np.zeros((5,5)) #2*rank+input
    bigSigma[0,0] = s_m1
    bigSigma[1,1] = s_m2
    bigSigma[2,2] = 1.
    bigSigma[3,3] = 1.
    bigSigma[4,4] = s**2
    bigSigma[0,2] = sigma1
    bigSigma[2,0] = sigma1
    bigSigma[1,3] = sigma2
    bigSigma[3,1] = sigma2
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
            
    mean = np.zeros(5)
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
        
    M = X[:,0:2]
    N = X[:,2:4]
    I = X[:,4]
    return(M, N, I)

                


sigma1 = 0.8
sigma2 = 0.8
s_m1 = 1.
s_m2 = 1.
Mnaive, Nnaive, Inaive = give_vectors( sigma1, sigma2, s_m1, s_m2, s= 1, hidden_units = hidden_size, max_iter = 100, bn2 = 0.5)

Is_naive = np.random.randn()*Mnaive[:,0]+np.random.randn()*Mnaive[:,1]
O_naive = np.random.randn()*Mnaive[:,0]+np.random.randn()*Mnaive[:,1]
O_naive = O_naive[:,np.newaxis]

I_naive = np.vstack((Inaive, Is_naive))

# =============================================================================
#   Input intervals and input amplitudes
# =============================================================================

tss = np.array((800, 1550))
tss2 = np.array(( 800,  1050, 1300, 1550))

gain = 2.
tss4 = np.linspace(tss2[0],  tss2[0]+gain*(tss2[-1]-tss2[0]), 32)



amps = np.linspace(0, 0.25, 4)
amps4 = np.linspace(0, 0.25*gain, len(tss4))

N_steps = 5


Tss  = fs.gen_intervals(tss,N_steps)
Tss2 = fs.gen_intervals(tss2,N_steps)
Tss3 = fs.gen_intervals(tss2,N_steps)
Tss4 = fs.gen_intervals(tss4,N_steps)



#%%
train_ = False
 
time = np.arange(Nt)

# Parameters of task
SR_on = 60
factor = 1
dela = 120



# =============================================================================
#   Define colors
# =============================================================================
# Colors
cls2 = fs.set_plot()

cls2[1,:] = cls2[2,:]
cls2[2,:] = cls2[4,:]
cls2[3,:] = cls2[5,:]

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

cls4 = np.zeros((21,3))
cl11 = np.array((102, 153, 255))/255.
cl12 = np.array((53, 153, 53))/255.

cl21 = np.array((255, 204, 51))/255.
cl22 = np.array((204, 0, 0))/255.
#New colors
cl11 = np.array((71, 89, 156))/255.#p.array((102, 153, 255))/255.
cl12 = np.array((53, 153, 53))/255.

cl21 = np.array((255, 204, 51))/255.
cl22 = np.array((203, 81, 71))/255.#np.array((204, 0, 0))/255.

cls4[0,:] = 0.4*np.ones((3,))

cls4[1*3,:] = cl11
cls4[2*3,:] = 0.5*cl11+0.5*cl12
cls4[3*3,:] = cl12

cls4[4*3,:] = cl21
cls4[5*3,:] = 0.5*cl21+0.5*cl22
cls4[6*3,:] = cl22

# New colors April 2021
cls[3,:] = cl11#0.4*np.ones((3,))

cls[2,:] = cl21#0.5*cl11+0.5*cl12
cls[1,:] = cl12
cls[0,:] = cl22

for i in range(6):
    cls4[i*3+1,:] = (2./3)*cls4[i*3,:]+(1./3)*cls4[(i+1)*3,:]
    cls4[i*3+2,:] = (1./3)*cls4[i*3,:]+(2./3)*cls4[(i+1)*3,:]


#%%
# =============================================================================
#   Load data from networks (+ show code of how they are trained)
# =============================================================================
T0_lr_all = []
T0_fr_all = []
for tr in range(repeat):
    A = np.load('TrainedNets/net_CSG'+str(tr+1)+'.npz') #This is how the networks are initialized
    M = A['arr_0'] 
    N = A['arr_1'] 
    Is = A['arr_2']  
    Wo = A['arr_3']
    
    corrWo = 0.7
    Wo = corrWo*Wo[:,np.newaxis]/hidden_size
    
    dtype = torch.FloatTensor  
    mrec_i = M/np.sqrt(hidden_size)
    nrec_i = N/np.sqrt(hidden_size)
    mrec_I = torch.from_numpy(mrec_i).type(dtype)
    nrec_I = torch.from_numpy(nrec_i).type(dtype)
    inp_I = torch.from_numpy(Is.T).type(dtype)
    out_I = torch.from_numpy(Wo).type(dtype)
    
    mrec_naive_i = Mnaive/np.sqrt(hidden_size)
    nrec_naive_i = Nnaive/np.sqrt(hidden_size)
    mrec_naive_I = torch.from_numpy(mrec_i).type(dtype)
    nrec_naive_I = torch.from_numpy(nrec_i).type(dtype)
    inp_naive_I = torch.from_numpy(I_naive).type(dtype)
    out_naive_I = torch.from_numpy(O_naive/hidden_size).type(dtype)
    print('Repeat '+str(tr))
    i = N_steps-1
 # train only directly on longest intervals
        
        
    H0_MN, k0_ = fs.run_FP_fs(M, N, T = 180, dt = 0.2, trajs=1)
    if k0_[0]>0:
        H0_MN=-H0_MN
    dtype = torch.FloatTensor  
    h0_MN = torch.from_numpy(H0_MN).type(dtype)
    net_low_all = md.OptimizedLowRankRNN(input_size, hidden_size, output_size, std_noise_rec, alpha,
                                      rank=rank, train_wi = True, train_wo = True, train_h0=True, wi_init=inp_I, 
                                      wo_init=out_I, m_init=mrec_I, n_init=nrec_I, h0_init = h0_MN)
    net_low_fr = md.FullRankRNN(input_size, hidden_size, output_size, 0.1*std_noise_rec, alpha,
                                       train_wi = True, train_wo = True, train_h0=True)
    
    

    if train_ == True:
        input_train, output_train, mask_train, ct_train, ct2_train = fs.create_inp_out2(trials_train, Nt, Tss3[:,i]//dt, 
                                                                                amps, R_on, 100//dt, perc=0.1)
        torch.save(net_low_all.state_dict(), "TrainedNets/CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tr)+"_2int_gen_0.pt")
        
        #Train low-rank network -both inputs and outputs
        print('train all')
        loss1 = md.train(net_low_all, input_train, output_train[:,:,0:1], mask_train[:,:,0:1], n_epochs=80, plot_learning_curve=True, plot_gradient=True, 
                          lr=1e-3, clip_gradient = 1., keep_best=False, cuda=False, save_loss=True)
        np.savez("TrainedNets/CSG3_rank"+str(rank)+"_rep_"+str(tr)+"_2int_trainall_loss", loss1)
        torch.save(net_low_all.state_dict(), "TrainedNets/CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tr)+"_2int_gener_LRvsHR.pt")
        
        #Train Full rank
        print('train full-rank')
        torch.save(net_low_fr.state_dict(), "TrainedNets/CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tr)+"_2int_generInit_full.pt")
        loss1 = md.train(net_low_fr, input_train, output_train[:,:,0:1], mask_train[:,:,0:1], n_epochs=80, plot_learning_curve=True, plot_gradient=True, 
                  lr=1e-4, clip_gradient = 1., keep_best=False, cuda=False, save_loss=True)

        torch.save(net_low_fr.state_dict(), "TrainedNets/CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tr)+"_2int_gener_full.pt")
        
        
    else:
        
        net_low_all.load_state_dict(torch.load("TrainedNets/CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tr)+"_2int_gener_LRvsHR.pt", map_location='cpu'))
        M_pre, N_pre, corr_pre, I_pre, O_pre, J_pre = fs.get_SVDweights_CSG(net_low_all, rank=rank)
        
        net_low_fr.load_state_dict(torch.load("TrainedNets/CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tr)+"_2int_gener_full.pt", map_location='cpu'))
        
        Traj = []
        fig_width = 1.5*2.2 # width in inches
        fig_height = 0.8*1.5*2  # height in inches
        fig_size =  [fig_width,fig_height]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        
        trials = 10
        for xx in range(len(Tss3[:,i])):
            input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt, Tss3[:,i]//dt, amps,
                                                                        R_on, 1, just=xx, perc=0.)
            outp, traj = net_low_all.forward(input_tr, return_dynamics=True)
            outp = outp.detach().numpy()
            traj = traj.detach().numpy()
            Traj.append(traj)
            avg_outp0 = np.mean(outp[:,:,0],0)
            ax.plot(time*dt, avg_outp0, color=cls[xx,:], lw=2)            
            ax.plot(time*dt, output_tr.detach().numpy()[0,:,0], '--', color='k', alpha=0.5)
            #ax.plot(time*dt, input_tr.detach().numpy()[0,:,0], color=cls[xx,:], alpha=0.5)
                              
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('time (ms)')
        plt.ylabel('read out')
        
        string = 'CSG_trainedtrials_lowrank.pdf'
        print(string)
        #plt.savefig(string)
        plt.show()
        

        #%%
        fig_width = 1.5*2.2 # width in inches
        fig_height = 0.8*1.5*2  # height in inches
        fig_size =  [fig_width, fig_height]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        T0s_lr = []
        trials = 10
        xx_cl=0
        
        evenly_spaced_interval = np.linspace(0, 1, len(Tss4[:,i])+10)
        colors = [cm.viridis(x) for x in evenly_spaced_interval]
        
        for xx in range(len(Tss4[:,i])):
            input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt2, Tss4[:,i]//dt, amps4,
                                                                        R_on, 1, just=xx, perc=0.)
            outp, traj = net_low_all.forward(input_tr, return_dynamics=True)
            outp = outp.detach().numpy()
            t0s = time2[np.argmin(np.abs(outp-0.35),1)]*dt-1000
            T0s_lr.append(t0s)
            
            avg_outp0 = np.mean(outp[:,:,0],0)
            
            dTf = Tss4[1,i]-Tss4[0,i]
            
            if np.min(np.abs(Tss4[xx,i]-Tss2[:,i]))<0.5*dTf :
                CL = 0*np.ones(3)
                ax.plot(time2*dt, avg_outp0, color=CL, lw=2)            
            else:
                CL = colors[xx]#cl1*(len(Tss4[:,i])-xx)/len(Tss4[:,i]) + cl2*xx/len(Tss4[:,i])
                ax.plot(time2*dt, avg_outp0, color=CL, lw=1.5, alpha=0.5)            

                              
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('time (ms)')
        plt.ylabel('read out')
        string = 'CSG_generalization_lowrank.pdf'
        print(string)
        #plt.savefig(string)
        plt.show()
        


        #%%
        T0s_fr1 = []
        Traj_fr = []
        fig_width = 1.5*2.2 # width in inches
        fig_height = 0.8*1.5*2  # height in inches
        fig_size =  [fig_width,fig_height]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        
        trials = 10
        for xx in range(len(Tss3[:,i])):
            input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt, Tss3[:,i]//dt, amps,
                                                                        R_on, 1, just=xx, perc=0.)
            outp, traj2 = net_low_fr.forward(input_tr, return_dynamics=True)
            outp = outp.detach().numpy()
            
            mask_out = np.diff(outp[:,:,0])<0
            outp2 = np.copy(outp[:,1:,:])
            outp2[mask_out,0] = 1.
            iT = np.argmin(np.abs(outp2-0.35),1)+1
            
            t0s = time[iT]*dt-1000
            T0s_fr1.append(t0s)
            traj2 = traj2.detach().numpy()
            Traj_fr.append(traj2)
            
            avg_outp0 = np.mean(outp[:,:,0],0)
            ax.plot(time*dt, avg_outp0, color=cls[xx,:], lw=2)            
            ax.plot(time*dt, output_tr.detach().numpy()[0,:,0], '--', color=cls[xx,:])
            #ax.plot(time*dt, input_tr.detach().numpy()[0,:,0], color=cls[xx,:], alpha=0.5)
                              
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('time (ms)')
        plt.ylabel('read out')
        
        string = 'CSG_trainedtrials_fullrank.pdf'
        print(string)
        #plt.savefig(string)

        plt.show()
        #%%
        fig_width = 1.5*2.2 # width in inches
        fig_height = 0.8*1.5*2  # height in inches
        fig_size =  [fig_width,fig_height]
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        T0s_fr = []
        trials = 10
        xx_cl=0
        
        evenly_spaced_interval = np.linspace(0, 1, len(Tss4[:,i])+10)
        colors = [cm.viridis(x) for x in evenly_spaced_interval]
        
        for xx in range(len(Tss4[:,i])):
            input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt2, Tss4[:,i]//dt, amps4,
                                                                        R_on, 1, just=xx, perc=0.)
            outp, traj = net_low_fr.forward(input_tr, return_dynamics=True)
            outp = outp.detach().numpy()
            mask_out = np.diff(outp[:,:,0])<0
            outp2 = np.copy(outp[:,1:,:])
            outp2[mask_out,0] = 1.
            iT = np.argmin(np.abs(outp2-0.35),1)+1
            
            t0s = time2[iT]*dt-1000
            T0s_fr.append(t0s)
            
            avg_outp0 = np.mean(outp[:,:,0],0)
            
            dTf = Tss4[1,i]-Tss4[0,i]
            
            if np.min(np.abs(Tss4[xx,i]-Tss2[:,i]))<0.5*dTf :
                CL = 0*np.ones(3)
                ax.plot(time2*dt, avg_outp0, color=CL, lw=2)            

            else:
                CL = colors[xx]#cl1*(len(Tss4[:,i])-xx)/len(Tss4[:,i]) + cl2*xx/len(Tss4[:,i])
                ax.plot(time2*dt, avg_outp0, color=CL, lw=1.5, alpha=0.5)            
                              
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('time (ms)')
        plt.ylabel('read out')
        
        string = 'CSG_generalization_fullrank.pdf'
        print(string)
        plt.savefig(string)
        plt.show()
        

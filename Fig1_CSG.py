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
        
        #%%
        fig = plt.figure()
        ax = fig.add_subplot(111)
        c1 = 0.9*np.ones(3)
        c2 = 0.5*np.ones(3)
        for xx in range(len(Tss4[:,i])):
            if xx==0:
                plt.scatter(amps4[xx]*np.ones(trials), T0s_fr[xx]/0.85, color=c1, edgecolor=c2, s=30, alpha=1., label='full rank')
                plt.scatter(amps4[xx]*np.ones(trials), T0s_lr[xx]/0.85, color=c2, edgecolor='k', s=30, alpha=1., label='rank two')
            else:
                plt.scatter(amps4[xx]*np.ones(trials), T0s_fr[xx]/0.85, color=c1, edgecolor=c2, s=30, alpha=0.7)
                plt.scatter(amps4[xx]*np.ones(trials), T0s_lr[xx]/0.85, color=c2, edgecolor='k', s=30, alpha=0.7)
        plt.legend()
        for xx in range(len(tss2)):
            plt.scatter(amps[xx], tss2[xx], marker='s',s= 80, color=cls[xx,:], edgecolor='k', zorder=4)
        plt.plot(amps4, Tss4[:,i],'--k', lw=0.8)
        plt.ylim([0, 4000])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel(r'cue amplitude')
        plt.ylabel(r'$t_p$ (ms)')
        string = 'CSG_March3_rank'+str(rank)+'_inp_'+str(i)+"_rep_"+str(tr)+'_perf_comparison.pdf'
        print(string)
        plt.savefig(string)
        if tr==0:
            string = 'FM_Fig1_D.pdf'
            plt.savefig(string)
        
        T0_lr_all.append(T0s_lr)
        T0_fr_all.append(T0s_fr)

        
        
        #%% Remove inputs
        # Remove all components orthogonal to M and N vectors
        M_pre, N_pre, corr_pre, I_pre, O_pre, J_pre = fs.get_SVDweights_CSG(net_low_all, rank=rank)
        #%%
        I_bkg = I_pre[0,:]
        I_pulses = I_pre[1,:]       
        
        vecs_J = np.hstack((M_pre, N_pre))
        Q, R = np.linalg.qr(vecs_J)
        corr_Q=np.dot(I_bkg, Q)/np.sum(I_bkg)**2
        Proj_Q = np.dot(Q,Q.T)
        ProjI_bkg = np.dot(Proj_Q, I_bkg)
        nProjI_bkg = I_bkg-ProjI_bkg
        
        #%%
        I_lr = net_low_all.wi.detach().numpy()
        I_rnd = np.copy(I_lr)
        I_rnd[0,:] = ProjI_bkg
        I_rnd =  torch.from_numpy(I_rnd).type(dtype)
        
        net_low_all_s = md.OptimizedLowRankRNN(input_size, hidden_size, output_size, std_noise_rec, alpha,
                                      rank=rank, train_wi = True, train_wo = True, train_h0=True, wi_init=I_rnd, 
                                      wo_init=net_low_all.wo, m_init=net_low_all.m, n_init=net_low_all.n, h0_init = net_low_all.h0)

        
        I_rnd2 = np.copy(I_lr)
        I_rnd2[0,:] = nProjI_bkg 
        I_rnd2 =  torch.from_numpy(I_rnd2).type(dtype)
        net_low_all_ns = md.OptimizedLowRankRNN(input_size, hidden_size, output_size, std_noise_rec, alpha,
                                      rank=rank, train_wi = True, train_wo = True, train_h0=True, wi_init=I_rnd2, 
                                      wo_init=net_low_all.wo, m_init=net_low_all.m, n_init=net_low_all.n, h0_init = net_low_all.h0)
        
        #%%
        T0s_lr_ns = []
        T0s_lr_s  = []
        trials = 10
        xx_cl=0
        
        #%%
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        for xx in range(len(Tss4[:,i])):
            input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt2, Tss4[:,i]//dt, amps4,
                                                                        R_on, 1, just=xx, perc=0.)
            outp, traj = net_low_all_s.forward(input_tr, return_dynamics=True)
            outp = outp.detach().numpy()
            t0s = time2[np.argmin(np.abs(outp-0.35),1)]*dt-1000
            T0s_lr_s.append(t0s)
                                
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
            
        #%%
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for xx in range(len(Tss4[:,i])):
            input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt2, Tss4[:,i]//dt, amps4,
                                                                        R_on, 1, just=xx, perc=0.)
            outp, traj = net_low_all_ns.forward(input_tr, return_dynamics=True)
            outp = outp.detach().numpy()
            t0s = time2[np.argmin(np.abs(outp-0.35),1)]*dt-1000
            T0s_lr_ns.append(t0s)
            
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
        #%%
        fig = plt.figure()
        ax = fig.add_subplot(111)
        c1 = 0.9*np.ones(3)
        c2 = 0.5*np.ones(3)
        for xx in range(len(Tss4[:,i])):
            if xx==0:
                #plt.scatter(amps4[xx]*np.ones(trials), T0s_fr[xx]/0.85, color=c1, edgecolor=c2, s=30, alpha=1., label='full rank')
                plt.scatter(amps4[xx]*np.ones(trials), T0s_lr_s[xx]/0.85, color=c2, edgecolor='k', s=30, alpha=1., label='rank two')
            else:
                #plt.scatter(amps4[xx]*np.ones(trials), T0s_fr[xx]/0.85, color=c1, edgecolor=c2, s=30, alpha=0.7)
                plt.scatter(amps4[xx]*np.ones(trials), T0s_lr_s[xx]/0.85, color=c2, edgecolor='k', s=30, alpha=0.7)
        #plt.legend()
        for xx in range(len(tss2)):
            plt.scatter(amps[xx], tss2[xx], marker='s',s= 80, color=cls[xx,:], edgecolor='k', zorder=4)
        plt.plot(amps4, Tss4[:,i],'--k', lw=0.8)
        plt.ylim([500, 3000])
        plt.xlim([-0.02, 0.46])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel(r'cue amplitude')
        plt.ylabel(r'$t_p$ (ms)')
        string = 'CSG_March3_rank'+str(rank)+'_inp_'+str(i)+"_rep_"+str(tr)+'_perf_comparison.pdf'
        print(string)
        plt.savefig(string)
        if tr==0:
            string = 'FM_Fig6_B.pdf'
            plt.savefig(string)

        
        #%%
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for xx in range(len(Tss4[:,i])):
            if xx==0:
                #plt.scatter(amps[xx]*np.ones(trials), T0s_fr1[xx]/0.85, color=c1, edgecolor=c2, s=50,  alpha=1., label='full rank')
                plt.scatter(amps4[xx]*np.ones(trials), T0s_lr_ns[xx]/0.85, color=c2, edgecolor='k', s=50, alpha=1., label='rank two')
            else:
                #plt.scatter(amps[xx]*np.ones(trials), T0s_fr1[xx]/0.85, color=c1, edgecolor=c2, s=50, alpha=0.7)
                plt.scatter(amps4[xx]*np.ones(trials), T0s_lr_ns[xx]/0.85, color=c2, edgecolor='k', s=50, alpha=0.7)
        #plt.legend()
        #plt.scatter(amps, tss2, s= 50, color='k')
        for xx in range(len(tss2)):
            plt.scatter(amps[xx], tss2[xx], marker='s',s= 80, color=cls[xx,:], edgecolor='k', zorder=4)
        
        plt.plot(amps4, Tss4[:,i],'--k', lw= 0.8)
        plt.ylim([500, 3000])
        
        plt.xlim([-0.02, 0.46])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel(r'cue amplitude')
        plt.ylabel(r'$t_p$ (ms)')
        
        string = 'CSG_March3_rank'+str(rank)+'_inp_'+str(i)+"_rep_"+str(tr)+'_perf_comparison_trained.pdf'
        print(string)
        plt.savefig(string)
        if tr==0:
            string = 'FM_Fig6_B_2.pdf'
            plt.savefig(string)

        #%%
        C = 0
        countr= 0
        for xx in range(len(Tss2[:,i])):
            for tr in range(trials):
                sol = Traj_fr[xx][ tr,:,:]
                C  += sol.T.dot(sol)
                countr += 1
        C = C/countr
        evs_fr, Vs_fr = np.linalg.eigh(C)
        evs_fr = evs_fr[::-1]
        Vs_fr = Vs_fr[:,::-1]

        C = 0
        countr= 0
        for xx in range(len(Tss2[:,i])):
            for tr in range(trials):
                sol = Traj[xx][tr,:,:]
                C  += sol.T.dot(sol)
                countr += 1
        C = C/countr
        evs_lr, Vs_lr = np.linalg.eigh(C)
        evs_lr = evs_lr[::-1]
        Vs_lr = Vs_lr[:,::-1]

        #%% Recurrent space 
        k1s = np.linspace(-2, 2, 100)
        k2s = np.linspace(-2, 2, 100)
        Qss = []
        Rths = []
        St = []
        Ut =[]
        amps2 = np.array((0., 0.0833, 0.1666, 0.25, 0.45))
        for Amp in amps:
            print(Amp)
            G0, G1, Q, m1, m2, I_pre, J_pre = fs.get_field(net_low_all, k1s, k2s, Amp)  

            thetas = np.linspace(-np.pi, np.pi, 150)
            
            Qs, Rth, trajs1, trajs2, st_fp, u_fp = fs.get_manifold(thetas, Q, G0, G1, k1s, k2s, m1, m2, Amp, I_pre, J_pre)
            Qss.append(Qs)
            Rths.append(Rth)
            St.append(st_fp)
            Ut.append(u_fp)
            plt.pcolor(k1s, k2s, np.log10(Q),  vmin = -2., vmax = 0.5,shading='auto')
            plt.streamplot(k1s, k2s, G1, G0, color=[0.9, 0.9, 0.9], linewidth=1., density=0.8)
            plt.plot(Rth*np.sin(thetas), Rth*np.cos(thetas), c='k')
            
            for re in range(len(st_fp)):
                plt.scatter(st_fp[re][0],st_fp[re][1],c='k', edgecolor='w', s=50, zorder=4, lw=1.2)
            for re in range(len(u_fp)):
                plt.scatter(u_fp[re][0],u_fp[re][1],c='w', edgecolor='k', s=50, zorder=4, lw=1.2)
                    
            plt.show()
                    
#                 #%% Manifolds in 3D
#                 St = np.array(St)
#                 Ut = np.array(Ut)
#                 Dat = np.zeros((hidden_size, len(thetas), len(amps)))
                
#                 Proj = np.zeros((3, len(thetas), len(amps)))
#                 Dat2 = np.zeros((hidden_size, len(thetas)*len(amps)))

#                 DatUt = np.zeros((hidden_size, np.shape(Ut)[0], len(amps)))
#                 DatSt = np.zeros((hidden_size, np.shape(St)[0], len(amps)))
#                 ProjUt = np.zeros((3, np.shape(Ut)[0], len(amps)))
#                 ProjSt = np.zeros((3, np.shape(St)[0], len(amps)))                

#                 for ia, Amp in enumerate(amps):
#                     for it, th in enumerate(thetas):
#                         Dat[:,it, ia] = Rths[ia][it]*np.sin(th)*m1+Rths[ia][it]*np.cos(th)*m2+Amp*I_pre[0,:]
#                         Dat2[:,it+ ia*len(thetas)] = Rths[ia][it]*np.sin(th)*m1+Rths[ia][it]*np.cos(th)*m2+Amp*I_pre[0,:]
#                     for iS in range(np.shape(St)[1]):
#                         DatSt[:,iS, ia] = St[ia,iS,0]*m1+St[ia,iS,1]*m2+Amp*I_pre[0,:]
#                     for iS in range(np.shape(Ut)[1]):
#                         DatUt[:,iS, ia] = Ut[ia,iS,0]*m1+Ut[ia,iS,1]*m2+Amp*I_pre[0,:]

#                 #%%
#                 Y = Dat2
#                 C = np.dot(Y, Y.T)
#                 eh, vh = np.linalg.eigh(C)
#                 eh = eh[::-1]
#                 vh = vh[:,::-1]
                
#                 for ia, Amp in enumerate(amps):
#                     Proj[:,:,ia] = np.dot(vh[:,0:3].T, Dat[:,:,ia])
#                     for iS in range(np.shape(St)[1]):
#                         ProjSt[:,iS, ia] = np.dot(vh[:,0:3].T, DatSt[:,iS,ia])
#                     for iS in range(np.shape(Ut)[1]):
#                         ProjUt[:,iS, ia] = np.dot(vh[:,0:3].T, DatUt[:,iS,ia])
                            
#                 #%%
#                 fig = plt.figure()
#                 ax = fig.add_subplot(projection='3d', azim=-17, elev=20)
#                 for ia, Amp in enumerate(amps):
#                     ax.plot(Proj[0,:,ia], Proj[1,:,ia], Proj[2,:,ia], lw=2, color=cls[ia,:])
#                     mask = np.logical_and(thetas>-np.pi/20, thetas<np.pi/2+np.pi/10)
#                     ax.plot(Proj[0,mask,ia], Proj[1,mask,ia], Proj[2,mask,ia], lw=15, color='C1', alpha=0.3)
#                     for iS in range(np.shape(St)[0]):
#                         if np.sqrt(np.sum(ProjUt[:,iS,ia]**2))>20:
#                             ax.scatter(ProjSt[0,iS,ia],ProjSt[1,iS,ia],ProjSt[2,iS,ia], color=cls[ia,:], edgecolor='w', lw=1.5,s=50)
#                     for iS in range(np.shape(Ut)[0]):
#                         if np.sqrt(np.sum(ProjUt[:,iS,ia]**2))>20:
#                             ax.scatter(ProjUt[0,iS,ia],ProjUt[1,iS,ia],ProjUt[2,iS,ia], edgecolor=cls[ia,:], color='w', lw=1.2,s=30)
#                 ax.xaxis.set_ticklabels([])
#                 ax.yaxis.set_ticklabels([])
                
#                 ax.zaxis.set_ticks([0, 3.3, 6.6, 10])
#                 ax.zaxis.set_ticklabels(['0', '','','0.2'])
#                 ax.tick_params(axis="z",direction="in", pad=-2)
#                 ax.set_xlabel(r'$\kappa_1$', labelpad=-10)
#                 ax.set_ylabel(r'$\kappa_2$', labelpad=-10)
#                 ax.set_zlabel(r'cue input', labelpad=-10)
#                 ax.xaxis.pane.fill = False
#                 ax.yaxis.pane.fill = False
#                 ax.zaxis.pane.fill = False
                
#                 # Now set color to white (or whatever is "invisible")
#                 ax.xaxis.pane.set_edgecolor('w')
#                 ax.yaxis.pane.set_edgecolor('w')
#                 ax.zaxis.pane.set_edgecolor('w')
#                 ax.dist= 8.5
#                 plt.savefig('Plots/FM_Fig2_C.pdf')   
#                 plt.show()
                
#                 # Bonus: To get rid of the grid as well:
#                 #ax.grid(False)
                
#                 #%%
#                 readO = O_pre*hidden_size
#                 read1  = np.mean(m1*readO)/np.mean(m1**2) 
#                 read2 = np.mean(m2*readO)/np.mean(m2**2)
#                 #%%
#                 fig = plt.figure()
#                 ax = fig.add_subplot()
#                 for ia, Amp in enumerate(amps):
#                     mask = np.logical_and(thetas>-np.pi/20, thetas<np.pi/2+np.pi/10)
#                     ax.plot(Proj[0,:,ia], Proj[1,:,ia],  lw=2, color=cls[ia,:], zorder=2)
#                     ax.plot(Proj[0,mask,ia], Proj[1,mask,ia],  lw=20, color='C1', alpha=0.12)
                    
#                     for iS in range(np.shape(St)[0]):
#                         ax.scatter(ProjSt[0,iS,ia],ProjSt[1,iS,ia], color=cls[ia,:], edgecolor='w', lw=1.5,s=80, zorder=4)
#                     for iS in range(np.shape(Ut)[0]):
#                         ax.scatter(ProjUt[0,iS,ia],ProjUt[1,iS,ia], edgecolor=cls[ia,:], color='w', lw=1.2,s=50, zorder=4)
                
#                 plt.plot([0, 80], [0,0], c='k', lw=0.7)
#                 plt.plot([0, 60*np.cos(np.pi/3)], [0,60*np.sin(np.pi/3)], c='k', lw=0.7)
#                 RR =20
#                 import matplotlib.patches as patches
#                 style = "Simple, tail_width=0.5, head_width=4, head_length=8"
#                 kw = dict(arrowstyle=style, color="k")
#                 a3 = patches.FancyArrowPatch((RR, 0), (RR*np.cos(np.pi/3), RR*np.sin(np.pi/3)), 
#                              connectionstyle="arc3,rad=.2", **kw)
#                 ax.add_patch(a3)

#                 plt.xlim([-70, 70])
#                 plt.ylim([-45, 45])
#                 vals = 30000*np.linspace(-1, 1)
#                 ax.plot(read1*vals, read2*vals, '-', lw=6, color='grey', alpha=0.3)
#                 ax.set_xlabel(r'$\kappa_1$')
#                 ax.set_ylabel(r'$\kappa_2$')
#                 ax.set_xticks([-50, 0, 50])
#                 ax.set_xticklabels(['', '0', ''])
#                 ax.set_yticks([-30, 0, 30])
#                 ax.set_yticklabels(['', '0', ''])
#                 ax.spines['top'].set_visible(False)
#                 ax.spines['right'].set_visible(False)
#                 ax.yaxis.set_ticks_position('left')
#                 ax.xaxis.set_ticks_position('bottom')
#                 plt.savefig('Plots/FM_Fig2_B.pdf')   
#                 plt.show()
#                 #%%
#                 th0 = 0.
#                 fig = plt.figure()
#                 ax = fig.add_subplot(111)
                
#                 thetas_ = thetas+th0
#                 thetas_[thetas_>np.pi] = thetas_[thetas_>np.pi]-2*np.pi
#                 thetas_[np.argmin(np.abs(thetas_-np.pi))]= np.nan
#                 for ia, Amp in enumerate(amps):
#                     plt.plot(thetas_, -Qss[ia], color=cls[ia,:], lw=2)
#                     np.diff(Qss[ia])
#                     mask = np.abs(np.diff(np.sign(Qss[ia])))>0
#                     dth = thetas[1]-thetas[0]
#                     for it, th in enumerate(thetas[:-1]):
#                         if mask[it]:
#                             if np.diff(np.sign(Qss[ia]))[it]<0:
#                                 plt.scatter(thetas_[it], -Qss[ia][it],color='w', edgecolor=cls[ia,:], s=40, zorder=4) 
#                             else:
#                                 plt.scatter(thetas_[it], -Qss[ia][it], color=cls[ia,:], edgecolor='w', s=70, zorder=4) 
#                 plt.plot(thetas_, 0*thetas, color='k', lw=0.8)
#                 plt.fill_between([-np.pi/18, np.pi/2+np.pi/8], [-0.5, -0.5 ], [0.5, 0.5], color='C1', alpha=0.3)
#                 ax.spines['top'].set_visible(False)
#                 ax.spines['right'].set_visible(False)
#                 ax.yaxis.set_ticks_position('left')
#                 ax.xaxis.set_ticks_position('bottom')
#                 ax.set_xticks([-np.pi, -np.pi*0.5, 0, 0.5*np.pi, np.pi])
#                 ax.set_xticklabels([r'$-\pi$', '','0', '', r'$\pi$'])
                
#                 plt.xlabel(r'angle $\theta$')
#                 plt.ylabel(r'speed on manifold')
#                 plt.ylim([-0.48, 0.32])
#                 plt.savefig('Plots/FM_Fig2_D_previous.pdf')   
#                 plt.show()
#                 #%%
#                 th0 = 0.
#                 fig = plt.figure()
#                 ax = fig.add_subplot(111)
                
#                 thetas_ = thetas+th0
#                 thetas_[thetas_>np.pi] = thetas_[thetas_>np.pi]-2*np.pi
#                 thetas_[np.argmin(np.abs(thetas_-np.pi))]= np.nan
#                 for ia, Amp in enumerate(amps):
#                     plt.plot(thetas_, Qss[ia], color=cls[ia,:], lw=2)
#                     np.diff(Qss[ia])
#                     mask = np.abs(np.diff(np.sign(Qss[ia])))>0
#                     dth = thetas[1]-thetas[0]
#                     for it, th in enumerate(thetas[:-1]):
#                         if mask[it]:
#                             if np.diff(np.sign(Qss[ia]))[it]<0:
#                                 plt.scatter(0.5*(thetas_[it]+thetas_[it+1]), 0.,color='w', edgecolor=cls[ia,:], s=40, zorder=4) 
#                             else:
#                                 plt.scatter(0.5*(thetas_[it]+thetas_[it+1]), 0., color=cls[ia,:], edgecolor='w', s=70, zorder=4) 
#                 plt.plot(thetas_, 0*thetas, color='k', lw=0.8)
#                 plt.fill_between([-np.pi/18, np.pi/2+np.pi/8], [-0.5, -0.5 ], [0.5, 0.5], color='C1', alpha=0.3)
#                 ax.spines['top'].set_visible(False)
#                 ax.spines['right'].set_visible(False)
#                 ax.yaxis.set_ticks_position('left')
#                 ax.xaxis.set_ticks_position('bottom')
#                 ax.set_xticks([-np.pi, -np.pi*0.5, 0, 0.5*np.pi, np.pi])
#                 ax.set_xticklabels([r'$-\pi$', '','0', r'$\pi/2$', r'$\pi$'])
#                 ax.set_xlim([-0.3, 2.2])

                
#                 plt.xlabel(r'angle $\theta$')
#                 plt.ylabel(r'speed on manifold')
#                 plt.ylim([-0.08, 0.28])
#                 plt.savefig('Plots/FM_Fig2_D.pdf')   
#                 plt.show()
                
#                 #%%
#                 trials = 10
#                 Traj = []
#                 i = N_steps-1
#                 for xx in range(len(Tss3[:,i])):
#                     input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt, Tss3[:,i]//dt, amps,
#                                                                                 R_on, 1, just=xx, perc=0.)
#                     outp, traj = net_low_all.forward(input_tr, return_dynamics=True)
#                     traj = traj.detach().numpy()
#                     Traj.append(np.mean(traj,0))

#                 time = np.arange(Nt)
#                 ks = np.zeros((len(amps), 2, np.shape(Traj)[1]-1))
#                 for ia, Amp in enumerate(amps):
#                     for it, ti in enumerate(time):
#                         ks[ia, 0,it]= np.mean(m1*Traj[ia][it,:])/np.mean(m1*m1)
#                         ks[ia, 1,it]= np.mean(m2*Traj[ia][it,:])/np.mean(m2*m2)

#                 k1s = np.linspace(-2, 2, 100)
#                 k2s = np.linspace(-2, 3.5, 100)
#                 amps2 = np.array((0., 0.0833, 0.1666, 0.25, 0.45))
#                 Amp = amps[0]
#                 G0, G1, Q, m1, m2, I_pre, J_pre = fs.get_field(net_low_all, k1s, k2s, Amp)  
                
#                 thetas = np.linspace(-np.pi, np.pi, 150)
                    
#                 Qs, Rth, trajs1, trajs2, st_fp, u_fp = fs.get_manifold(thetas, Q, G0, G1, k1s, k2s, m1, m2, Amp, I_pre, J_pre)

#                 fig = plt.figure()
#                 ax = fig.add_subplot(111)
#                 plt.pcolor(k1s, k2s, np.log10(Q.T),  vmin = -2.5, vmax = 0.,shading='auto')#, cmap='gist_gray')
#                 cbar = plt.colorbar()
#                 cbar.set_ticks([-2, -1, 0])
#                 plt.streamplot(k1s, k2s, G0.T, G1.T, color=[0.9, 0.9, 0.9], linewidth=1., density=0.8)
#                 plt.plot(Rth*np.cos(thetas), Rth*np.sin(thetas), '--',c='k', lw=0.8)

#                 for ia, Amp in enumerate(amps):
#                     plt.plot(ks[ia, 0,50:], ks[ia, 1,50:], color=cls[ia,:], lw=2)                
#                 for ia, Amp in enumerate(amps):
#                     if ia == 0:
#                         for iS in range(np.shape(St)[1]):
#                             plt.scatter(St[ia,iS,1],St[ia,iS,0], zorder=4, color='w', edgecolor='k', lw=1.,s=50)
#                         for iS in range(np.shape(Ut)[1]):
#                             plt.scatter(Ut[ia,iS,1],Ut[ia,iS,0], zorder=4, edgecolor='w', color='none', lw=1.5,s=40)
#                 ax.set_xlabel(r'$\kappa_1$')
#                 ax.set_ylabel(r'$\kappa_2$')
#                 ax.set_ylim([-1.5, 3.5])
#                 plt.savefig('Plots/FM_Fig2_A.pdf')   
#                 plt.show()

#                 #%%
# net_low_fr.load_state_dict(torch.load("CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tr)+"_2int_gener_full.pt", map_location='cpu'))
 
# J_fr = net_low_fr.wrec.detach().numpy()

# net_low_fr0 = md.FullRankRNN(input_size, hidden_size, output_size, 0.1*std_noise_rec, alpha,
#                                                train_wi = True, train_wo = True, train_h0=True)
# net_low_fr0.load_state_dict(torch.load("CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tr)+"_2int_generInit_full.pt", map_location='cpu'))
# J0         = net_low_fr0.wrec.detach().numpy()

# #%%
# do_redRank = True
# repeatt = 10
# if do_redRank:
#     loss0s = np.zeros(repeat)
#     lossfs = np.zeros(repeat)
    
#     tran = 15
    
#     loss_rank = np.zeros((repeatt, tran))
    
#     for tR in range(repeatt):
#         print(tR)
        
#         input_train, output_train, mask_train, ct_train, ct2_train = fs.create_inp_out2(100, Nt, Tss3[:,i]//dt, 
#                                                                                         amps, R_on, 100//dt, perc=0.05)
#         net_low_fr0 = md.FullRankRNN(input_size, hidden_size, output_size, 0.1*std_noise_rec, alpha, 
#                                                    train_wi = True, train_wo = True, train_h0=True)
        
#         net_low_fr0.load_state_dict(torch.load("CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tR)+"_2int_generInit_full.pt", map_location='cpu'))
#         J0 = net_low_fr0.wrec.detach().numpy()  
#         print('initial')
#         loss0s[tR] = md.net_loss(net_low_fr0, input_train, output_train[:,:,0:1], mask_train[:,:,0:1], n_epochs=100, cuda=False)
#         net_low_fr.load_state_dict(torch.load("CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tR)+"_2int_gener_full.pt", map_location='cpu'))
#         J =  net_low_fr.wrec.detach().numpy() 
#         print('final')
#         lossfs[tR] = md.net_loss(net_low_fr, input_train, output_train[:,:,0:1], mask_train[:,:,0:1], n_epochs=100, cuda=False)
    
        
#         DJ = J-J0
#         u, s, v = np.linalg.svd(DJ, full_matrices=True)
#         for itr in range(tran):
#             print(itr)
#             itr+=1
#             J1 = J0 + u @ np.diag(s)[:, :itr] @ v[:itr,:]
#             net_low_R = md.FullRankRNN(input_size, hidden_size, output_size, 0.1*std_noise_rec, alpha, 
#                                        train_wi = True, train_wo = True, train_h0=True, wrec_init = torch.from_numpy(J1).type(dtype),
#                                        wi_init=net_low_fr.wi, wo_init=net_low_fr.wo, h0_init = net_low_fr.h0)
#             ll = md.net_loss(net_low_R, input_train, output_train[:,:,0:1], mask_train[:,:,0:1], n_epochs=100, cuda=False)
#             loss_rank[tR, itr-1] = ll
    
#     #%%
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     for tR in range(repeatt):
#         plt.plot(np.arange(tran)+1, loss_rank[tR,:]+1e-20, '-o', c='C1')      
#     plt.plot(np.arange(tran)+1, 0.01*np.ones(tran), lw=5, c='k', alpha=0.1)
#     ax.set_yscale('log')
#     ax.set_ylim([0.0001, 2.])
#     ax.set_xlabel('Reduced rank')
#     ax.set_ylabel('Loss')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_ticks_position('bottom')
#     plt.savefig('Plots/CSG_ReducedDimensionality.pdf')   
#     plt.show()
#     #%%
#     svd = np.zeros((repeat, hidden_size))
    
#     for tR in range(repeat):
#         print(tR)
        
#         input_train, output_train, mask_train, ct_train, ct2_train = fs.create_inp_out2(100, Nt, Tss3[:,i]//dt, 
#                                                                                         amps, R_on, 100//dt, perc=0.05)
#         net_low_fr0 = md.FullRankRNN(input_size, hidden_size, output_size, 0.1*std_noise_rec, alpha, 
#                                                    train_wi = True, train_wo = True, train_h0=True)
        
#         net_low_fr0.load_state_dict(torch.load("CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tR)+"_2int_generInit_full.pt", map_location='cpu'))
#         J0 = net_low_fr0.wrec.detach().numpy()  
#         net_low_fr.load_state_dict(torch.load("CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tR)+"_2int_gener_full.pt", map_location='cpu'))
#         J =  net_low_fr.wrec.detach().numpy() 
        
#         DJ = J-J0
#         u, s, v = np.linalg.svd(DJ, full_matrices=True)
#         svd[tR, :] = s
#     #%%
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     for tR in range(repeat):
#         plt.plot(np.arange(hidden_size)+1, svd[tR,:], '-o', c='C1')      
#     ax.set_xticks([1, 4, 7, 10])
#     ax.set_xlim([0,10])
#     ax.set_ylabel('Variance ')
#     ax.set_xlabel('# singular value')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_ticks_position('bottom')
#     plt.savefig('Plots/CSG_SingularValuesDJ.pdf')   
#     plt.show()

# #%%
# eig_J = np.linalg.eigvals(J_fr)
# eig_J0 = np.linalg.eigvals(J0)
# eig_DJ = np.linalg.eigvals(J_fr-J0)
# #%%
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(np.real(eig_J), np.imag(eig_J))
# plt.scatter(np.real(eig_J0), np.imag(eig_J0))
# #%%
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(np.real(eig_DJ), np.imag(eig_DJ))
# # #%%
# # u, sdj, v = np.linalg.svd(J_fr-J0)
# # plt.plot(sdj, '-o')
# # plt.xlim([0,20])

# #%%
# if train_ == False:
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.plot(100*evs_lr[0:21]/np.sum(evs_lr), '-o', label='low rank')
#     plt.plot(100*evs_fr[0:21]/np.sum(evs_fr), '-o', label='full rank')
#     plt.legend()
#     plt.xlim([0,20])
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.yaxis.set_ticks_position('left')
#     ax.xaxis.set_ticks_position('bottom')
#     plt.ylabel('variance explained (%)')
#     plt.xlabel('PC index')
#     plt.yscale('log')
#     plt.savefig('Plots/CSG_PCAdecomp.pdf')  
                    

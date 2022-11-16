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

#%%

tss = np.array((800, 1550))
tss2 = np.array(( 800,  1050, 1300, 1550))

gain = 2.




amps = np.linspace(0, 0.25, 4)
amps4 = np.linspace(-0.2, 0.25*gain, 20)

m = (tss2[-1]-tss2[0])/(amps[-1]-amps[0])
n = tss2[0]
tss4 = amps4*m+n

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
repeat = 10 #20 number of examples

# Colors
cls2 = fs.set_plot()

cls2[1,:] = cls2[2,:]
cls2[2,:] = cls2[4,:]
cls2[3,:] = cls2[5,:]

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
corr_OM_pre = np.zeros((rank, repeat))
corr_OM_pos = np.zeros((rank, repeat))

corr_bkgM_pre = np.zeros((rank, repeat))
corr_bkgM_pos = np.zeros((rank, repeat))

corr_pulM_pre = np.zeros((rank, repeat))
corr_pulM_pos = np.zeros((rank, repeat))

corr_bkgN_pre = np.zeros((rank, repeat))
corr_bkgN_pos = np.zeros((rank, repeat))

corr_pulN_pre = np.zeros((rank, repeat))
corr_pulN_pos = np.zeros((rank, repeat))

corr_eig_pre = np.zeros((rank, repeat))
corr_eig_pos = np.zeros((rank, repeat))

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
#%%
trials = 10
i = N_steps-1
try:
    A = np.load('TrainedNets/CSG_several.npz')
    T0_lr_all = A['arr_0']
    T0_fr_all = A['arr_1']
    Evs_all = A['arr_2']
    Evs_fr_all = A['arr_3']
    P_all = A['arr_4']
    P_fr_all = A['arr_5']
    P_all2 = A['arr_6']
    P_fr_all2 = A['arr_7']
        
except:    
    T0_lr_all = []
    T0_fr_all = []
    Evs_all = []
    Evs_fr_all = []
    Evs_all2 = []
    Evs_fr_all2 = []
    P_all = []
    P_fr_all= []
    P_all2 = []
    P_fr_all2= []    
    
    for tr in range(repeat):
    
        A = np.load('net_CSG'+str(tr+1)+'.npz')
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
        
        for i in range(N_steps):
            if i>N_steps-2 and tr>-1: # train only directly on longest intervals
                
                
                H0_MN, k0_ = fs.run_FP_fs(M, N, T = 180, dt = 0.2, trajs=1)
                if k0_[0]>0:
                    H0_MN=-H0_MN
                dtype = torch.FloatTensor  
                h0_MN = torch.from_numpy(H0_MN).type(dtype)
                net_low_all = md.OptimizedLowRankRNN(input_size, hidden_size, output_size, std_noise_rec, alpha,
                                                  rank=rank, train_wi = True, train_wo = True, train_h0=True, wi_init=inp_naive_I, 
                                                  wo_init=out_naive_I, m_init=mrec_naive_I, n_init=nrec_naive_I)
                net_low_fr = md.FullRankRNN(input_size, hidden_size, output_size, 0.1*std_noise_rec, alpha,
                                                train_wi = True, train_wo = True, train_h0=True)
                
                
    
    
                
                net_low_all.load_state_dict(torch.load("CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tr)+"_2int_gener_LRvsHR_June.pt", map_location='cpu'))
                M_pre, N_pre, corr_pre, I_pre, O_pre, J_pre = fs.get_SVDweights_CSG(net_low_all, rank=rank)
                
                net_low_fr.load_state_dict(torch.load("CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tr)+"_2int_gener_full.pt", map_location='cpu'))
                
                Traj = []
                fig_width = 1.5*2.2 # width in inches
                fig_height = 0.8*1.5*2  # height in inches
                fig_size =  [fig_width,fig_height]
                # fig = plt.figure(figsize=fig_size)
                # ax = fig.add_subplot(111)
                
                for xx in range(len(Tss3[:,i])):
                    input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt, Tss3[:,i]//dt, amps,
                                                                                R_on, 1, just=xx, perc=0.)
                    outp, traj = net_low_all.forward(input_tr, return_dynamics=True)
                    # outp = outp.detach().numpy()
                    traj = traj.detach().numpy()
                    Traj.append(traj)

                #% Generalization low_rank
                T0s_lr1 =[]
                fig_width = 1.5*2.2 # width in inches
                fig_height = 0.5*1.5*2  # height in inches
                fig_size =  [fig_width,fig_height]
                
                trials = 10
                
                for xx in range(len(Tss3[:,i])):
                    input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt, Tss3[:,i]//dt, amps,
                                                                                R_on, 1, just=xx, perc=0.)
                    outp, traj = net_low_all.forward(input_tr, return_dynamics=True)
                    outp = outp.detach().numpy()
                    t0s = time[np.argmin(np.abs(outp-0.35),1)]*dt-1000
                    T0s_lr1.append(t0s)
                    traj = traj.detach().numpy()
                    Traj.append(traj)
                    avg_outp0 = np.mean(outp[:,:,0],0)
                    Y = input_tr.detach().numpy()[0,:,0]
                    Y[0:10] = 0.
                    Y[-10:] = 0.
    
                
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
                string = 'CSG_March3_rank'+str(rank)+'_inp_'+str(i)+"_rep_"+str(tr)+'_perf_2int_gen_lr2_June.pdf'
                print(string)
                plt.savefig(string)
                plt.show()
                
    
    
                # Full rank normal
                T0s_fr1 = []
                Traj_fr = []
                fig_width = 1.5*2.2 # width in inches
                fig_height = 0.8*1.5*2  # height in inches
                fig_size =  [fig_width,fig_height]
                # fig = plt.figure(figsize=fig_size)
                # ax = fig.add_subplot(111)
                
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
                
                # Full rank several
                fig_width = 1.5*2.2 # width in inches
                fig_height = 0.8*1.5*2  # height in inches
                fig_size =  [fig_width,fig_height]
                # fig = plt.figure(figsize=fig_size)
                # ax = fig.add_subplot(111)
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

                c1 = 0.9*np.ones(3)
                c2 = 0.5*np.ones(3)

                T0_lr_all.append(T0s_lr)
                T0_fr_all.append(T0s_fr)
                # plt.show()
                #%     Do only for production
    
                off = -40#20
                T0 = 100
                net_low_all_nonoise = md.OptimizedLowRankRNN(input_size, hidden_size, output_size, 0.*std_noise_rec, alpha,
                                                  rank=rank, train_wi = True, train_wo = True, train_h0=True, train_wrec = True, m_init=net_low_all.m, n_init = net_low_all.n, 
                                                  wo_init=net_low_all.wo, h0_init = net_low_all.h0, wi_init=net_low_all.wi)
                
                net_low_fr_nonoise = md.FullRankRNN(input_size, hidden_size, output_size, 0*std_noise_rec, alpha, 
                                                            train_wi = True, train_wo = True, train_h0=True, wrec_init=net_low_fr.wrec, 
                                                  wo_init=net_low_fr.wo, h0_init = net_low_fr.h0, wi_init=net_low_fr.wi)
                
                net_low_all_nonoise
                
                trials = 10
                for xx in range(len(Tss3[:,i])):
                    input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt, Tss3[:,i]//dt, amps,
                                                                                R_on, 1, just=xx, perc=0.)
                    outp, traj = net_low_fr_nonoise.forward(input_tr, return_dynamics=True)
                    if xx==0:
                        Traj_fr = traj.detach().numpy()[:,T0+off:,:]
                    else:
                        Traj_fr = np.hstack((Traj_fr, traj.detach().numpy()[:,T0+off:,:]))
                    
                for xx in range(len(Tss3[:,i])):
                    input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt, Tss3[:,i]//dt, amps,
                                                                                R_on, 1, just=xx, perc=0.)
                    outp, traj = net_low_all_nonoise.forward(input_tr, return_dynamics=True)
                    if xx==0:
                        Traj = traj.detach().numpy()[:,T0+off:,:]
                    else:
                        Traj = np.hstack((Traj, traj.detach().numpy()[:,T0+off:,:]))
                    
    
                ##%%
                # neuron is the last index
                traj = Traj
                traj_fr = Traj_fr
                C=0
                for ii in range(np.shape(traj)[0]):
                    TT = traj[ii,:,:]
                    C += np.dot(TT.T,TT)
                C_fr=0
                for ii in range(np.shape(traj)[0]):
                    TT = traj_fr[ii,:,:]
                    C_fr += np.dot(TT.T,TT)
                    
                C2=0
                for ii in range(np.shape(traj)[0]):
                    TT = traj[ii,:,:]-np.mean(traj[ii,:,:],0)
                    C2 += np.dot(TT.T,TT)
                C_fr2=0
                for ii in range(np.shape(traj)[0]):
                    TT = traj_fr[ii,:,:]-np.mean(traj[ii,:,:],0)
                    C_fr2 += np.dot(TT.T,TT)
                # #%%
                # cls2 = fs.set_plot()
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                ev = np.linalg.eigvalsh(C)
                var=np.sum(ev)/(trials*hidden_size)
                ev = 100*ev[::-1]/np.sum(ev)
                P = np.sum(ev)**2/np.sum(ev**2)
                ev_fr = np.linalg.eigvalsh(C_fr)
                var_fr = np.sum(ev_fr)/(trials*hidden_size)
                ev_fr = 100*ev_fr[::-1]/np.sum(ev_fr)
                P_fr = np.sum(ev_fr)**2/np.sum(ev_fr**2)
                
                ev2 = np.linalg.eigvalsh(C2)
                var2=np.sum(ev2)/(trials*hidden_size)
                ev2 = 100*ev2[::-1]/np.sum(ev2)
                P2 = np.sum(ev2)**2/np.sum(ev2**2)
                ev_fr2 = np.linalg.eigvalsh(C_fr2)
                var_fr2 = np.sum(ev_fr2)/(trials*hidden_size)
                ev_fr2 = 100*ev_fr2[::-1]/np.sum(ev_fr2)
                P_fr2 = np.sum(ev_fr2)**2/np.sum(ev_fr2**2)
                

                Evs_all2.append(ev2)
                Evs_fr_all2.append(ev_fr2)
                P_all2.append(P2)
                P_fr_all2.append(P_fr2)
                Evs_all.append(ev)
                Evs_fr_all.append(ev_fr)
                P_all.append(P)
                P_fr_all.append(P_fr)                

            

    np.savez('CSG_several', T0_lr_all, T0_fr_all, Evs_all, Evs_fr_all, P_all, P_fr_all, P_all2, P_fr_all2)


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
ax.set_xticklabels(['rank two', 'full rank'])
ax.set_ylabel('participation ratio', fontsize=12)
plt.xticks(rotation = 25)
plt.yticks([1.5, 2., 2.5, 3.], labels=['','2', '', '3'])
plt.ylim([1.35, 3.1])
plt.xlim([-0.3, 1.3])

            # plt.xlim([0.5,7.5])
            # xa = np.linspace(1, 1000)
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig('PR_CSG.pdf', transparent=True)
plt.show()
            


#%%

def mean_outlier(t0s, val = 0.5):
    mean_sol = np.zeros(len(t0s))
    std_sol = np.zeros(len(t0s))
    
    for xx in range(len(t0s)):
        vals = np.array(t0s[xx])/0.85
        if np.std(vals)>val*np.mean(vals):
            vals[vals>np.mean(vals)] = np.nan
            
        mean_sol[xx] = np.nanmean(vals)
        std_sol[xx] = np.std(vals)
        
    return(mean_sol, std_sol  )


#%%
for tr in range(trials):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = 0
    m_fr = 0
    sm = 0
    sm_fr =0
    
    T0s_fr = T0_fr_all[tr]
    T0s_lr = T0_lr_all[tr]
    
    c1 = 0.9*np.ones(3)
    c2 = 0.5*np.ones(3)
    # for xx in range(len(Tss4[:,i])):
    #     if xx==0 and tr==0:
    #         plt.scatter(amps4[xx]*np.ones(trials), T0s_fr[xx]/0.85, color=c1, edgecolor=c2, s=30, alpha=1., label='full rank')
    #         plt.scatter(amps4[xx]*np.ones(trials), T0s_lr[xx]/0.85, color=c2, edgecolor='k', s=30, alpha=1., label='rank two')
    #     else:
    #         plt.scatter(amps4[xx]*np.ones(trials), T0s_fr[xx]/0.85, color=c1, edgecolor=c2, s=30, alpha=0.7)
    #         plt.scatter(amps4[xx]*np.ones(trials), T0s_lr[xx]/0.85, color=c2, edgecolor='k', s=30, alpha=0.
    #plt.plot(amps4,  mean_outlier(T0s_lr), c='k')
    m, sm  = mean_outlier(T0s_lr, val=0.2)
    m_fr, sm_fr  = mean_outlier(T0s_fr, val=0.2)
        
    plt.plot(amps4,  m_fr, '--',lw=2, c='k', label='full rank')
    plt.fill_between(amps4, m_fr-sm_fr, m_fr+sm_fr, color=0.7*np.ones(3))
    plt.plot(amps4,  m, lw=3, c='k', label='rank two')
    plt.fill_between(amps4, m-sm/np.sqrt(trials), m+sm/np.sqrt(trials), color=0.4*np.ones(3))
       
    for xx in range(len(tss2)):
        plt.scatter(amps[xx], tss2[xx], marker='s',s= 80, color=cls[xx,:], edgecolor='k', zorder=4)
    plt.plot(amps4, Tss4[:,i],'--k', lw=0.8)
    plt.ylim([0, 4000])
    plt.xlim([-0.22, 0.48])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel(r'cue amplitude')
    plt.ylabel(r'output interval (ms)')
    
    plt.legend()
    #plt.savefig('Performance_sev_CSG_'+str(tr)+'.pdf')
    plt.show()
    

 
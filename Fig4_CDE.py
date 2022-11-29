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
val0 = -0.2

amps = np.linspace(0, 0.25, 4)
n = 800
m = (tss2[-1]-tss2[0])/(amps[-1]-amps[0])
amps4 = np.linspace(val0, 0.25*gain, 32)#np.linspace(0, 0.25*gain, len(tss4))
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
repeat = 20 #20 number of examples

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
np.random.seed(21)

                #%% Repeat with good
tr_fr = 10
for tr in [5,]:

    A = np.load('TrainedNets/net_CSG'+str(tr+1)+'.npz')
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
            
            
            
            net_low_all.load_state_dict(torch.load("TrainedNets/CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tr)+"_2int_gener_LRvsHR_June.pt", map_location='cpu'))
            M_pre, N_pre, corr_pre, I_pre, O_pre, J_pre = fs.get_SVDweights_CSG(net_low_all, rank=rank)
            
            net_low_fr.load_state_dict(torch.load("TrainedNets/CSG3_rank"+str(rank)+"_prog_"+str(i)+"_rep_"+str(tr_fr)+"_2int_gener_full.pt", map_location='cpu'))
            
            Traj = []
            #%%
            fig_width = 1.5*2.2 # width in inches
            fig_height = 1.5*2  # height in inches
            fig_size =  [fig_width,fig_height]
            
            trials = 10
            for xx in range(len(Tss3[:,i])):
                input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt, Tss3[:,i]//dt, amps,
                                                                            R_on, 1, just=xx, perc=0.)
                outp, traj = net_low_all.forward(input_tr, return_dynamics=True)
                outp = outp.detach().numpy()
                traj = traj.detach().numpy()
                Traj.append(traj)
                
            #%%
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(111)
            
            trials = 10
            for xx in range(len(Tss3[:,i])):
                input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt, Tss3[:,i]//dt, amps,
                                                                            R_on, 1, just=xx, perc=0.)
                outp, traj = net_low_fr.forward(input_tr, return_dynamics=True)
                outp = outp.detach().numpy()
                avg_outp0 = np.mean(outp[:,:,0],0)
                if xx==0:
                    ax.plot(time*dt-1000, avg_outp0, '--', color='k', alpha=0.8, lw=2, label='full rank')   
                    ax.plot(time*dt-1000, avg_outp0, '--', color=cls[xx,:],lw=2)   
                    
                    ax.plot(time*dt-1000, output_tr.detach().numpy()[0,:,0], '.', color='k', alpha=0.5, zorder=5)
                    
                else:
                    ax.plot(time*dt-1000, avg_outp0, '--', color=cls[xx,:],lw=2)            
                    ax.plot(time*dt-1000, output_tr.detach().numpy()[0,:,0], '.', color='k', alpha=0.5, zorder=5)
            for xx in range(len(Tss3[:,i])):
                input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt, Tss3[:,i]//dt, amps,
                                                                            R_on, 1, just=xx, perc=0.)
                outp, traj = net_low_all.forward(input_tr, return_dynamics=True)
                outp = outp.detach().numpy()
                traj = traj.detach().numpy()
                Traj.append(traj)
                avg_outp0 = np.mean(outp[:,:,0],0)
                if xx==0:
                    ax.plot(time*dt-1000, avg_outp0, color='k', alpha=0.8,  lw=2, label='rank two')
                    ax.plot(time*dt-1000, avg_outp0, color=cls[xx,:], lw=2)
                    
                    plt.legend()
                else:
                    ax.plot(time*dt-1000, avg_outp0, color=cls[xx,:], lw=2)            
                #ax.plot(time*dt-1000, output_tr.detach().numpy()[0,:,0], '--', color='k', alpha=0.5)
                #ax.plot(time*dt, input_tr.detach().numpy()[0,:,0], color=cls[xx,:], alpha=0.5)
                                  
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.xlabel('time after Set (ms)')
            plt.ylabel('output')
            plt.xlim([-300, 1850])
            plt.yticks([-0.5, 0, 0.5])
            plt.xticks([0, 500, 1000, 1500])
            ax.set_xticklabels(['0', '', '1000', ''])
            
            
            # string = 'FM_Fig1_C_June.pdf'
            # print(string)
            # plt.savefig(string)
            plt.show()

           
            #%%     Do only for production            
            off = 20
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
                #plt.plot(traj.detach().numpy()[0,:,0])

            #%%
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
            #%%
            cls2 = fs.set_plot()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ev = np.linalg.eigvalsh(C)
            var=np.sum(ev)/(trials*hidden_size)
            ev = 100*ev[::-1]/np.sum(ev)
            P = np.sum(ev)**2/np.sum(ev**2)
            ev_fr = np.linalg.eigvalsh(C_fr)
            var_fr = np.sum(ev_fr)/(trials*hidden_size)
            ev_fr = 100*ev_fr[::-1]/np.sum(ev_fr)
            P_fr = np.sum(ev_fr)**2/np.sum(ev_fr**2)
            xa1 = np.arange(len(ev_fr))+1
            plt.plot(xa1, np.cumsum(ev_fr), '--', c='k', label='full rank', lw=2)
            plt.scatter(xa1,  np.cumsum(ev_fr), facecolor=0.5*np.ones(3), edgecolor='k', s=40, zorder=3)
            xa1 = np.arange(len(ev))+1
            plt.plot(xa1, np.cumsum(ev), c='k', label='rank two', lw=2)
            plt.scatter(xa1, np.cumsum(ev), c='k', s=30)
            
            plt.xlim([0.5,7.5])
            xa = np.linspace(1, 1000)
                
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.ylabel('cumulative variance (%)')
            plt.xlabel('# principal component')
            plt.xticks([1, 2, 3, 4, 5, 6, 7,  ])
            plt.legend()
            #plt.savefig('Plots/FM_Fig1_B1_Aug.pdf')  
            plt.show()
           

            #%% Manifolds in 3D - with generalization
            amps_j = np.array([0.        , 0.08333333, 0.16666667, 0.25 , 0.25+0.0833, 0.25+2*0.0833  ])
            
            k1s = np.linspace(-2, 2, 100)
            k2s = np.linspace(-2, 2, 100)
            Qss = []
            Rths = []
            St = []
            Ut =[]
            amps2 = np.array((0., 0.0833, 0.1666, 0.25, 0.45))
            for Amp in amps_j:
                print(Amp)
                G0, G1, Q, m1, m2, I_pre, J_pre = fs.get_field(net_low_all, k1s, k2s, Amp)  

                thetas = np.linspace(-np.pi, np.pi, 150)
                
                Qs, Rth, trajs1, trajs2, st_fp, u_fp = fs.get_manifold(thetas, Q, G0, G1, k1s, k2s, m1, m2, Amp, I_pre, J_pre, dim = 0.05)
                Qss.append(Qs)
                Rths.append(Rth)
                St.append(st_fp)
                Ut.append(u_fp)

            St = np.array(St)
            Ut = np.array(Ut)
            
            Dat = np.zeros((hidden_size, len(thetas), len(amps_j)))
            

            Proj = np.zeros((3, len(thetas), len(amps_j)))
            Dat2 = np.zeros((hidden_size, len(thetas)*len(amps_j)))

            DatUt = np.zeros((hidden_size, np.shape(Ut)[0], len(amps_j)))
            DatSt = np.zeros((hidden_size, np.shape(St)[0], len(amps_j)))
            ProjUt = np.zeros((3, np.shape(Ut)[0], len(amps_j)))
            ProjSt = np.zeros((3, np.shape(St)[0], len(amps_j)))                
            
            for ia, Amp in enumerate(amps_j):
                for it, th in enumerate(thetas):
                    Dat[:,it, ia] = Rths[ia][it]*np.sin(th)*m1+Rths[ia][it]*np.cos(th)*m2+Amp*I_pre[0,:]
                    Dat2[:,it+ ia*len(thetas)] = Rths[ia][it]*np.sin(th)*m1+Rths[ia][it]*np.cos(th)*m2+Amp*I_pre[0,:]
                for iS in range(np.shape(St)[1]):
                    DatSt[:,iS, ia] = St[ia,iS,0]*m1+St[ia,iS,1]*m2+Amp*I_pre[0,:]
                for iS in range(np.shape(Ut)[1]):
                    DatUt[:,iS, ia] = Ut[ia,iS,0]*m1+Ut[ia,iS,1]*m2+Amp*I_pre[0,:]

            Y = Dat2
            C = np.dot(Y, Y.T)
            eh, vh = np.linalg.eigh(C)
            eh = eh[::-1]
            vh = vh[:,::-1]
            
            for ia, Amp in enumerate(amps_j):
                Proj[:,:,ia] = np.dot(vh[:,0:3].T, Dat[:,:,ia])
                for iS in range(np.shape(St)[1]):
                    ProjSt[:,iS, ia] = np.dot(vh[:,0:3].T, DatSt[:,iS,ia])
                for iS in range(np.shape(Ut)[1]):
                    ProjUt[:,iS, ia] = np.dot(vh[:,0:3].T, DatUt[:,iS,ia])
                        
            #%%
            sign = -1
            TH1 = -np.pi*0.5+np.pi
            TH2 = 1.1*np.pi
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d', azim=-135, elev=33)
            thetas3 = np.hstack((thetas-np.pi*2, thetas, thetas+2*np.pi))
            for ia, Amp in enumerate(amps_j):
                
                xxx = sign*np.hstack((Proj[0,:,ia],Proj[0,:,ia],Proj[0,:,ia]))
                yyy = np.hstack((Proj[1,:,ia],Proj[1,:,ia],Proj[1,:,ia]))
                zzz = np.hstack((Proj[2,:,ia],Proj[2,:,ia],Proj[2,:,ia]))
                
                mask = np.logical_and(thetas3>0, thetas3<2*np.pi)
                if ia<4:
                    ax.plot(xxx[mask], yyy[mask], zzz[mask], lw=2, color=cls[ia,:])
                else:
                    ax.plot(xxx[mask], yyy[mask], zzz[mask],'--', lw=1.5,  color='k', alpha=0.8)
                
                
                mask = np.logical_and(thetas3>TH1, thetas3<TH2)
                ax.plot(xxx[mask], yyy[mask], zzz[mask], lw=15, color='C1', alpha=0.3)
                if ia<4:
                    for iS in range(np.shape(St)[0]):
                        if np.sqrt(np.sum(ProjSt[:,iS,ia]**2))>15:
                            ax.scatter(sign*ProjSt[0,iS,ia],ProjSt[1,iS,ia],ProjSt[2,iS,ia], color=cls[ia,:], edgecolor=[0.2,0.2,0.2],  zorder=4, lw=1.2,s=80)
                    for iS in range(np.shape(Ut)[0]):
                        if np.sqrt(np.sum(ProjUt[:,iS,ia]**2))>15:
                            ax.scatter(sign*ProjUt[0,iS,ia],ProjUt[1,iS,ia],ProjUt[2,iS,ia], edgecolor=cls[ia,:], color='w', lw=1.2,s=50, )
                else:
                    for iS in range(np.shape(St)[0]):
                        if np.sqrt(np.sum(ProjSt[:,iS,ia]**2))>10:
                            #print(np.sqrt(np.sum(sign*ProjSt[:,iS,ia]**2)))
                            ax.scatter(sign*ProjSt[0,iS,ia],ProjSt[1,iS,ia],ProjSt[2,iS,ia], color='k', edgecolor=[0.2,0.2,0.2],  zorder=4, lw=1.2,s=50, alpha=0.8)
                    for iS in range(np.shape(Ut)[0]):
                        if np.sqrt(np.sum(ProjUt[:,iS,ia]**2))>20:
                            ax.scatter(sign*ProjUt[0,iS,ia],ProjUt[1,iS,ia],ProjUt[2,iS,ia], edgecolor='k', color='w', lw=1.2,s=30, alpha=0.8)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            
            ax.zaxis.set_ticks([0, 3.3, 6.6, 10])
            ax.zaxis.set_ticklabels(['0', '','','0.2'])
            ax.tick_params(axis="z",direction="in", pad=-2)
            ax.set_xlabel(r'$\kappa_2$', labelpad=-10)
            ax.set_ylabel(r'$\kappa_1$', labelpad=-10)
            ax.set_zlabel(r'cue input', labelpad=-5)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            
            # Now set color to white (or whatever is "invisible")
            ax.xaxis.pane.set_edgecolor('w')
            ax.yaxis.pane.set_edgecolor('w')
            ax.zaxis.pane.set_edgecolor('w')
            ax.dist= 9.5
            plt.savefig('Fig4_D_2.pdf')   
            plt.show()            

            
            
            #%%
            readO = O_pre*hidden_size
            read1  = np.mean(m1*readO)/np.mean(m1**2) 
            read2 = np.mean(m2*readO)/np.mean(m2**2)

            fig = plt.figure()
            ax = fig.add_subplot()
            sign = -1
            for ia, Amp in enumerate(amps_j):
                xxx = sign*np.hstack((Rths[ia]*np.sin(thetas),Rths[ia]*np.sin(thetas),Rths[ia]*np.sin(thetas)))
                yyy = sign*np.hstack((Rths[ia]*np.cos(thetas),Rths[ia]*np.cos(thetas),Rths[ia]*np.cos(thetas)))
                #zzz = np.hstack((Proj[2,:,ia]/np.sqrt(hidden_size),Proj[2,:,ia]/np.sqrt(hidden_size),Proj[2,:,ia]/np.sqrt(hidden_size)))
                if ia<4:
                    mask = (thetas3>0)*(thetas3<2*np.pi)
                    ax.plot(xxx[mask], yyy[mask],  lw=2, color=cls[ia,:], zorder=2)
                    mask = np.logical_and(thetas3>TH1, thetas3<TH2)
                    ax.plot(xxx[mask], yyy[mask],  lw=20, color='C1', alpha=0.12)
                    
                    for re in range(len(St[ia])):
                        plt.scatter(sign*St[ia][re][0],sign*St[ia][re][1],color=cls[ia,:], edgecolor=[0.2,0.2,0.2],  zorder=4, lw=1.2, s=80)
                    for re in range(len(Ut[ia])):
                        plt.scatter(sign*Ut[ia][re][0],sign*Ut[ia][re][1],color='w', edgecolor=cls[ia,:],  zorder=4, lw=1.5, s=40)
                else:
                    mask = (thetas3>0)*(thetas3<2*np.pi)
                    ax.plot(xxx[mask], yyy[mask], '--', lw=1.5, color='k', zorder=2, alpha=0.5)
                    mask = np.logical_and(thetas3>TH1, thetas3<TH2)
                    ax.plot(xxx[mask], yyy[mask],  lw=20, color='C1', alpha=0.12)
                    
                    for re in range(len(St[ia])):
                        plt.scatter(sign*St[ia][re][0],sign*St[ia][re][1],color='k', edgecolor=[0.2,0.2,0.2],  zorder=4, lw=1.2, s=50, alpha=0.8)
                    for re in range(len(Ut[ia])):
                        plt.scatter(sign*Ut[ia][re][0],sign*Ut[ia][re][1],color='w', edgecolor='k',  zorder=4, lw=1.5, s=30, alpha=0.8)
                    
                #                         if np.sqrt(np.sum(ProjSt[:,iS,ia]**2))>15:
                #         ax.scatter(ProjSt[0,iS,ia],ProjSt[1,iS,ia],ProjSt[2,iS,ia], color=cls[ia,:], edgecolor='w', lw=1.5,s=80)
                # for iS in range(np.shape(Ut)[0]):
                #     if np.sqrt(np.sum(ProjUt[:,iS,ia]**2))>15:
                #         ax.scatter(ProjUt[0,iS,ia],ProjUt[1,iS,ia],ProjUt[2,iS,ia], edgecolor=cls[ia,:], color='w', lw=1.2,s=50, )
            plt.plot([0, -80], [0,0], c='k', lw=0.7)
            plt.plot([0, -60*np.cos(np.pi/3)], [0,60*np.sin(np.pi/3)], c='k', lw=0.7)
            RR =20
            import matplotlib.patches as patches
            style = "Simple, tail_width=0.5, head_width=4, head_length=8"
            kw = dict(arrowstyle=style, color="k")
            a3 = patches.FancyArrowPatch((RR, 0), (RR*np.cos(np.pi/3), RR*np.sin(np.pi/3)), 
                          connectionstyle="arc3,rad=.2", **kw)
            ax.add_patch(a3)

            plt.xlim([-1.3, 1.3])
            plt.ylim([-1.3, 1.3])
            vals = 30000*np.linspace(-1, 1)
            ax.plot(read2*vals, read1*vals, '-', lw=6, color='grey', alpha=0.3)
            ax.set_xlabel(r'$\kappa_1$')
            ax.set_ylabel(r'$\kappa_2$')
            ax.set_xticks([-1, 0, 1])
            #ax.set_xticklabels(['', '0', ''])
            ax.set_yticks([-1, 0, 1])
            #ax.set_yticklabels(['', '0', ''])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.savefig('Fig4_D1.pdf')   
            plt.show()
            
          
            
           #  #%%
           #  fig = plt.figure()
           #  ax = fig.add_subplot()
           #  sign = -1
           #  for ia, Amp in enumerate(amps_j):
           #      if ia==0:
           #          xxx = sign*np.hstack((Rths[ia]*np.sin(thetas),Rths[ia]*np.sin(thetas),Rths[ia]*np.sin(thetas)))
           #          yyy = sign*np.hstack((Rths[ia]*np.cos(thetas),Rths[ia]*np.cos(thetas),Rths[ia]*np.cos(thetas)))
           #          #zzz = np.hstack((Proj[2,:,ia]/np.sqrt(hidden_size),Proj[2,:,ia]/np.sqrt(hidden_size),Proj[2,:,ia]/np.sqrt(hidden_size)))
           #          if ia<4:
           #              mask = (thetas3>0)*(thetas3<2*np.pi)
           #              ax.plot(xxx[mask], yyy[mask],  lw=2, color=cls[ia,:], zorder=2)
           #              mask = np.logical_and(thetas3>TH1, thetas3<TH2)
           #              #ax.plot(xxx[mask], yyy[mask],  lw=20, color='C1', alpha=0.12)
                        
           #              for re in range(len(St[ia])):
           #                  plt.scatter(sign*St[ia][re][0],sign*St[ia][re][1],color=cls[ia,:], edgecolor=[0.2,0.2,0.2],  zorder=4, lw=1.2, s=80)
           #              for re in range(len(Ut[ia])):
           #                  plt.scatter(sign*Ut[ia][re][0],sign*Ut[ia][re][1],color='w', edgecolor=cls[ia,:],  zorder=4, lw=1.5, s=40)
           #          else:
           #              mask = (thetas3>0)*(thetas3<2*np.pi)
           #              ax.plot(xxx[mask], yyy[mask], '--', lw=1.5, color='k', zorder=2, alpha=0.5)
           #              mask = np.logical_and(thetas3>TH1, thetas3<TH2)
           #              #ax.plot(xxx[mask], yyy[mask],  lw=20, color='C1', alpha=0.12)
                        
           #              for re in range(len(St[ia])):
           #                  plt.scatter(sign*St[ia][re][0],sign*St[ia][re][1],color='k', edgecolor=[0.2,0.2,0.2],  zorder=4, lw=1.2, s=50, alpha=0.8)
           #              for re in range(len(Ut[ia])):
           #                  plt.scatter(sign*Ut[ia][re][0],sign*Ut[ia][re][1],color='w', edgecolor='k',  zorder=4, lw=1.5, s=30, alpha=0.8)
                        
           #      #                         if np.sqrt(np.sum(ProjSt[:,iS,ia]**2))>15:
           #      #         ax.scatter(ProjSt[0,iS,ia],ProjSt[1,iS,ia],ProjSt[2,iS,ia], color=cls[ia,:], edgecolor='w', lw=1.5,s=80)
           #      # for iS in range(np.shape(Ut)[0]):
           #      #     if np.sqrt(np.sum(ProjUt[:,iS,ia]**2))>15:
           #      #         ax.scatter(ProjUt[0,iS,ia],ProjUt[1,iS,ia],ProjUt[2,iS,ia], edgecolor=cls[ia,:], color='w', lw=1.2,s=50, )
           #  #plt.plot([0, -80], [0,0], c='k', lw=0.7)
           #  #plt.plot([0, -60*np.cos(np.pi/3)], [0,60*np.sin(np.pi/3)], c='k', lw=0.7)
           #  RR =20
           #  import matplotlib.patches as patches
           #  style = "Simple, tail_width=0.5, head_width=4, head_length=8"
           #  kw = dict(arrowstyle=style, color="k")
           #  a3 = patches.FancyArrowPatch((RR, 0), (RR*np.cos(np.pi/3), RR*np.sin(np.pi/3)), 
           #                connectionstyle="arc3,rad=.2", **kw)
           #  ax.add_patch(a3)

           #  plt.xlim([-1.3, 1.3])
           #  plt.ylim([-1.3, 1.3])
           #  vals = 30000*np.linspace(-1, 1)
           #  #ax.plot(read2*vals, read1*vals, '-', lw=6, color='grey', alpha=0.3)
           #  ax.set_xlabel(r'$\kappa_1$')
           #  ax.set_ylabel(r'$\kappa_2$')
           #  ax.set_xticks([-1, 0, 1])
           #  #ax.set_xticklabels(['', '0', ''])
           #  ax.set_yticks([-1, 0, 1])
           #  #ax.set_yticklabels(['', '0', ''])
           #  ax.spines['top'].set_visible(False)
           #  ax.spines['right'].set_visible(False)
           #  ax.yaxis.set_ticks_position('left')
           #  ax.xaxis.set_ticks_position('bottom')
           #  #plt.savefig('Plots/FM_Fig2_B_July_pre0.pdf')   
           # # plt.show()

            #%%
            th0 = 0.
            fig = plt.figure()
            ax = fig.add_subplot(111)
            

            thetas_ = thetas+th0
            thetas_[thetas_>np.pi] = thetas_[thetas_>np.pi]-2*np.pi
            thetas_[np.argmin(np.abs(thetas_-np.pi))]= np.nan
            for ia, Amp in enumerate(amps_j):
                if ia<4:
                    plt.plot(thetas_, -Qss[ia], color=cls[ia,:], lw=2)
                    np.diff(Qss[ia])
                    mask = np.abs(np.diff(np.sign(Qss[ia])))>0
                    dth = thetas[1]-thetas[0]
                    for it, th in enumerate(thetas[:-1]):
                        if mask[it]:
                            if np.diff(np.sign(Qss[ia]))[it]<0:
                                plt.scatter(thetas_[it], 0,color='w', edgecolor=cls[ia,:], s=40, zorder=4) 
                            else:
                                plt.scatter(thetas_[it], 0, color=cls[ia,:], edgecolor='w', s=70, zorder=4) 
                else:
                    plt.plot(thetas_, -Qss[ia], '--', color='k', lw=1.5, alpha=0.6)
                    np.diff(Qss[ia])
                    mask = np.abs(np.diff(np.sign(Qss[ia])))>0
                    dth = thetas[1]-thetas[0]
                    for it, th in enumerate(thetas[:-1]):
                        if mask[it]:
                            if np.diff(np.sign(Qss[ia]))[it]<0:
                                plt.scatter(thetas_[it], 0,color='w', edgecolor='k', s=30, zorder=4, alpha=0.6) 
                            else:
                                plt.scatter(thetas_[it], 0, color='k', edgecolor='w', s=50, zorder=4, alpha=0.6)                     
            plt.plot(thetas_, 0*thetas, color='k', lw=0.8)
            

            thetas_ = thetas+2*np.pi
            for ia, Amp in enumerate(amps_j):
                if ia<4:
                    plt.plot(thetas_, -Qss[ia], color=cls[ia,:], lw=2)
                    np.diff(Qss[ia])
                    mask = np.abs(np.diff(np.sign(Qss[ia])))>0
                    dth = thetas[1]-thetas[0]
                    for it, th in enumerate(thetas[:-1]):
                        if mask[it]:
                            if np.diff(np.sign(Qss[ia]))[it]<0:
                                plt.scatter(thetas_[it], 0,color='w', edgecolor=cls[ia,:], s=40, zorder=4) 
                            else:
                                plt.scatter(thetas_[it], 0, color=cls[ia,:], edgecolor='w', s=70, zorder=4) 
                else:
                    plt.plot(thetas_, -Qss[ia], '--', color='k', lw=1.5, alpha=0.6)
                    np.diff(Qss[ia])
                    mask = np.abs(np.diff(np.sign(Qss[ia])))>0
                    dth = thetas[1]-thetas[0]
                    for it, th in enumerate(thetas[:-1]):
                        if mask[it]:
                            if np.diff(np.sign(Qss[ia]))[it]<0:
                                plt.scatter(thetas_[it], 0,color='w', edgecolor='k', s=30, zorder=4, alpha=0.6) 
                            else:
                                plt.scatter(thetas_[it], 0, color='k', edgecolor='w', s=50, zorder=4, alpha=0.6)                     
            plt.plot(thetas_, 0*thetas, color='k', lw=0.8)
            
            
            plt.fill_between([TH1, TH2], [-0.5, -0.5 ], [0.5, 0.5], color='C1', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticks([-np.pi+np.pi, -np.pi*0.5+np.pi, 0+np.pi, 0.5*np.pi+np.pi, np.pi+np.pi])
            ax.set_xticklabels([r'$0$', '$0$','$-\pi/2$', '', r'$2\pi$'])
            ax.set_xlim([0.1*np.pi, 1.5*np.pi])
            plt.xlabel(r'angle $\theta$')
            plt.ylabel(r'speed on manifold')
            plt.ylim([-0.08, 0.35])
            plt.savefig('Fig4_E.pdf')   
            plt.show()


            
            #%%
            trials = 10
            Traj = []
            i = N_steps-1
            for xx in range(len(Tss3[:,i])):
                input_tr, output_tr, mask_tr, ct_train, ct2_train = fs.create_inp_out2(10, Nt, Tss3[:,i]//dt, amps,
                                                                            R_on, 1, just=xx, perc=0.)
                outp, traj = net_low_all.forward(input_tr, return_dynamics=True)
                traj = traj.detach().numpy()
                Traj.append(np.mean(traj,0))

            time = np.arange(Nt)
            ks = np.zeros((len(amps), 2, np.shape(Traj)[1]-1))
            for ia, Amp in enumerate(amps):
                for it, ti in enumerate(time):
                    ks[ia, 0,it]= np.mean(m1*Traj[ia][it,:])/np.mean(m1*m1)
                    ks[ia, 1,it]= np.mean(m2*Traj[ia][it,:])/np.mean(m2*m2)

            k1s = np.linspace(-2, 2, 100)
            k2s = np.linspace(-2, 2, 100)
            amps2 = np.array((0., 0.0833, 0.1666, 0.25, 0.45))
            Amp = amps[0]
            G0, G1, Q, m1, m2, I_pre, J_pre = fs.get_field(net_low_all, k1s, k2s, Amp)  
            
            thetas = np.linspace(-np.pi, np.pi, 150)
                
            Qs, Rth, trajs1, trajs2, st_fp, u_fp = fs.get_manifold(thetas, Q, G0, G1, k1s, k2s, m1, m2, Amp, I_pre, J_pre)

          
            
#%%
            fig = plt.figure()
            ax = fig.add_subplot(111)
            #plt.pcolor(sign*k1s, sign*k2s, np.log10(Q), vmin=-1.6, vmax = -0.25, shading='auto', alpha=0.4) #vmin = -1.5, vmax = -0.2,
            plt.grid(None)
            
            inT = 4
            inT2 = 3
            plt.quiver(sign*k1s[::inT], sign*k2s[::inT2], -G1[::inT2, ::inT], -G0[::inT2, ::inT], color=0.4*np.ones(3), units='width', pivot='mid', scale=4)
            
            #plt.streamplot(sign*k1s, sign*k2s, -G1, -G0, color=[0.9, 0.9, 0.9], linewidth=1., density=0.8)
            plt.plot(sign*Rth*np.sin(thetas), sign*Rth*np.cos(thetas),  color=cls[0,:], lw=2.5)

            for ia, Amp in enumerate(amps):
                if ia==0:
                    plt.plot(sign*ks[ia, 1,50:], sign*ks[ia, 0,50:], color='k', lw=2, zorder = 5)                
            for ia, Amp in enumerate(amps):
                if ia == 0:
                    for iS in range(np.shape(St)[1]):
                        plt.scatter(sign*St[ia,iS,0],sign*St[ia,iS,1], zorder=4, color=cls[ia,:], edgecolor=0.2*np.ones(3), lw=1.,s=70)
                    for iS in range(np.shape(Ut)[1]-1):
                        plt.scatter(sign*Ut[ia,iS,0],sign*Ut[ia,iS,1], zorder=4, edgecolor=cls[ia,:], color='w', lw=1.5,s=70)
                    plt.scatter(0,0, zorder=4, color='w', edgecolor=cls[ia, :], lw=1.5, s=70)
                    
            ax.set_xlabel(r'$\kappa_1$')
            ax.set_ylabel(r'$\kappa_2$')
            ax.set_ylim([-1.2, 1.2])
            ax.set_xlim([-1.5, 1.5])
            ax.set_xticks([-1,0,1])
            ax.set_yticks([-1,0,1])
            
            
            plt.savefig('Fig4_C.pdf')   
            plt.show()
            


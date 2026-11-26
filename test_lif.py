import numpy as np
from lif import default_params,run_LIF,analyse_spikes,Poisson_generator,synaptic_activation,run_LIF_EI

def test_rest():
    p=default_params(sim_len=100);v,s=run_LIF(p,0);assert np.allclose(v,p['E_L']) and len(s)==0

def test_spikes():
    p=default_params();v,s=run_LIF(p,250);assert len(s)>0 and np.all(np.diff(s)>=p['tref']);assert analyse_spikes(s,p['sim_len'])[0]==len(s)

def test_poisson_and_synapse():
    p=default_params(sim_len=10);x=Poisson_generator(p,0,3);assert not x.any();x[0,0]=1;z=synaptic_activation(p,x);assert z[0,1]==1 and z[0,2]<1

def test_no_synaptic_input():
    p=default_params(sim_len=100);z=np.zeros(len(p['t_steps']));v,s=run_LIF_EI(p,z,z,I_ext=0);assert np.allclose(v,p['E_L']) and not len(s)

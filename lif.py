"""Reusable numerical functions for the LIF model."""
import numpy as np

def default_params(**kwargs):
  params = {}


  # Params for neuron
  params['V_th'] = -55. # Threshold in mV
  params['V_reset'] = -75. # Potential at rest in mV
  params['V_init'] = -75. # Initial potential of membrane in mV
  params['g_L'] = 10. # Leak conductance in nS (nanosiemens)
  params['tau_m'] = 20. # Membrane time constant in ms
  params['E_L'] = -75. # Resting potential in mV
  params['tref'] = 2. # refractory period in ms

  # Params for simulation
  params['sim_len'] = 1000. # Total simulation duration in ms
  params['dt'] = 0.1 # Time step in ms

  # We should also allow individual params to be overwritten
  for k in kwargs:
    params[k] = kwargs[k]

  params['t_steps'] = np.arange(0,params['sim_len'],params['dt']) # Vector of discrete time points ms


  return params

def run_LIF (params, I):
  # set parameters
  V_th, V_reset, V_init, g_L, tau_m, E_L, tref = params['V_th'], params['V_reset'], params['V_init'], params['g_L'], params['tau_m'], params['E_L'], params['tref']

  t_steps, dt = params['t_steps'], params['dt']
  t = t_steps.size

  # initialise voltage and current
  v = np.zeros(t)
  v[0] = V_init
  I = I * np.ones(t)

  tr = 0

  # Simulate LIF dynamics

  rec_spikes = []

  for i in range(t-1):

    if tr>0:
      v[i] = V_reset
      tr = tr-1
    elif v[i] >= V_th:
      rec_spikes.append(i)
      v[i] = V_reset
      tr = tref/dt


    # calculate change in potential of membrane
    dv = (-(v[i] - E_L) + I[i] / g_L) * (dt / tau_m)

    v[i+1] = v[i] + dv
  rec_spikes = np.array(rec_spikes) * dt

  return v, rec_spikes

def analyse_spikes(spike_times, sim_len):

  # firing rate in Hz
  sim_len_s = sim_len / 1000
  firing_rate = len(spike_times) / sim_len_s

  # Interspike intervals in ms
  s = np.diff(spike_times)

  # CV of s
  if len(s) > 1:
    cv_s = np.std(s) / np.mean(s)
  else:
    cv_s = np.nan

  return firing_rate, s, cv_s

def gaussian_current (params, mu, sigma, seed = None):
  if seed is not None:
    np.random.seed(seed)

  dt = params['dt']
  t = params['t_steps'].size

  I = mu + sigma * np.random.randn(t) / np.sqrt(dt/1000.)

  return I

def Poisson_generator(params, rate, N):
  dt, t_steps = params['dt'], params['t_steps']
  t = t_steps.size

  rand_ = np.random.rand(N,t)

  spike_train = 1. * (rand_ < rate * (dt / 1000.))

  return spike_train

def synaptic_activation(params, spike_trains, tau_s = 5.0):
  dt = params['dt']

  N, t = spike_trains.shape
  P = np.zeros((N,t))

  decay = np.exp(-dt/tau_s)

  for i in range (t-1):
    # synapric activation decay
    P[:, i+1] = P[:,i] * decay
    # synapses that recieve a spike become active
    spiking_neurons = spike_trains[:,i] == 1

    P [spiking_neurons, i+1] = 1.0
  return P

def run_LIF_synaptic(params,g_total,E_s=0.0,I_ext=0.0):

    # Set params
    V_th, V_reset, V_init, g_L, tau_m, E_L = params['V_th'], params['V_reset'], params['V_init'],params['g_L'], params['tau_m'], params['E_L']
    tref = params['tref']
    dt = params['dt']
    t_steps = params['t_steps']
    t = t_steps.size

    # initialize the membrane voltage
    v = np.zeros(t)
    v[0] = V_init

    # and convert external current to array if it is a constant
    I_ext = I_ext * np.ones(t)

    # Refractory counter
    tr = 0

    rec_spikes = []

    for i in range(t - 1):
        if tr > 0:
            v[i] = V_reset
            tr = tr - 1
        elif v[i] >= V_th:
            rec_spikes.append(i)
            v[i] = V_reset
            tr = tref/dt

        # Conductance-based synaptic contribution
        synaptic_term = -(g_total[i] / g_L) * (v[i] - E_s)

        # LIF membrane equation we are working with
        dv = (-(v[i] - E_L) + synaptic_term + I_ext[i] / g_L) * (dt / tau_m)

        # Update the membrane voltage
        v[i + 1] = v[i] + dv

    # Convert spike indices to milliseconds
    rec_spikes = np.array(rec_spikes) * dt

    return v, rec_spikes

def run_LIF_EI(params, g_E, g_I, E_E=0.0, E_I=-80.0, I_ext=0.0):

    V_th, V_reset, V_init, g_L, tau_m, E_L = params['V_th'], params['V_reset'], params['V_init'],params['g_L'], params['tau_m'], params['E_L']
    tref = params['tref']
    dt = params['dt']
    t_steps = params['t_steps']
    t = t_steps.size

    v = np.zeros(t)
    v[0] = V_init
    I_ext = I_ext * np.ones(t)

    tr = 0
    rec_spikes = []

    for i in range(t - 1):
        if tr > 0:
            v[i] = V_reset
            tr = tr - 1
        elif v[i] >= V_th:
            rec_spikes.append(i)
            v[i] = V_reset
            tr = tref /dt

        excitatory_term = -(g_E[i] / g_L) * (v[i] - E_E)
        inhibitory_term = -(g_I[i] / g_L) * (v[i] - E_I)

        dv = (-(v[i] - E_L) + excitatory_term + inhibitory_term + I_ext[i] / g_L) * (dt / tau_m)

        v[i + 1] = v[i] + dv

    rec_spikes = np.array(rec_spikes) * dt

    return v, rec_spikes


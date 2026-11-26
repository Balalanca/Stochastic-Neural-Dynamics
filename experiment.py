"""Deterministic and stochastic LIF experiments. Units: ms, mV, pA, nS."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
_output=Path('results/notebook');_output.mkdir(parents=True,exist_ok=True)
_fig_num=0
def _save_show(*args,**kwargs):
    global _fig_num
    for number in plt.get_fignums():
        _fig_num+=1
        plt.figure(number).savefig(_output/f'figure_{_fig_num:02d}.png',dpi=130,bbox_inches='tight')
    plt.close('all')
plt.show=_save_show

# Experiment cell 1
import numpy as np
import matplotlib.pyplot as plt

# Experiment cell 4
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

# Experiment cell 6
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

# Experiment cell 8
params = default_params()

I_const = 250.0   # constant input current [pA]

v, spike_times = run_LIF(params, I_const)

print("Number of spikes:", len(spike_times))
print("Spike times [ms]:", spike_times)

# Experiment cell 10
plt.figure(figsize=(10,4))

plt.plot(params['t_steps'], v, label = 'membrane potential')

plt. axhline( params['V_th'], linestyle = '--' ,label = 'Threshold')

plt.xlabel('Time [ms]')
plt.ylabel('Membrane potential [mV]')
plt.title('LIF neuron with constant current')

plt.legend()
plt.show()



# Experiment cell 12
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

# Experiment cell 13
# And to test it
firing_rate, s, cv_s = analyse_spikes(spike_times,params['sim_len'])

print("Firing rate [Hz]:", firing_rate)
print("Mean s [ms]:", np.mean(s))
print("CV_s:", cv_s)

# Experiment cell 15
def gaussian_current (params, mu, sigma, seed = None):
  if seed is not None:
    np.random.seed(seed)

  dt = params['dt']
  t = params['t_steps'].size

  I = mu + sigma * np.random.randn(t) / np.sqrt(dt/1000.)

  return I

# Experiment cell 16
# and a quick test

I_gauss = gaussian_current(params, mu=200, sigma=2, seed=42)

plt.figure(figsize=(10, 4))

plt.plot(params['t_steps'][:1000], I_gauss[:1000])

plt.xlabel('Time [ms]')
plt.ylabel('Current [pA]')
plt.title('Gaussian noisy current')

plt.show()

# Experiment cell 18
v_gauss, spikes_gauss = run_LIF(params, I_gauss)

# Experiment cell 19
firing_rate_gauss, s_gauss, cv_gauss = analyse_spikes(spikes_gauss, params['sim_len'])

print("Firing rate [Hz]:", firing_rate_gauss)
print("Mean s [ms]:", np.mean(s_gauss))
print("CV_s:", cv_gauss)

plt.figure(figsize=(10, 4))

plt.plot(params['t_steps'], v_gauss)

plt.axhline(params['V_th'], linestyle='--', label='Threshold')

plt.xlabel('Time [ms]')
plt.ylabel('Membrane potential [mV]')
plt.title('LIF neuron with Gaussian noisy current')

plt.legend()
plt.show()

# Experiment cell 21
## First, vary miu, keep sigma fixed

mu_values = np.arange(100,301,20)

sigma_f = 2.0

rates_mu = []
cv_mu = []

for mu in mu_values:
  I_gauss = gaussian_current(params, mu=mu, sigma=sigma_f, seed=42)
  v_gauss, spikes_gauss = run_LIF(params, I_gauss)
  firing_rate_gauss, s_gauss, cv_gauss = analyse_spikes(spikes_gauss, params['sim_len'])

  rates_mu.append(firing_rate_gauss)
  cv_mu.append(cv_gauss)



# Experiment cell 22
plt.figure(figsize=(8, 4))

plt.plot(mu_values, rates_mu, marker='o')

plt.xlabel('Mean current μ [pA]')
plt.ylabel('Firing rate [Hz]')
plt.title('Effect of mean Gaussian current on firing rate')

plt.show()

# Experiment cell 23
plt.figure(figsize=(8, 4))

plt.plot(mu_values, cv_mu, marker='o')

plt.xlabel('Mean current μ [pA]')
plt.ylabel('CV_s')
plt.title('Effect of mean Gaussian current on firing regularity')

plt.show()

# Experiment cell 24
# Now we vary sigma, and keep miu fixed

sigma_values = np.arange(0,5.5, 0.5)

mu_f = 200.

rates_sigma = []
cv_sigma = []

for sigma in sigma_values:
  I_gauss = gaussian_current(params, mu=mu_f, sigma=sigma, seed=42)
  v_gauss, spikes_gauss = run_LIF(params, I_gauss)
  firing_rate_gauss, s_gauss, cv_gauss = analyse_spikes(spikes_gauss, params['sim_len'])

  rates_sigma.append(firing_rate_gauss)
  cv_sigma.append(cv_gauss)


# Experiment cell 25
plt.figure(figsize=(8, 4))

plt.plot(sigma_values, rates_sigma, marker='o')

plt.xlabel('Noise strength σ')
plt.ylabel('Firing rate [Hz]')
plt.title('Effect of Gaussian noise strength on firing rate')

plt.show()

# Experiment cell 26
plt.figure(figsize=(8, 4))

plt.plot(sigma_values, cv_sigma, marker='o')

plt.xlabel('Noise strength σ')
plt.ylabel('CV_s')
plt.title('Effect of Gaussian noise strength on firing regularity')

plt.show()

# Experiment cell 29
# Let's create a poisson spike generator

def Poisson_generator(params, rate, N):
  dt, t_steps = params['dt'], params['t_steps']
  t = t_steps.size

  rand_ = np.random.rand(N,t)

  spike_train = 1. * (rand_ < rate * (dt / 1000.))

  return spike_train

# Experiment cell 30
poisson_input = Poisson_generator(params, rate=10, N=100)
print(poisson_input.shape)

# Experiment cell 32
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

# Experiment cell 33
P = synaptic_activation(params, poisson_input, tau_s=5.0)
print(P.shape)


# Experiment cell 34
plt.figure(figsize=(10, 4))
plt.plot(params['t_steps'], P[0])
plt.xlabel('Time [ms]')
plt.ylabel('Synaptic activation P')
plt.title('Synaptic activation of one presynaptic neuron')
plt.show()

# Experiment cell 36
g_bar = 1 # arbitrary for now, but a sweep is performed later in the project

# Experiment cell 37
g_syn = g_bar * P
g_total = np.sum(g_syn, axis=0)
print(g_total.shape)
plt.figure(figsize=(10, 4))
plt.plot(params['t_steps'], g_total)
plt.xlabel('Time ms')
plt.ylabel('Total synaptic conductance')
plt.title('Total synaptic conductance from Poisson input')
plt.show()

# Experiment cell 39
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

# Experiment cell 40
v_syn, spikes_syn = run_LIF_synaptic(params, g_total, E_s=0.0, I_ext=0.0)

# Experiment cell 41
firing_rate_syn, s_syn, cv_syn = analyse_spikes(spikes_syn, params['sim_len'])

print("Firing rate [Hz]:", firing_rate_syn)
print("Mean s [ms]:", np.mean(s_syn) if len(s_syn) > 0 else np.nan)
print("CV_s:", cv_syn)
plt.figure(figsize=(10, 4))
plt.plot(params['t_steps'], v_syn)
plt.axhline(params['V_th'], linestyle='--', label='Threshold')
plt.xlabel('Time [ms]')
plt.ylabel('Membrane potential [mV]')
plt.title('LIF neuron driven by Poisson synaptic input')
plt.legend()
plt.show()

# Experiment cell 43
gbar_values = np.arange(0.1, 1.6, 0.1)

rates_gbar = []
cv_gbar = []

for g_bar in gbar_values:

    g_syn = g_bar * P
    g_total = np.sum(g_syn, axis=0)

    v_syn, spikes_syn = run_LIF_synaptic(params, g_total, E_s=0.0, I_ext=0.0)

    firing_rate_syn, s_syn, cv_syn = analyse_spikes(spikes_syn, params['sim_len'])

    rates_gbar.append(firing_rate_syn)
    cv_gbar.append(cv_syn)

# Experiment cell 44
plt.figure(figsize=(8, 4))
plt.plot(gbar_values, rates_gbar, marker='o')
plt.xlabel('Synaptic strength g_bar [nS]')
plt.ylabel('Postsynaptic firing rate [Hz]')
plt.title('Effect of synaptic strength on firing rate')
plt.show()
plt.figure(figsize=(8, 4))
plt.plot(gbar_values, cv_gbar, marker='o')
plt.xlabel('Synaptic strength g_bar [nS]')
plt.ylabel('CV_s')
plt.title('Effect of synaptic strength on firing regularity')
plt.show()

# Experiment cell 46
rate_values = [1, 2, 5, 10, 15, 20, 30, 40, 50]

rates_post = []
cv_post = []

# Experiment cell 47
for rate in rate_values:

    poisson_input = Poisson_generator(params, rate=rate, N=100)

    P = synaptic_activation(params, poisson_input, tau_s=5.0)

    g_syn = 1.0 * P
    g_total = np.sum(g_syn, axis=0)

    v_syn, spikes_syn = run_LIF_synaptic(params, g_total, E_s=0.0, I_ext=0.0)
    firing_rate_syn, s_syn, cv_syn = analyse_spikes(spikes_syn, params['sim_len'])
    rates_post.append(firing_rate_syn)
    cv_post.append(cv_syn)

# Experiment cell 48
plt.figure(figsize=(8, 4))
plt.plot(rate_values, rates_post, marker='o')
plt.xlabel('Presynaptic firing rate λ [Hz]')
plt.ylabel('Postsynaptic firing rate [Hz]')
plt.title('Presynaptic rate vs postsynaptic firing rate')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(rate_values, cv_post, marker='o')
plt.xlabel('Presynaptic firing rate λ [Hz]')
plt.ylabel('CV_s')
plt.title('Presynaptic rate vs firing irregularity')
plt.show()

# Experiment cell 50
poisson_input_ref = Poisson_generator(params, rate=10, N=100)
P_ref = synaptic_activation(params, poisson_input_ref, tau_s=5.0)
g_syn_ref = 1.0 * P_ref
g_total_ref = np.sum(g_syn_ref, axis=0)

v_poisson_ref, spikes_poisson_ref = run_LIF_synaptic(params, g_total_ref, E_s=0.0, I_ext=0.0)

rate_poisson_ref, s_poisson_ref, cv_poisson_ref = analyse_spikes(spikes_poisson_ref, params['sim_len'])

print("Poisson firing rate:", rate_poisson_ref)
print("Poisson CV_s:", cv_poisson_ref)

# Experiment cell 51
mu_match_values = np.arange(200, 401, 10)

gauss_match_rates = []
gauss_match_cv = []

for mu in mu_match_values:

    I_gauss = gaussian_current(params, mu=mu, sigma=2.0, seed=42)

    v_gauss, spikes_gauss = run_LIF(params, I_gauss)

    rate_gauss, s_gauss, cv_gauss = analyse_spikes(spikes_gauss, params['sim_len'])

    gauss_match_rates.append(rate_gauss)
    gauss_match_cv.append(cv_gauss)

# Experiment cell 52
for mu, rate, cv in zip(mu_match_values, gauss_match_rates, gauss_match_cv):
    print("mu =", mu, "rate =", rate, "CV =", cv)

# Experiment cell 54
I_gauss_match = gaussian_current(params, mu=280, sigma=2.0, seed=42)
v_gauss_match, spikes_gauss_match = run_LIF(params, I_gauss_match)
rate_gauss_match, s_gauss_match, cv_gauss_match = analyse_spikes(spikes_gauss_match, params['sim_len'])

# Experiment cell 55
plt.figure(figsize=(10, 4))
plt.plot(params['t_steps'], v_gauss_match)
plt.axhline(params['V_th'], linestyle='--', label='Threshold')
plt.xlabel('Time ms')
plt.ylabel('Membrane potential mV')
plt.title('Gaussian input, matched firing rate')
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(params['t_steps'], v_poisson_ref)
plt.axhline(params['V_th'], linestyle='--', label='Threshold')
plt.xlabel('Time ms')
plt.ylabel('Membrane potential mV')
plt.title('Poisson synaptic input, matched firing rate')
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(s_gauss_match, bins=10)
plt.xlabel('s ms')
plt.ylabel('Count')
plt.title('s distribution, Gaussian input')
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(s_poisson_ref, bins=10)
plt.xlabel('s ms')
plt.ylabel('Count')
plt.title('s distribution, Poisson synaptic input')
plt.show()

# Experiment cell 57
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

# Experiment cell 59
poisson_E = Poisson_generator(params, rate=10, N=80)
poisson_I = Poisson_generator(params, rate=10, N=20)

P_E = synaptic_activation(params, poisson_E, tau_s=5.0)
P_I = synaptic_activation(params, poisson_I, tau_s=5.0)

g_bar_E = 1.0
g_bar_I = 1.0
g_E = np.sum(g_bar_E * P_E, axis=0)
g_I = np.sum(g_bar_I * P_I, axis=0)

v_EI, spikes_EI = run_LIF_EI(params, g_E, g_I, E_E = 0.0, E_I = -80.0, I_ext = 0.0)

# Experiment cell 60
rate_EI, s_EI, cv_EI = analyse_spikes(spikes_EI, params['sim_len'])

print("Firing rate [Hz]:", rate_EI)
print("Mean s [ms]:", np.mean(s_EI) if len(s_EI) > 0 else np.nan)
print("CV_s:", cv_EI)
plt.figure(figsize=(10, 4))
plt.plot(params['t_steps'], v_EI)
plt.axhline(params['V_th'], linestyle='--', label='Threshold')
plt.xlabel('Time [ms]')
plt.ylabel('Membrane potential [mV]')
plt.title('LIF neuron with excitatory and inhibitory Poisson input')
plt.legend()
plt.show()

# Experiment cell 62
EI_conditions = [(100, 0), (80, 20), (60, 40), (50, 50), (40, 60), (20, 80)]

rates_EI = []
cv_EI = []

# Experiment cell 63
for N_E, N_I in EI_conditions:

    if N_E > 0:
        poisson_E=Poisson_generator(params, rate=10, N=N_E)
        P_E=synaptic_activation(params, poisson_E, tau_s=5.0)
        g_E=np.sum(1.0 * P_E, axis=0)
    else:
        g_E = np.zeros(params['t_steps'].size)


    if N_I > 0:
        poisson_I=Poisson_generator(params, rate=10, N=N_I)
        P_I=synaptic_activation(params, poisson_I, tau_s=5.0)
        g_I=np.sum(1.0 * P_I, axis=0)
    else:
        g_I = np.zeros(params['t_steps'].size)

    v_EI, spikes_EI = run_LIF_EI(params, g_E, g_I, E_E=0.0, E_I=-80.0, I_ext=0.0)

    rate_EI, s_EI, cv_value = analyse_spikes(spikes_EI, params['sim_len'])

    rates_EI.append(rate_EI)
    cv_EI.append(cv_value)

# Experiment cell 64
labels_EI = [f"{E}/{I}" for E, I in EI_conditions]
plt.figure(figsize=(8, 4))
plt.plot(labels_EI, rates_EI, marker='o')
plt.xlabel('Excitatory / Inhibitory presynaptic neurons')
plt.ylabel('Postsynaptic firing rate Hz')
plt.title('Effect of E/I composition on firing rate')
plt.show()


plt.figure(figsize=(8, 4))
plt.plot(labels_EI, cv_EI, marker='o')
plt.xlabel('Excitatory / Inhibitory presynaptic neurons')
plt.ylabel('CV_s')
plt.title('Effect of E/I composition on firing irregularity')
plt.show()

# Experiment cell 66
EI_conditions = [(100, 0), (80, 20), (60, 40), (50, 50), (40, 60), (20, 80)]

n_trials = 20

mean_rates_EI = []
std_rates_EI = []
mean_cv_EI = []
std_cv_EI = []

for N_E, N_I in EI_conditions:

    trial_rates = []
    trial_cv = []

    for trial in range(n_trials):

        if N_E > 0:
            poisson_E= Poisson_generator(params, rate= 10, N=N_E)
            P_E=synaptic_activation(params, poisson_E, tau_s=5.0)
            g_E=np.sum(1.0 * P_E, axis=0)
        else:
            g_E=np.zeros(params['t_steps'].size)


        if N_I > 0:
            poisson_I=Poisson_generator(params, rate=10, N=N_I)
            P_I=synaptic_activation(params, poisson_I, tau_s=5.0)
            g_I=np.sum(1.0 * P_I, axis=0)
        else:
            g_I=np.zeros(params['t_steps'].size)

        v_EI, spikes_EI=run_LIF_EI(params, g_E, g_I, E_E=0.0, E_I=-80.0, I_ext=0.0)

        rate_EI, s_EI, cv_value=analyse_spikes(spikes_EI, params['sim_len'])

        trial_rates.append(rate_EI)

        if not np.isnan(cv_value):
            trial_cv.append(cv_value)

    mean_rates_EI.append(np.mean(trial_rates))
    std_rates_EI.append(np.std(trial_rates))

    if len(trial_cv) > 0:
        mean_cv_EI.append(np.mean(trial_cv))
        std_cv_EI.append(np.std(trial_cv))
    else:
        mean_cv_EI.append(np.nan)
        std_cv_EI.append(np.nan)

# Experiment cell 67
labels_EI = [f"{E}/{I}" for E, I in EI_conditions]

plt.figure(figsize=(8, 4))
plt.errorbar(labels_EI, mean_rates_EI, yerr=std_rates_EI, marker='o', capsize=4)
plt.xlabel('Excitatory / Inhibitory presynaptic neurons')
plt.ylabel('Mean postsynaptic firing rate [Hz]')
plt.title('E/I composition vs firing rate')
plt.show()

plt.figure(figsize=(8, 4))
plt.errorbar(labels_EI, mean_cv_EI, yerr=std_cv_EI, marker='o', capsize=4)
plt.xlabel('Excitatory / Inhibitory presynaptic neurons')
plt.ylabel('Mean CV_s')
plt.title('E/I composition vs firing irregularity')
plt.show()

# Experiment cell 69
rate_values = [1, 2, 5, 10, 15, 20, 30, 40, 50]

n_trials = 20

mean_rates_pre = []
std_rates_pre = []
mean_cv_pre = []
std_cv_pre = []

for rate in rate_values:

    trial_rates = []
    trial_cv = []

    for trial in range(n_trials):

        poisson_input = Poisson_generator(params, rate=rate, N=100)
        P = synaptic_activation(params, poisson_input, tau_s=5.0)

        g_syn = 1.0 * P
        g_total = np.sum(g_syn, axis=0)

        v_syn, spikes_syn = run_LIF_synaptic(params, g_total, E_s=0.0, I_ext=0.0)
        firing_rate_syn, s_syn, cv_syn = analyse_spikes(spikes_syn, params['sim_len'])
        trial_rates.append(firing_rate_syn)

        if not np.isnan(cv_syn):
            trial_cv.append(cv_syn)

    mean_rates_pre.append(np.mean(trial_rates))
    std_rates_pre.append(np.std(trial_rates))

    if len(trial_cv) > 0:
        mean_cv_pre.append(np.mean(trial_cv))
        std_cv_pre.append(np.std(trial_cv))
    else:
        mean_cv_pre.append(np.nan)
        std_cv_pre.append(np.nan)

# Experiment cell 70
plt.figure(figsize=(8, 4))
plt.errorbar(rate_values, mean_rates_pre, yerr=std_rates_pre, marker='o', capsize=4)
plt.xlabel('Presynaptic firing rate λ [Hz]')
plt.ylabel('Mean postsynaptic firing rate [Hz]')
plt.title('Presynaptic rate vs postsynaptic firing rate')
plt.show()
plt.figure(figsize=(8, 4))
plt.errorbar(rate_values, mean_cv_pre, yerr=std_cv_pre, marker='o', capsize=4)
plt.xlabel('Presynaptic firing rate λ [Hz]')
plt.ylabel('Mean CV_s')
plt.title('Presynaptic rate vs firing irregularity')
plt.show()

# Experiment cell 72
n_trials = 10

rates_poisson_trials = []
cv_poisson_trials = []
rates_gaussian_trials = []
cv_gaussian_trials = []

for trial in range(n_trials):

    # Poisson synaptic input
    poisson_input = Poisson_generator(params, rate=10, N=100)
    P = synaptic_activation(params, poisson_input, tau_s=5.0)

    g_syn=1.0 * P
    g_total=  np.sum(g_syn, axis=0)

    v_poisson, spikes_poisson=run_LIF_synaptic(params, g_total, E_s=0.0, I_ext=0.0)
    rate_poisson, s_poisson, cv_poisson=analyse_spikes(spikes_poisson, params['sim_len'])
    rates_poisson_trials.append(rate_poisson)

    if not np.isnan(cv_poisson):
        cv_poisson_trials.append(cv_poisson)

    # Gaussian noisy input
    I_gauss=gaussian_current(params, mu=280, sigma=2.0)

    v_gauss,spikes_gauss=run_LIF(params, I_gauss)

    rate_gauss,s_gauss,cv_gauss=analyse_spikes(spikes_gauss, params['sim_len'])

    rates_gaussian_trials.append(rate_gauss)

    if not np.isnan(cv_gauss):
        cv_gaussian_trials.append(cv_gauss)

# Experiment cell 73
print("Poisson mean firing rate:", np.mean(rates_poisson_trials))
print("Poisson firing-rate std:", np.std(rates_poisson_trials))
print("Poisson mean CV:", np.mean(cv_poisson_trials))
print()

print("Gaussian mean firing rate:", np.mean(rates_gaussian_trials))
print("Gaussian firing-rate std:", np.std(rates_gaussian_trials))
print("Gaussian mean CV:", np.mean(cv_gaussian_trials))

# Experiment cell 74
labels = ['Poisson', 'Gaussian']
mean_rates = [np.mean(rates_poisson_trials), np.mean(rates_gaussian_trials)]
std_rates = [np.std(rates_poisson_trials), np.std(rates_gaussian_trials)]

plt.figure(figsize=(6, 4))
plt.bar(labels, mean_rates, yerr=std_rates, capsize=5)
plt.ylabel('Mean firing rate [Hz]')
plt.title('Matched firing rates: Poisson vs Gaussian')
plt.show()
mean_cv = [np.mean(cv_poisson_trials), np.mean(cv_gaussian_trials)]
std_cv = [np.std(cv_poisson_trials), np.std(cv_gaussian_trials)]

plt.figure(figsize=(6, 4))
plt.bar(labels, mean_cv, yerr=std_cv, capsize=5)
plt.ylabel('Mean CV_s')
plt.title('Firing irregularity: Poisson vs Gaussian')
plt.show()

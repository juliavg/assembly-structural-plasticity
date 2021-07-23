import numpy as np
import nest
import matplotlib.pyplot as plt
import pylab as pl

###################################################################################
# Parameters
###################################################################################

# General simulation parameters
dt                  = 0.1   # simulation resolution (ms)
MSP_update_interval = 100   # update interval for MSP (ms)

# Parameters for asynchronous irregular firing
g     = 12.0                # ratio between maximum amplitude of EPSP and EPSP
eta   = 1.5                 # ratio between external rate and external frequency needed for the mean input to reach threshold in absence of feedback
eps   = 0.1                 # connection probability for static connections (all but EE)
order = 100                 # order of network size
NE    = 4*order             # number of excitatory neurons
NI    = 1*order             # number of inhibitory neurons
N     = NE+NI               # total number of neurons
CE    = int(eps*NE)         # number of incoming excitatory synapses per inhibitory neuron
CI    = int(eps*NI)         # number of incominb inhibitory synapses per neuron  

# Growth
growth_time = 20000.            # growth time (ms)
cicles      = 10                # cicles for recording during growth
growth_step = growth_time/cicles

# Stimulation
stimulation_time     = 20000.   # stimulation time (ms)
stimulation_cicles   = 10       # cicles for recording during stimulation
stimulation_strength = 1.1      # modulation of external input firing rate during stimulation
stimulated_fraction  = 0.1      # fraction of excitatory neurons stimulated
stimulated_pop       = int(stimulated_fraction*NE)
stimulation_end      = growth_time + stimulation_time
stimulation_step     = stimulation_time / stimulation_cicles

# Post stimulation
post_stimulation_time = 20000. # time post stimulation (ms)
post_stimulation_cicles = 10   # cicles for recording post stimulation
post_stimulation_end = stimulation_end + post_stimulation_time
post_stimulation_step = post_stimulation_time / post_stimulation_cicles

# Decay
decay_time = 40000.             # time after post stimulation period (ms)
decay_cicles = 10               # cicles for recording after post stimulation period
decay_end = post_stimulation_end + decay_time
decay_step = decay_time / decay_cicles

# Parameters of the integrate and fire neuron
neuron_model = "iaf_psc_delta"
CMem         = 250.0                # membrane capacitance (pF)
tauMem       = 20.0                 # membrane time constant (ms)
theta        = 20.0                 # spike threshold (mV)
t_ref        = 2.                   # refractory period (ms)
E_L          = 0.                   # resting membrane potential (mV)
V_reset      = 10.                  # reset potential of the membrane (mV)
V_m          = 0.                   # initial membrane potential (mV)
tau_Ca       = 1000.                # time constant for calcium trace (ms)
beta_Ca      = 1./tau_Ca            # increment on calcium trace per spike (1/ms)
J            = 0.1                  # postsynaptic amplitude in mV
delay        = 1.                   # synaptic delay (ms)

neuron_params   = {
                    "C_m"       : CMem,
                    "tau_m"     : tauMem,
                    "t_ref"     : t_ref,
                    "E_L"       : E_L,
                    "V_reset"   : V_reset,
                    "V_m"       : V_m,
                    "beta_Ca"   : beta_Ca,
                    "tau_Ca"    : tau_Ca,
                    "V_th"      : theta
                   }

# External input rate
nu_th  = theta/(J*CE*tauMem)
nu_ex  = eta*nu_th
rate = 1000.0*nu_ex*CE
 
# Parameter for structural plasticity
growth_curve  = "linear"            # type of growth curve for synaptic elements
z0            = 1.                  # initial number of synaptic elements
slope         = -0.5                # slope of growth curve for synaptic elements
synapse_model = "static_synapse"    # plastic EE synapse type


###################################################################################
# Estimation of target rate for 10% connection probability
###################################################################################

simtime = 10000.
rectime = 8000.

nest.ResetKernel()
nest.SetDefaults(neuron_model, neuron_params)
pop_exc           = nest.Create(neuron_model, NE)
pop_inh           = nest.Create(neuron_model, NI)
poisson_generator = nest.Create('poisson_generator',params={'rate':rate})
spike_detector    = nest.Create('spike_detector',params={'start':simtime-rectime})
nest.Connect(pop_exc, pop_exc+pop_inh,{'rule': 'fixed_indegree','indegree': CE},syn_spec={"weight":J, "delay":delay})
nest.Connect(pop_inh, pop_exc+pop_inh,{'rule': 'fixed_indegree','indegree': CI},syn_spec={"weight":-g*J, "delay":delay})
nest.Connect(poisson_generator, pop_exc+pop_inh,'all_to_all',syn_spec={"weight":J, "delay":delay})
nest.Connect(pop_exc, spike_detector,'all_to_all')
nest.Simulate(simtime)

target_rate = nest.GetStatus(spike_detector,'n_events')[0]/rectime/NE       # target rate of excitatory neurons (/ms)


###################################################################################
# Simulation setup
###################################################################################

# Set Kernel
nest.ResetKernel()
nest.EnableStructuralPlasticity()
nest.SetKernelStatus({"resolution"                              : dt, 
                      "print_time"                              : True,
                      "structural_plasticity_update_interval"   : int(MSP_update_interval/dt)   # update interval for MSP in time steps
                      })

# Set model defaults
nest.SetDefaults(neuron_model, neuron_params)
nest.CopyModel(neuron_model, 'excitatory')
nest.CopyModel(neuron_model, 'inhibitory')
nest.CopyModel("static_synapse","device",{"weight":J, "delay":delay})
nest.CopyModel("static_synapse","inhibitory_synapse",{"weight":-g*J, "delay":delay})
nest.CopyModel("static_synapse","EI_synapse",{"weight":J, "delay":delay})
nest.CopyModel(synapse_model, 'msp_excitatory')
nest.SetDefaults('msp_excitatory',{'weight': J,'delay': delay})


# Assign synaptic elements with growth curve to excitatory neuron model
gc_den  = {'growth_curve': growth_curve, 'z': z0, 'growth_rate': -slope*target_rate, 'eps': target_rate, 'continuous': False}
gc_axon = {'growth_curve': growth_curve, 'z': z0, 'growth_rate': -slope*target_rate, 'eps': target_rate, 'continuous': False}
nest.SetDefaults('excitatory', 'synaptic_elements', {'Axon_exc': gc_axon, 'Den_exc': gc_den})

# Use SetKernelStatus to activate the plastic synapses
nest.SetKernelStatus({
    'structural_plasticity_synapses': {
        'syn1': {
            'model': 'msp_excitatory',
            'post_synaptic_element': 'Den_exc',
            'pre_synaptic_element': 'Axon_exc',
        }
    },
    'autapses': False,
})

# Create nodes
pop_exc               = nest.Create('excitatory', NE)
pop_inh               = nest.Create('inhibitory', NI)
poisson_generator_ex  = nest.Create('poisson_generator',2)
poisson_generator_inh = nest.Create('poisson_generator')
spike_detector        = nest.Create("spike_detector")

nest.SetStatus(poisson_generator_ex, {"rate": rate})
nest.SetStatus(poisson_generator_inh, {"rate": rate})
nest.SetStatus(spike_detector,{"withtime": True, "withgid": True})

# Connect nodes
nest.Connect(pop_exc, pop_inh,{'rule': 'fixed_indegree','indegree': CE},'EI_synapse')
nest.Connect(pop_inh, pop_exc+pop_inh,{'rule': 'fixed_indegree','indegree': CI},'inhibitory_synapse')
nest.Connect([poisson_generator_ex[0]], pop_exc[:stimulated_pop],'all_to_all', model="device")
nest.Connect([poisson_generator_ex[1]], pop_exc[stimulated_pop:],'all_to_all', model="device")
nest.Connect(poisson_generator_inh, pop_inh,'all_to_all',model="device")
nest.Connect(pop_exc+pop_inh, spike_detector,'all_to_all',model="device")

def simulate_cicle(growth_steps,global_index):
    step = np.diff(growth_steps)[0]
    for simulation_time in growth_steps:
        nest.Simulate(step)

        local_connections = nest.GetConnections(pop_exc, pop_exc)
        sources = np.array(nest.GetStatus(local_connections,'source'))
        targets = np.array(nest.GetStatus(local_connections,'target'))

        matrix = np.zeros((NE,NE))
        for ii in np.arange(sources.shape[0]):
            matrix[targets[ii]-1,sources[ii]-1] += 1
        connectivity[0,0,global_index] = np.mean(matrix[:stimulated_pop,:stimulated_pop])
        connectivity[0,1,global_index] = np.mean(matrix[:stimulated_pop,stimulated_pop:])
        connectivity[1,0,global_index] = np.mean(matrix[stimulated_pop:,:stimulated_pop])
        connectivity[1,1,global_index] = np.mean(matrix[stimulated_pop:,stimulated_pop:])

        events  = nest.GetStatus(spike_detector,'events')[0]
        times   = events['times']
        senders = events['senders']

        spike_count = np.histogram(senders,bins=np.array([0,stimulated_pop,NE,N]))[0]
        firing_rate[:,global_index] = spike_count/np.array([stimulated_pop,NE-stimulated_pop,NI]).astype(float)/step*1000.
        nest.SetStatus(spike_detector,'n_events',0)

        global_index += 1

    return matrix,global_index

# Create time steps and initialize recording arrays
growth_steps            = np.arange(growth_step,growth_time+1,growth_step)
stimulation_steps       = np.arange(growth_time+stimulation_step, stimulation_end+1, stimulation_step)
post_stimulation_steps  = np.arange(stimulation_end+post_stimulation_step, post_stimulation_end+1, post_stimulation_step)
decay_steps             = np.arange(post_stimulation_end+decay_step, decay_end+1, decay_step)
all_steps               = np.concatenate(([0],growth_steps,stimulation_steps,post_stimulation_steps,decay_steps))
connectivity            = np.zeros((2,2,all_steps.shape[0]))
firing_rate             = np.zeros((3,all_steps.shape[0]))
global_index            = 1


###################################################################################
# Simulate
###################################################################################

# Grow network
matrix_before,global_index = simulate_cicle(growth_steps,global_index)

# Stimulate
nest.SetStatus([poisson_generator_ex[0]], {"rate": rate*stimulation_strength})
matrix,global_index = simulate_cicle(stimulation_steps,global_index)

# Post stimulation
nest.SetStatus([poisson_generator_ex[0]], {"rate": rate})
matrix_post,global_index = simulate_cicle(post_stimulation_steps,global_index)

# Decay
matrix,global_index = simulate_cicle(decay_steps,global_index)


###################################################################################
# Plotting
###################################################################################

all_steps /= 1000.

fig = plt.figure(figsize=(10,7))

matrix_before = matrix_before[:int(NE/2),:int(NE/2)]      # plot matrix only for half of the excitatory population
matrix_post   = matrix_post[:int(NE/2),:int(NE/2)]        # plot matrix only for half of the excitatory population

max1    = np.max(matrix_before)
max2    = np.max(matrix_post)
max_syn = max(max1,max2)+1
cmap    = pl.cm.get_cmap('CMRmap_r',max_syn)

ax   = fig.add_subplot(2,2,1)
cax  = ax.imshow(matrix_before,cmap=cmap,interpolation="nearest")
cax.set_clim(-0.5,max_syn-0.5)
ax.set_xlabel("Pre")
ax.set_ylabel("Post")
ax.set_title("Time = %.f s" %(growth_steps[-1]/1000.))

ax   = fig.add_subplot(2,2,2)
cax  = ax.imshow(matrix_post,cmap=cmap,interpolation="nearest")
cbar = fig.colorbar(cax,ticks=np.arange(0,max_syn,1))
cbar.set_label("Number synapses")
cax.set_clim(-0.5,max_syn-0.5)
ax.set_title("Time = %.f s" %(post_stimulation_steps[-1]/1000.))

ax = fig.add_subplot(2,2,3)
ax.plot(all_steps,connectivity[0,0,:],'g',label=r'S$\to$S')
ax.plot(all_steps,connectivity[0,1,:],'orange',label=r'E$\to$S')
ax.plot(all_steps,connectivity[1,0,:],'gray',label=r'S$\to$E')
ax.plot(all_steps,connectivity[1,1,:],'b',label=r'E$\to$E')
ax.legend(loc=2)
ax.set_ylabel("Connection probability")
ax.set_xlabel("Time (s)")

ax = fig.add_subplot(2,2,4)
ax.plot(all_steps,firing_rate[0,:],'g',label='S')
ax.plot(all_steps,firing_rate[1,:],'b',label='E')
ax.plot(all_steps,firing_rate[2,:],'r',label='I')
ax.legend(loc=4)
ax.set_ylabel("Population rate (Hz)")
ax.set_xlabel("Time (s)")

fig.savefig('figure.pdf',format='pdf')

# -*- coding: utf-8 -*-
"""
RACE STRATEGY SIMULATION (RSS)

@author: Manuel Montesinos del Puerto
@email: m.montesinos.delpuerto@gmail.com
@phone: +34 619 58 12 64
"""

"""
MAIN SCRIPT. CLASS PROGRAMMING
"""

# %% 00_Set Configuration 
import matplotlib.pyplot as plt
plt.close('all')

from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')

from rss_class import RaceStrategySimulation

# %% 05_Options selector

# Data to train
years = [2017,2018,2019,2020]

# Own team and strategy
team_drivers = ['vettel','leclerc']
team_strategy = [1,2]

# Race circuit
sim_circuit = ['monaco']

# Race drivers and teams
me = ['hamilton','bottas']
fe = ['leclerc','sainz']

rb = ['max_verstappen','perez']
at = ['gasly','kvyat']

mc = ['norris','ricciardo']
al = ['alonso','ocon']

am = ['vettel','stroll']
ha = ['kevin_magnussen','grosjean']

wi = ['russell','sirotkin']
ar = ['raikkonen','giovinazzi']

sim_grid = me+fe+rb+at+mc+al+am+ha+wi+ar
del me,fe,rb,at,mc,al,am,ha,wi,ar

# sim_grid = ['hamilton','bottas','vettel','leclerc','sainz','norris']
# sim_grid = ['vettel','leclerc','hamilton','bottas','norris','sainz','gasly',
#             'kvyat','albon','hulkenberg','russell','ricciardo','giovinazzi',
#             'raikkonen','perez','ocon','alonso','grosjean','vandoorne',
#             'max_verstappen','stroll','kevin_magnussen']


# %% 100_START RSS
rss = RaceStrategySimulation()
rss.contact_info()

# %% 110_Importing data
# % Read inputs and import database
rss.read_inputs(years,team_drivers,team_strategy,sim_grid,sim_circuit)
drivers,d_dr,d_ra,d_qu,d_lt,d_ps,d_re,d_ci,d_st,matrix = rss.import_database()

# %% 120_Initialize parameters
matrix = rss.init_fuel_model()
matrix = rss.init_ltvar_model()
matrix = rss.init_dnf_model()

# %% 130_Simulation solver
srace,matrix = rss.race_starting_grid(1) # 0 = sim_grid, 1 = solver_grid
srace,sltime,sgrid = rss.race_simulation()

# %% 150_Results analysis
rss.analysis_race_results()


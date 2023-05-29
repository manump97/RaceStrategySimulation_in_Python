# -*- coding: utf-8 -*-
"""
RACE STRATEGY SIMULATION (RSS)

@author: Manuel Montesinos del Puerto
@email: m.montesinos.delpuerto@gmail.com
@phone: +34 619 58 12 64
"""

"""
CLASS RSS DEFINITION
"""
class RaceStrategySimulation():
    
    def contact_info(self):
        print('\n###### RACE STRATEGY SIMULATOR (RSS) ######')
        print('\n\n-------------- CONTACT INFO --------------') 
        print('Autor: Manuel Montesinos del Puerto')
        print('Email: m.montesinos.delpuerto@gmail.com')
        print('Phone: +34 619 58 12 65')
    
    
    # %% 05 read_inputs  
    def read_inputs(self,y,td,ts,simg,simc):
        self.train_year = y
        self.team_drivers = td
        self.team_strategy = ts
        self.sim_grid = simg
        self.sim_circuit = simc
        
        print('\n\n---------------- OPTIONS -----------------')
        print('Train year: ',self.train_year)
        print('Input team drivers: ',self.team_drivers)
        print('Input team strategy: ',self.team_strategy)
        print('Simulation circuit: ',self.sim_circuit)
        print('Simulation grid: ',self.sim_grid)
        
        
    # %% 100 import_database 
    def import_database(self):
        import pandas as pd
        import numpy as np
        print('\n\n----------- IMPORTING DATABASE -----------')
        # % RSS USEFUL
        data_drivers = pd.read_csv('f1_2020_data/drivers.csv')
        data_races = pd.read_csv('f1_2020_data/races.csv')
        
        data_qualifying = pd.read_csv('f1_2020_data/qualifying.csv')
        data_laptimes = pd.read_csv('f1_2020_data/lap_times.csv')
        data_pitstops = pd.read_csv('f1_2020_data/pit_stops.csv')
        data_results = pd.read_csv('f1_2020_data/results.csv')
        
        data_status = pd.read_csv('f1_2020_data/status.csv')
        data_circuits = pd.read_csv('f1_2020_data/circuits.csv')
        
        # % RSS USELESS
        data_driver_standings = pd.read_csv('f1_2020_data/driver_standings.csv')
        data_teams = pd.read_csv('f1_2020_data/constructors.csv')
        data_team_results = pd.read_csv('f1_2020_data/constructor_results.csv')
        data_team_standings = pd.read_csv('f1_2020_data/constructor_standings.csv')
        data_seasons = pd.read_csv('f1_2020_data/seasons.csv')
        
        # 10_Database pre-simulation processing
        # RSS USEFUL
        all_data_drivers = data_drivers
        data_drivers = data_drivers[data_drivers.driverRef.isin(self.sim_grid)]
        input_driverid = list(data_drivers.driverId)
        data_races = data_races[data_races.year.isin(self.train_year)]
        train_raceid = list(data_races.raceId)
        self.train_raceid = train_raceid
        
        data_qualifying = data_qualifying[data_qualifying.raceId.isin(train_raceid)&
                                          data_qualifying.driverId.isin(input_driverid)]
        data_laptimes = data_laptimes[data_laptimes.raceId.isin(train_raceid)&
                                      data_laptimes.driverId.isin(input_driverid)]
        data_pitstops = data_pitstops[data_pitstops.raceId.isin(train_raceid)&
                                      data_pitstops.driverId.isin(input_driverid)]
        data_results = data_results[data_results.raceId.isin(train_raceid)&
                                    data_results.driverId.isin(input_driverid)]
        
        # RSS USELESS
        data_driver_standings = data_driver_standings[data_driver_standings.raceId.isin(input_driverid)]
        data_team_results = data_team_results[data_team_results.raceId.isin(train_raceid)]
        data_team_standings = data_team_standings[data_team_standings.raceId.isin(train_raceid)]
        input_teams = list(data_team_results.constructorId)
        data_teams = data_teams[data_teams.constructorId.isin(input_teams)]
        
        self.data_drivers = data_drivers
        self.data_races = data_races
        self.data_qualifying = data_qualifying
        self.data_laptimes = data_laptimes
        self.data_pitstops = data_pitstops
        self.data_results = data_results
        self.data_circuits = data_circuits
        self.data_status = data_status
                
        # Drivers MATRIX and Circuit parameters
        matrix = np.zeros((8,len(self.sim_grid)))
        
        for i in np.arange(0,len(self.sim_grid),1):
            name = data_drivers[data_drivers.driverRef==self.sim_grid[i]]
            matrix[0,i] = name.driverId
        self.matrix = matrix
        
        input_driverid = list(matrix[0,:])
        self.input_driverid = input_driverid
        
        circuit = data_circuits[data_circuits.circuitRef.isin(self.sim_circuit)]
        sim_circuitid = circuit.circuitId
        self.sim_circuitid = sim_circuitid
        
        train_raceid = list(data_races.raceId)
        sim_raceid = data_races[data_races.circuitId.isin(sim_circuitid)]
        sim_raceid = list(sim_raceid.raceId)
        self.sim_raceid = sim_raceid
        
        # global sim_total_laps
        sim_total_laps = data_laptimes[data_laptimes.raceId.isin(sim_raceid)]
        sim_total_laps = sim_total_laps.lap
        sim_total_laps = max(sim_total_laps);
        self.sim_total_laps = sim_total_laps
        
        # Distinguishable 10 colors and linestyle
        
        mycolor = ['#191919','#191919','#d62728','#d62728',
                   '#9467bd','#9467bd','#e377c2','#e377c2',
                   '#ff7f0e','#ff7f0e','#1f77b4','#1f77b4',
                   '#2ca02c','#2ca02c','#bcbd22','#bcbd22',
                   '#17becf','#17becf','#8c564b','#8c564b',
                   '#a6761d','#a6761d','#2c7567','#2c7567'] 
        
        mylinestyle = ['-','--','-','--',
                       '-','--','-','--',
                       '-','--','-','--',
                       '-','--','-','--',
                       '-','--','-','--',]
        
        self.mylinestyle = mylinestyle
        self.mycolor = mycolor
    
        print('DONE <- Database F1 imported to train')
        
        return all_data_drivers,data_drivers,data_races,data_qualifying,data_laptimes,data_pitstops,data_results,data_circuits,data_status,matrix
    
    
    # %% 200 Init_fuel_model
    def init_fuel_model(self):
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LinearRegression
        print('\n\n----------- INIT PARAMETERS --------------')
        
        def qtime_str2sec(q):
            try:
                iim = q.find(':')
                iis = q.find('.')
                
                m = float(q[0:iim])    
                s = float(q[(iim+1):iis])
                ms = float(q[iis:])
            except AttributeError or ValueError:
                m = 3
                s = 0
                ms = 0
            
            return m*60+s+ms
        
        v_laps = np.arange(0,self.sim_total_laps+1,1) # Laps vector
        v0_fuel = (1-v_laps/(self.sim_total_laps))*100 # Fuel remain vector
        v0_fuel = np.delete(v0_fuel,0)
        
        matrix = self.matrix
        
        matb0 = np.ones((len(self.sim_raceid),len(self.sim_grid)))
        matb1 = np.ones((len(self.sim_raceid),len(self.sim_grid)))
        matrsq = np.ones((len(self.sim_raceid),len(self.sim_grid)))
        eit = np.ones((len(self.sim_raceid),len(self.sim_grid)))
        i = 0
        j = 0
        while j < len(self.sim_raceid):
            while i < len(self.sim_grid):
                qdata = self.data_qualifying[self.data_qualifying.driverId==int(self.input_driverid[i])]
                qdata = qdata[qdata.raceId==int(self.sim_raceid[j])]
                                
                v_fuel = v0_fuel
                
                if len(qdata)==1:
                    qlap = qdata[qdata.columns[6:9]]
                    qlap = pd.Series.tolist(qlap)
                    try:
                        qblap = min(min(qlap))
                        qblap = qtime_str2sec(qblap)
                    except TypeError:
                        qblap = min(qlap)
                        qblap = qblap[1]
                        try:
                            qblap = qtime_str2sec(qblap)
                        except AttributeError:
                            qblap = qblap[0]
                            qblap = qtime_str2sec(qblap)                    
                    
                    ltdata = self.data_laptimes[self.data_laptimes.driverId==int(self.input_driverid[i])]
                    ltdata = ltdata[ltdata.raceId==int(self.sim_raceid[j])]
                    
                    tlap = np.array(ltdata['milliseconds'])/1000
                    v_qlap = qblap*np.ones(len(tlap))
                    diff_blap = tlap-v_qlap
                    # print(diff_blap)
                    
                    if len(diff_blap) < len(v_fuel):
                        v_fuel = v_fuel[0:len(diff_blap)]
                    
                    
                    psdata = self.data_pitstops[self.data_pitstops.driverId==int(self.input_driverid[i])]
                    psdata = psdata[psdata.raceId==int(self.sim_raceid[j])]
                    nps = np.array(psdata['lap'])
                    nps = nps-1
                    
                    diff_blap = np.delete(diff_blap,nps)
                    diff_blap = np.delete(diff_blap,0)
                    
                    v_fuel = np.delete(v_fuel,nps)
                    v_fuel = np.delete(v_fuel,0)
                    
                    # print([len(v_fuel),len(diff_blap)])
                    # print([self.sim_raceid[j],self.input_driverid[i]],1)
                    # print('----------')
                    
                    v_fuel = v_fuel.reshape((-1,1))
                    LR_model = LinearRegression().fit(v_fuel,diff_blap)
                    
                    # r_sq = LR_model.score(v_fuel,diff_blap)
                    b0 = LR_model.intercept_
                    b1 = LR_model.coef_
                
                    # matrsq[j,i] = r_sq
                    matb0[j,i] = abs(b0)
                    matb1[j,i] = -abs(b1)
                    # eit[j,i] = list(diff_blap)
                else:
                    # print([0,0])
                    # print([self.sim_raceid[j],self.input_driverid[i]],0)
                    # print('----------')
                    
                    # matrsq[j,i] = 0
                    qblap = 1000
                    matb0[j,i] = 0
                    matb1[j,i] = 0
                    eit[j,i] = 0
                
                if j>0:
                    if qblap < matrix[1,i]:
                        matrix[1,i] = qblap
                else:
                    matrix[1,i] = qblap
                        
                i = i+1
                        
                
            j = j+1
            i = 0
        
        i = 0
        j = 0
        b0 = 0
        b1 = 0
        n = 0
        
        while i < len(self.sim_grid):
            while j < len(self.sim_raceid):
                if matb0[j,i] != 0 and matb0[j,i]<50:
                    n = n+1
                    b0 = b0+matb0[j,i]
                
                    if matb1[j,i] != 0:
                        b1 = b1+matb1[j,i]
                    
                j = j+1
            
            matrix[2,i] = b0/n
            matrix[3,i] = b1/n
            
            i = i+1
            j = 0
            b0 = 0
            b1 = 0
            n = 0        
        
        # self.eit = eit
        self.matrix = matrix   
        
        print('DONE <- Fuel model initialize')
        
        return matrix #,matb0,matb1,matrsq
    
    
    # %% 210 Init_ltvar_model
    def init_ltvar_model(self):
        import numpy as np
        import pandas as pd
        
        def qtime_str2sec(q):
            try:
                iim = q.find(':')
                iis = q.find('.')
                
                m = float(q[0:iim])    
                s = float(q[(iim+1):iis])
                ms = float(q[iis:])
            except AttributeError or ValueError:
                m = 3
                s = 0
                ms = 0
            
            return m*60+s+ms
        
        v_laps = np.arange(0,self.sim_total_laps+1,1) # Laps vector
        v0_fuel = (1-v_laps/(self.sim_total_laps))*100 # Fuel remain vector
        v0_fuel = np.delete(v0_fuel,0)
        
        matrix = self.matrix
        
        ltvar = np.ones((len(self.sim_raceid),len(self.sim_grid)))
        i = 0
        j = 0
        while j < len(self.sim_raceid):
            while i < len(self.sim_grid):
                qdata = self.data_qualifying[self.data_qualifying.driverId==int(self.input_driverid[i])]
                qdata = qdata[qdata.raceId==int(self.sim_raceid[j])]
                                
                v_fuel = v0_fuel
                
                if len(qdata)==1:
                    qlap = qdata[qdata.columns[6:9]]
                    qlap = pd.Series.tolist(qlap)

                    try:
                        qblap = min(min(qlap))
                        qblap = qtime_str2sec(qblap)
                    except TypeError:
                        qblap = min(qlap)
                        qblap = qblap[1]
                        try:
                            qblap = qtime_str2sec(qblap)
                        except AttributeError:
                            qblap = qblap[0]
                            qblap = qtime_str2sec(qblap)
                    
                    ltdata = self.data_laptimes[self.data_laptimes.driverId==int(self.input_driverid[i])]
                    ltdata = ltdata[ltdata.raceId==int(self.sim_raceid[j])]
                    
                    tlap = np.array(ltdata['milliseconds'])/1000
                    v_qlap = qblap*np.ones(len(tlap))
                    diff_blap = tlap-v_qlap
                    
                    if len(diff_blap) < len(v_fuel):
                        v_fuel = v_fuel[0:len(diff_blap)]
                        
                    
                    b0 = self.matrix[2,i]
                    b1 = self.matrix[3,i]
                    eit = diff_blap-b0-b1*v_fuel
                    
                    
                    psdata = self.data_pitstops[self.data_pitstops.driverId==int(self.input_driverid[i])]
                    psdata = psdata[psdata.raceId==int(self.sim_raceid[j])]
                    nps = np.array(psdata['lap'])
                    nps = nps-1
                    
                    # print([nps,len(eit)])
                    # print([self.sim_raceid[j],self.input_driverid[i]])
                    eit_lpv = np.delete(eit,nps)
                    eit_lpv = np.delete(eit_lpv,0)
                    
                    ltvar[j,i] = np.std(eit_lpv[eit_lpv<3])

                else:

                    ltvar[j,i] = 15
                
                        
                i = i+1
                        
                
            j = j+1
            i = 0
        
        
        
        i = 0
        while i < len(self.sim_grid):
            ltvar_races = ltvar[:,i]     
            best_ltvar = min(ltvar_races)
            
            matrix[4,i] = best_ltvar
            
            i = i+1
                 
        self.matrix = matrix   
                
        
        print('DONE <- LTvar model initialize')
        
        return matrix
    
    
    # %% 220_Init_dnf_model
    def init_dnf_model(self):
        import time
        import numpy as np
        # Formato
        # Columnas = Pilotos
        # Filas[1:] = nYear
        # 1r fila reservada para los driverId
        matrix = self.matrix
        train_year = self.train_year
        train_raceid = self.train_raceid
        
        dnf_matrix = np.zeros((6,len(self.sim_grid)))
        dnf_matrix[0,:] = matrix[0,:]
                
        for i in range(len(self.sim_grid)):
            datalt_d = self.data_laptimes[self.data_laptimes['driverId']==self.input_driverid[i]]
            dnf_driver = self.data_results[self.data_results['driverId']==self.input_driverid[i]]
            ndnf = 0
            total_laps = 0; tlaps = 0
            driven_laps = 0; dlaps = 0
            
            for j in range(len(train_raceid)):
                datalt_r = self.data_laptimes[self.data_laptimes['raceId']==int(train_raceid[j])]
                tlaps = max(list(datalt_r['lap']))
                                
                dnf_race = dnf_driver[dnf_driver['raceId']==int(train_raceid[j])]
                datalt_r = datalt_d[datalt_d['raceId']==int(train_raceid[j])]
                
                dlaps = len(list(datalt_r['lap']))
                
                if dlaps == 0:
                    tlaps = 0

                total_laps = total_laps + tlaps
                driven_laps = driven_laps + dlaps
                
                if len(dnf_race)==0:
                    pass
                
                else:
                    try:
                        dnf_status = int(dnf_race['statusId'])
                    except TypeError:
                        dnf_status = -1
                    
                    if dnf_status in [1,11,12,13]:
                        pass
                    else:
                        ndnf = ndnf+1                    
                
            dnf_matrix[1,i] = ndnf
            dnf_matrix[2,i] = (ndnf+1)/len(train_raceid)
            dnf_matrix[3,i] = total_laps
            dnf_matrix[4,i] = driven_laps
            dnf_matrix[5,i] = 1-driven_laps/total_laps
            
            matrix[5,i] = dnf_matrix[2,i]*dnf_matrix[5,i]

        self.dnf_matrix = dnf_matrix
        self.matrix = matrix
        print('DONE <- DNF model initialize')
        time.sleep(1)
        
        return matrix
    
    
    # %% 300 Solver starting grid
    def race_starting_grid(self,option):
        # option = 0 -> sim_grid
        # option = 1 -> solver_grid
        import numpy as np
        print('\n\n----------- SOLVER SIMULATION --------------')        
        
        def find(element, matrix):
            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    if matrix[i][j] == element:
                        return (i, j)
        
        self.option = option
        matrix = self.matrix
        mycolor = self.mycolor
        mylinestyle = self.mylinestyle
        
        
        srace = np.zeros((self.sim_total_laps+2,len(self.sim_grid)))
        srace[1,:] = np.arange(0,len(self.sim_grid)/2,0.5)
        
        if option == 0:
            srace[0,:] = self.input_driverid
            bqlap = 0
            srace2 = 0
        elif option == 1:
            bqlap = matrix[1,:]
            bqlap = sorted(bqlap)
            
            bqlap_sort = np.zeros([len(matrix[:,0]),len(matrix[0,:])])
            color_sort = []
            line_sort = []
            for i in range(len(bqlap)):
                [r,c] = find(bqlap[i],matrix)
                bqlap_sort[:,i] = matrix[:,c]
                color_sort.append(mycolor[c])
                line_sort.append(mylinestyle[c])
            
            srace[0,:] = bqlap_sort[0,:]
            srace2 = 0
            
            self.mylinestyle = line_sort
            self.mycolor = color_sort
            self.matrix = bqlap_sort
            
            
        self.srace = srace
        
        matrix = self.matrix
        print('DONE <- Cars on the grid and ready to race')
        
        return srace,matrix
    
    
    # %% 350 Solver race simulation
    def race_simulation(self):
        import numpy as np
        print('DONE <- GREEN GREEN GREEN!\n')
        
        def find(element, matrix):
            for a in range(len(matrix)):
                for b in range(len(matrix[a])):
                    if matrix[a][b] == element:
                        return (a, b)                        
        
        matrix = self.matrix
        srace = self.srace
        
        sltime = np.zeros((self.sim_total_laps+2,len(self.sim_grid)))
        sltime[1,:] = np.arange(0,len(self.sim_grid)/2,0.5)
        sltime[0,:] = matrix[0,:]
        
        sgrid = np.zeros((self.sim_total_laps+2,len(self.sim_grid)))
        sgrid[1,:] = np.arange(0,len(self.sim_grid),1)
        sgrid[0,:] = matrix[0,:]
        
        ff0 = 1/8 # Correción beta0 Fuel
        ff1 = 1/10 # Correción beta1 Fuel
        fv = 1/3 # Correción lt variability
        
        fp = 1/6 # Corrección DNF probability
        
        vsc_prob = 0.25 # Virtual Safety Car deployment probability
        sc_prob = 0.2 # Safety Car deployment probability
        
        ot_prob = 0.2 # Overtake success probability
        
        dnf = np.zeros((1,len(self.sim_grid))); dnf = dnf[0]
        crash = 0 # Zero crash at the start
        vsc_deploy = 0 # VSC & SC not deploy at the start
        vsc_ini = 0 # VSC & SC Laps on track 
        novertake = 0  # Number of total overtakes
        
        for i in range(self.sim_total_laps): 
            
            if crash == 1:
                if vsc_deploy == 0: # Crash in last lap
                    vsc_deploy = np.random.choice([0,1,2],p=[1-vsc_prob-sc_prob,vsc_prob,sc_prob])
                    if vsc_deploy == 1:
                        print('\nVSC ON - Lap ',i)
                    elif vsc_deploy == 2:
                        print('\nSC ON - Lap ',i)
                    vsc_ini = i
                
            crash = 0                         
                
            for j in range(len(self.sim_grid)):
                if vsc_deploy == 1 and i-vsc_ini>1: # VSC max 2 laps
                    vsc_deploy = 0
                    print('End VSC - Lap ',i-1,'\n')
                    
                if vsc_deploy == 2 and i-vsc_ini>2: # SC max 3 laps
                    ot_plus = 0
                    vsc_deploy = 0
                    print('End SC - Lap ',i-1,'\n')
                         
                # VIRTUAL SAFETY CAR
                if vsc_deploy == 1:
                    if dnf[j] == 0: # Still driving
                        lap_time = max(matrix[1,:])+1.5
                    elif dnf[j] == 1: # DNF
                        lap_time = 200
                    srace[i+2,j] = srace[i+1,j] + lap_time  
                    sltime[i+2,j] = lap_time
                
                # SAFETY CAR
                elif vsc_deploy == 2:
                    sc_gridtime = np.arange(0,len(self.sim_grid),0.5)
                    sc_gridtime = sgrid[i+1,:] # Posicion en la vuelta anterior
                    if dnf[j] == 0: # Still driving
                        lap_time = max(matrix[1,:])+5
                        sltime[i+2,j] = lap_time
                        if i-vsc_ini <= 0:
                            srace[i+2,j] = (sc_gridtime[j]-1)/2
                            sgrid[i+2,j] = sgrid[i+1,j]
                        elif i-vsc_ini > 0:
                            srace[i+2,j] = srace[i+1,j]
                            sgrid[i+2,j] = sgrid[i+1,j]
                        
                    elif dnf[j] == 1: # DNF
                        lap_time = 200
                        sltime[i+2,j] = lap_time
                        srace[i+2,j] = srace[i+1,j] + lap_time
                        sgrid[i+2,j] = sgrid[i+1,j]
                
                # RACE
                elif vsc_deploy == 0:
                    dnfprob = matrix[5,j]*fp
                    if i <= 3: # Aumento probabilidad DNF. 3 primeras vueltas
                        dnfprob = dnfprob + 0.05
                        
                    # dnfprob = matrix[5,j]*0
                    
                    dnf_c = np.random.choice([0,1],p=[1-dnfprob,dnfprob])
                    if dnf_c == 1 and dnf[j] == 0 and i>0: # DNF car and driver
                        dnf[j] = 1
                        if crash == 0:
                            crash = 1
                            print('CRASH. Driver ',srace[0,j],' Lap: ',i)                     
    
                    if dnf[j] == 0: # Still driving
                        # Con variables
                        pace_fuel = matrix[2,j]*ff0 + matrix[3,j]*i*ff1
                        pace_base = matrix[1,j]
                        pace_ltvar = abs(np.random.normal(0,matrix[4,j]*fv,1))
                        
                        lap_time = pace_base + pace_ltvar + pace_fuel    
                    elif dnf[j] == 1: # DNF
                        pace_base = matrix[1,j]
                        lap_time = 200
                        
                    srace[i+2,j] = srace[i+1,j] + lap_time  
                    sltime[i+2,j] = lap_time
                    
            
            ## OVERTAKE MODEL ##
            if i == 0:
                sumdnf2 = 0
            
            if i > 0:
                sumdnf1 = sumdnf2
                sumdnf2 = sum(dnf)
                
                
                grid_eval = sorted(srace[i+2,:])
                grid_before = sgrid[i+1,:]
                grid_after = np.arange(0,len(self.sim_grid),1)
                    
                for j in range(len(self.sim_grid)):
                    [r,c] = find(grid_eval[j],srace)
                    grid_after[c] = j+1
                    # print(grid_after)
                
                if sumdnf1 != sumdnf2: # DNF en esta vuelta                
                    # print('DNF')  
                    # print(sumdnf1-sumdnf2)
                    grid_eval = sorted(srace[i+2,:])
                    
                    for j in range(len(self.sim_grid)):
                        if vsc_deploy < 2: # No hay safety Car                
                            [r,c] = find(grid_eval[j],srace)
                            
                            sgrid[i+2,c] = j+1
                            
                else: 
                    # print('NO DNF')
                    
                    for j in range(len(self.sim_grid)):
                        if vsc_deploy < 2: # No hay safety car
                            [r,c] = find(grid_eval[j],srace)

                            # print(grid_before[c]-grid_after[c])
                            
                            if grid_before[c]-grid_after[c] == 0: # No hay cambio de posicion
                                sgrid[i+2,c] = j+1
                                
                            else:
  
                                if sltime[i+2,c] == 200:
                                    sgrid[i+2,c] = sgrid[i+1,c]
                                    
                                else:
                                    if grid_before[c]-grid_after[c] >= 1:
                                        # ot_prob = 0
                                        ot_success = np.random.choice([0,1],p=[1-ot_prob,ot_prob])
                                        if ot_success == 0: # No adelantamiento
                                            # print('NOK OVERTAKE')    
                                            sgrid[i+2,c] = sgrid[i+1,c]
                                        
                                            [r,ca] = find(grid_eval[j+1],srace)

                                            srace[i+2,c] = srace[i+2,ca]+1
                                            sltime[i+2,c] = sltime[i+2,ca]+1
                                            
                                            sgrid[i+2,ca] = sgrid[i+1,ca]
                                        
                                        elif ot_success == 1: # Exito adelantamiento
                                            # print('OVERTAKE')    
                                            sgrid[i+2,c] = j+1

                                    elif grid_before[c]-grid_after[c] <= -1:
                                        if ot_success == 0:
                                            sgrid[i+2,c] = sgrid[i+1,c] # ot_prob = 0
                                        elif ot_success == 1:
                                            sgrid[i+2,c] = j+1 # ot_prob = 1
                                         
            else:
                print('FORMATION LAP\n')
                sumdnf2 = 0
                grid_eval = sorted(srace[i+2,:])
                for j in range(len(self.sim_grid)):
                    if vsc_deploy < 2: # No hay safety Car                
                        [r,c] = find(grid_eval[j],srace)
                        sgrid[i+2,c] = j+1
            
            print('Lap: ',i)
            # print('Lap: ',i,' - Grid: ',sgrid[i+2,:])
            # print(dnf,'\n')
        
        self.srace = srace
        self.sltime = sltime
        self.sgrid = sgrid
        
        print('\nDONE <- Race simulation finished')
        
        return srace,sltime,sgrid
    
    
    # %% 500_Results analysis        
    def analysis_race_results(self):
        import numpy as np
        import math
        
        import matplotlib.pyplot as plt

        
        def take_second(elem):
            return elem[1]
        
        def mycolor_cycle(mycolor,mylinestyle):
            from cycler import cycler          
            cc = (cycler(color = mycolor)+
                  cycler(linestyle = mylinestyle))
            
            return cc
        
        srace = self.srace
        sltime = self.sltime
        sgrid = self.sgrid
        
        laps = np.arange(0,len(srace[:,0])-2,1)
        
        grid_result = sgrid[-1,:]
        i_ref = np.asarray(np.where(grid_result==1))
        i_ref = int(i_ref)
        
        v_ref = srace[2:,i_ref] # REFERENCIA
        ref = np.copy(srace)
        
        grid_finish = list(np.arange(0,len(self.sim_grid),1))
        
        
        for i in range(len(self.sim_grid)):
            ref[2:,i] = v_ref
            
        if self.option == 0: # ACABAR
            grid_legend = self.sim_grid
        else:
            grid_res = np.zeros([len(srace[0,:]),2])
            grid_res[:,0] = srace[0,:]
            grid_res[:,1] = sgrid[-1,:]
            grid_res = tuple(grid_res)
            grid_res = sorted(grid_res,key=take_second)
            
            pos = list(np.arange(1,len(self.sim_grid)+1,1))
            grid_finish = list(np.arange(0,len(self.sim_grid),1))
            grid_legend = list(np.arange(0,len(self.sim_grid),1))
            d_dr = self.data_drivers
            for i in range(len(self.sim_grid)):
                driver_id = grid_res[i]
                driver_id = driver_id[0]
                d = d_dr[d_dr.driverId == driver_id]
                name = list(d['driverRef'])
                grid_finish[i] = str(pos[i])+'. '+str(name[0])
                
                driver_id = srace[0,i]
                d = d_dr[d_dr.driverId == driver_id]
                name = list(d['driverRef'])
                grid_legend[i] = str(pos[i])+'. '+str(name[0])
        
        mycolor = self.mycolor
        mylinestyle = self.mylinestyle
        mycolor = mycolor_cycle(mycolor,mylinestyle)
        
        fig = plt.figure()
        
        ax = plt.subplot(311)
        ax.set_prop_cycle(mycolor)
        plt.plot(laps,sgrid[2:,:])
        plt.title('Summary Race Results: ' + self.sim_circuit[0].upper())
        plt.legend(grid_legend,fontsize='xx-small',bbox_to_anchor=(1.12, 1.5),
                   title='Start')
        plt.ylabel('Grid Position')
        plt.yticks(np.arange(1,len(grid_legend)+1,2))
        plt.xticks(np.arange(0,self.sim_total_laps+2,2))
        plt.grid(linestyle='--',linewidth=0.75)
        

        ax=plt.subplot(312)
        ax.set_prop_cycle(mycolor)
        plt.plot(laps,sltime[2:,:])
        plt.ylabel('Lap Time [s]')
        valmin = math.modf(sltime[2:,:].min())
        valmax = math.modf(sltime[2:,:].max()+1)
        plt.legend(grid_finish,fontsize='xx-small',bbox_to_anchor=(1.10, 0.5),
                   title='Finish',handletextpad=0, handlelength=0)
        plt.ylim(valmin[1],valmin[1]+5)
        plt.yticks(np.arange(valmin[1],valmin[1]+6,0.5))
        plt.xticks(np.arange(0,self.sim_total_laps+2,2))
        plt.grid(linestyle='--',linewidth=0.75)
        
        ax=plt.subplot(313)
        ax.set_prop_cycle(mycolor)
        plt.plot(laps,srace[2:,:]-ref[2:,:])
        plt.ylim(-5,80)
        plt.xlabel('Laps')
        plt.ylabel('Time [s]')
        plt.yticks(np.arange(-10,90,10))
        plt.xticks(np.arange(0,self.sim_total_laps+2,2))
        plt.grid(linestyle='--',linewidth=0.75)
        
        print('\n\n----------- RACE RESULTS --------------')
        
        print('\nStarting Grid\n')
        for i in range(len(self.sim_grid)):
            print(grid_legend[i])
            
        print('\n\nFinishing Grid\n')
        for i in range(len(self.sim_grid)):
            print(grid_finish[i])



        

from pyomo.environ import *
import numpy as np

class OptimizeBESS:
    
    # Parameterize the energy community and the optimizer
    #
    def __init__(self, 
                 P_el_predicted, 
                 nr_of_households, 
                 price, 
                 P_perfect,
                 delta_t = 1, # 1h
                 ):
        
        self.delta_t = delta_t
        self.P_el_predicted = P_el_predicted
        self.price = price
        self.P_perfect = P_perfect
        
        # Optimal batterie size per household accord to https://doi.org/10.1016/j.apenergy.2017.12.056
        wh_storage_per_household = 12000.0
        self.C = nr_of_households * wh_storage_per_household
        
        # Maximum charge Power in W
        self.P_el_max = self.C / 4
        self.eta_ch = 0.922
        self.eta_dis = 0.922

        # Time parameters
        self.time_steps_per_day = 24
        available_timesteps = len(P_el_predicted)
        self.test_days = available_timesteps // self.time_steps_per_day
    
        # Open-source solver HiGHS:
        self.solver = SolverFactory("appsi_highs")  
        
    # Solve the Optimization Problem
    #
    def run(self):
        
        # Initialize variables to store optimization results
        optimal_P_batt = np.zeros(self.test_days * self.time_steps_per_day)

        # Initial battery state (50% full)
        E_start = self.C / 2
        E_final = self.C / 2

        for day in range(self.test_days):
            start_hour = day * self.time_steps_per_day

            # Initialize Pyomo m
            m = ConcreteModel()
            m.T = range(self.time_steps_per_day)
            m.T_ext = range(self.time_steps_per_day + 1)

            # Define variables
            m.P_grid = Var(m.T, within=NonNegativeReals)
            m.P_ch = Var(m.T, bounds=(0, self.P_el_max))
            m.P_dis = Var(m.T, bounds=(0, self.P_el_max))
            m.E_batt = Var(m.T_ext, bounds=(0, self.C))
            
            m.b_ch = Var(m.T, within=Binary)
            m.b_dis = Var(m.T, within=Binary)

            # Objective function: minimize grid costs
            m.obj = Objective(
                expr=sum(self.price[start_hour + t] * m.P_grid[t] * self.delta_t for t in m.T),
                sense=minimize
            )

            # Constraints
            m.InitialSOC = Constraint(expr=m.E_batt[0] == E_start)
            m.FinalSOC = Constraint(expr=m.E_batt[self.time_steps_per_day] == E_final)

        #   m.PowerBase = ConstraintList()
            m.PowerBalance = ConstraintList()
            m.EnergyBalance = ConstraintList()
            m.ChargingIndicator = ConstraintList()
            m.DishargingIndicator = ConstraintList()
            m.SOS1 = ConstraintList()

            for t in m.T:
                # Power balance: power in = power out
            #   m.PowerBase.add(m.P_grid[t] >= 0.1*P_el_predicted[pred_type, start_hour + t])

                m.PowerBalance.add(m.P_grid[t] + m.P_dis[t] == m.P_ch[t] + self.P_el_predicted[start_hour + t])
                # Energy balance of the battery
                m.EnergyBalance.add(m.E_batt[t + 1] == m.E_batt[t] + (m.P_ch[t]*self.eta_ch - m.P_dis[t]/self.eta_dis) * self.delta_t)

                # Binary variables to ensure that the battery cannot charge and discharge at the same time
                m.ChargingIndicator.add(m.P_ch[t] <= m.b_ch[t] * self.P_el_max)
                m.DishargingIndicator.add(m.P_dis[t] <= m.b_dis[t] * self.P_el_max)
                m.SOS1.add(m.b_ch[t] + m.b_dis[t] == 1)

            # Solve the m
            result = self.solver.solve(m)

            # Store Solutions
            for t in m.T:
                optimal_P_batt[start_hour + t] = value(m.P_ch[t]) - value(m.P_dis[t])
            E_start = value(m.E_batt[self.time_steps_per_day])
        
        # Final cost calculation without profit for feed-in (feed-in tariff = 0 €/Wh)
        # Calculate the resulting grid power for consumption
        P_total = np.maximum(optimal_P_batt + self.P_perfect, 0)
        costs_year = np.sum(P_total * self.price * self.delta_t)

        # Store the annual cost for the current predictor
        return costs_year

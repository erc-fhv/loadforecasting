import numpy as np
from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Binary, Objective, Constraint, \
    ConstraintList, SolverFactory, minimize, value

class OptimizeBess:
    """Optimize the operation of a Battery Energy Storage System (BESS)"""

    # Parameterize the energy community and the optimizer
    #
    def __init__(self,
                 p_el_predicted,
                 nr_of_households,
                 price,
                 p_perfect,
                 delta_t = 1, # 1h
                 buffer_size = 0.1,
                 battery_size = 0
                 ):

        self.delta_t = delta_t
        self.price = price
        self.p_perfect = p_perfect
        self.buffer_size = buffer_size
        self.capacity = battery_size

        if battery_size == 0:
            # Linear scale of the optimal batterie size per household 
            # accord to https://doi.org/10.1016/j.apenergy.2017.12.056
            #
            wh_storage_per_household = 12000.0
            self.capacity = nr_of_households * wh_storage_per_household

        # Maximum charge Power in W
        self.p_el_max = self.capacity / 4
        self.eta_ch = 0.922
        self.eta_dis = 0.922

        # Time parameters
        self.time_steps_per_day = 24
        available_timesteps = len(p_el_predicted)
        self.test_days = available_timesteps // self.time_steps_per_day

        # Ensure that the load prediction is only positive.
        # (This case study does not include any electricity generation.)
        p_el_predicted = np.clip(p_el_predicted, a_min=0, a_max=None)
        self.p_el_predicted = p_el_predicted    

        # Open-source solver HiGHS:
        self.solver = SolverFactory("appsi_highs")  

    def run(self):
        """Solve the Optimization Problem"""

        # Initialize variables to store optimization results
        optimal_p_batt = np.zeros(self.test_days * self.time_steps_per_day)

        # Initial battery state (50% full)
        energy_start = self.capacity / 2
        energy_final = self.capacity / 2

        for day in range(self.test_days):
            start_hour = day * self.time_steps_per_day

            # Initialize Pyomo m
            m = ConcreteModel()
            m.T = range(self.time_steps_per_day)
            m.T_ext = range(self.time_steps_per_day + 1)

            # Define variables
            m.P_grid = Var(m.T, within=NonNegativeReals)
            m.P_ch = Var(m.T, bounds=(0, self.p_el_max))
            m.P_dis = Var(m.T, bounds=(0, self.p_el_max))
            m.E_batt = Var(m.T_ext, bounds=(0, self.capacity))

            m.b_ch = Var(m.T, within=Binary)
            m.b_dis = Var(m.T, within=Binary)

            # Objective function: minimize grid costs
            m.obj = Objective(
                expr=sum(self.price[start_hour + t] * m.P_grid[t] * self.delta_t for t in m.T),
                sense=minimize
            )

            # Constraints
            m.InitialSOC = Constraint(expr=m.E_batt[0] == energy_start)
            m.FinalSOC = Constraint(expr=m.E_batt[self.time_steps_per_day] == energy_final)

            m.PowerBase = ConstraintList()
            m.PowerBalance = ConstraintList()
            m.EnergyBalance = ConstraintList()
            m.ChargingIndicator = ConstraintList()
            m.DishargingIndicator = ConstraintList()
            m.SOS1 = ConstraintList()

            for t in m.T:
                
                # Power balance: power in = power out
                m.PowerBase.add(m.P_grid[t] >= self.buffer_size*self.p_el_predicted[start_hour + t])

                m.PowerBalance.add(m.P_grid[t] + m.P_dis[t] == m.P_ch[t] 
                    + self.p_el_predicted[start_hour + t])
                
                # Energy balance of the battery
                m.EnergyBalance.add(m.E_batt[t + 1] == m.E_batt[t] 
                    + (m.P_ch[t]*self.eta_ch - m.P_dis[t]/self.eta_dis) * self.delta_t)

                # Binary variables to ensure that the battery cannot charge and discharge at the 
                # same time
                m.ChargingIndicator.add(m.P_ch[t] <= m.b_ch[t] * self.p_el_max)
                m.DishargingIndicator.add(m.P_dis[t] <= m.b_dis[t] * self.p_el_max)
                m.SOS1.add(m.b_ch[t] + m.b_dis[t] == 1)

            # Solve the m
            _ = self.solver.solve(m)

            # Store Solutions
            for t in m.T:
                optimal_p_batt[start_hour + t] = value(m.P_ch[t]) - value(m.P_dis[t])
            energy_start = value(m.E_batt[self.time_steps_per_day])

        # Final cost calculation without profit for feed-in (feed-in tariff = 0 €/Wh)
        # Calculate the resulting grid power for consumption
        p_total = np.maximum(optimal_p_batt + self.p_perfect, 0)
        costs_year = np.sum(p_total * self.price * self.delta_t)

        # Store the annual cost for the current predictor
        return costs_year

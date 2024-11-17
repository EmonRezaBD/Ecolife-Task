import numpy as np
import random
import utils

class ACO:
    def __init__(self, parameters, server_pair, function_name, ci_avg, cur_ci, cur_interval):
        # Problem-specific parameters
        self.size = parameters[0]  # number of ants
        self.kat_options = parameters[1]  # possible kat values
        self.lam = parameters[2]
        self.server_pair = server_pair
        self.function_name = function_name
        self.cur_ci = cur_ci
        self.cur_interval = cur_interval

        # compute max:
        old_cold,_ = utils.get_st(function_name, server_pair[0])
        new_cold,_ = utils.get_st(function_name, server_pair[1])
        cold_carbon_max,_= utils.compute_exe(function_name, server_pair,ci_avg)
        self.max_st = max(old_cold, new_cold)
        self.max_carbon_st = max(cold_carbon_max)
        self.max_carbon_kat = max(utils.compute_kat(function_name, server_pair[0],7, ci_avg),utils.compute_kat(function_name, server_pair[1],7, ci_avg))

        # Initialize pheromone matrix
        self.pheromone_ka_loc = [1.0, 1.0]  # pheromones for ka_loc choices (0, 1)
        self.pheromone_kat = [1.0] * len(self.kat_options)  # pheromones for kat values

        # Other constants
        self.alpha = 1.0  # pheromone importance
        self.beta = 2.0   # heuristic importance
        self.evaporation_rate = 0.1
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def prob_cold(self, cur_interval, kat):
        if len(cur_interval) == 0:
            # No invocation
            return 0.5, 0.5
        else:
            cold = 0
            warm = 0
            for interval in cur_interval:
                if interval <= kat:
                    # Hit (warm start)
                    warm += 1
                else:
                    cold += 1
            return cold / (cold + warm), warm / (cold + warm)

    def fitness(self, ka_loc, kat, ci, past_interval):
        # kat carbon calculation
        score = 0
        old_kat_carbon = utils.compute_kat(self.function_name, self.server_pair[0], kat, ci)
        new_kat_carbon = utils.compute_kat(self.function_name, self.server_pair[1], kat, ci)
        cold_carbon, warm_carbon = utils.compute_exe(self.function_name, self.server_pair, ci)

        # Service time for old and new servers
        old_st = utils.get_st(self.function_name, self.server_pair[0])
        new_st = utils.get_st(self.function_name, self.server_pair[1])

        # Carbon score component
        score += (1 - self.lam) * (((1 - ka_loc) * old_kat_carbon + ka_loc * new_kat_carbon) / self.max_carbon_kat)

        # Cold and warm start probabilities
        cold_prob, warm_prob = self.prob_cold(past_interval, kat)

        # Service time and carbon impact with probabilities
        part_time_prob = cold_prob * ((1 - ka_loc) * old_st[0] + ka_loc * new_st[0]) + warm_prob * ((1 - ka_loc) * old_st[1] + ka_loc * new_st[1])
        part_carbon_prob = cold_prob * ((1 - ka_loc) * cold_carbon[0] + ka_loc * cold_carbon[1]) + warm_prob * ((1 - ka_loc) * warm_carbon[0] + ka_loc * warm_carbon[1])

        # Add time and carbon components to the score
        score += self.lam * (part_time_prob / self.max_st)
        score += (1 - self.lam) * (part_carbon_prob / self.max_carbon_st)

        return score


    def construct_solution(self, carbon_intensity, invoke_interval):
        # Each ant constructs a solution
        # Probabilistic selection based on pheromone and heuristic information
        ka_loc = np.random.choice([0, 1], p=self._calc_prob(self.pheromone_ka_loc))
        kat = np.random.choice(self.kat_options, p=self._calc_prob(self.pheromone_kat))

        # Evaluate solution
        #fitness = self.fitness(ka_loc, kat)
        fitness = self.fitness(ka_loc, kat, carbon_intensity, invoke_interval)

        return ka_loc, kat, fitness

    def _calc_prob(self, pheromone_levels):
        # Convert pheromone levels into selection probabilities
        total = sum(pheromone_levels)
        return [pheromone / total for pheromone in pheromone_levels]

    def update_pheromones(self, solutions):
        # Evaporate some pheromone on all paths
        self.pheromone_ka_loc = [p * (1 - self.evaporation_rate) for p in self.pheromone_ka_loc]
        self.pheromone_kat = [p * (1 - self.evaporation_rate) for p in self.pheromone_kat]

        # Deposit pheromone based on solution fitness
        for ka_loc, kat, fitness in solutions:
            # Higher fitness (lower cost) => more pheromone
            deposit = 1.0 / (fitness + 1e-10)
            self.pheromone_ka_loc[ka_loc] += deposit
            self.pheromone_kat[kat] += deposit

    #def main(self, iterations=100):
    def main(self, carbon_intensity, invoke_interval, iterations=100):
    # Main loop for ACO
        for _ in range(iterations):
            solutions = []
            for _ in range(self.size):
                # Adjust `construct_solution` if it requires carbon_intensity or invoke_interval
                ka_loc, kat, fitness = self.construct_solution(carbon_intensity, invoke_interval)
                solutions.append((ka_loc, kat, fitness))
                
                # Track best solution
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = (ka_loc, kat)

            # Update pheromone trails based on solutions
            self.update_pheromones(solutions)

        return self.best_solution, self.best_fitness

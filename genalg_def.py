"""
Genetic Algorithm Module.

Classes:
    
    Individual
        General purpose genetic individual.
    IndividualSymbolicRegression
        Subclass of Individual, which is used for symbolic regression.
    IndividualTapePlacement
        Subclass of Individual, which is used for tape placement process window exploration.
    Evolver
        General purpose genentic algorithm optimizer.
    
Functions:
    
    my_method
        Description here.
    
Misc variables:
    
    None
        Description here.
    
Author: Eimantas Survila
Date: Sat Dec 19 09:37:16 2020
"""

import random
import numpy as np
import datetime
from timeit import default_timer as timer

# %%


class Individual():

    mutation_config = {
        "uniform_prob": 1}

    crossover_config = {
        "uniform_prob": 1}

    # %%
    def __init__(
            self,
            func_to_opt=None,
            func_in=None,
            func_fit=None,
            vartypes=None,
            varbounds=None
    ):

        self.vartypes = vartypes
        self.varbounds = varbounds

        # Passing the functions.
        self.func_to_opt = func_to_opt
        self.func_in = func_in
        self.func_fit = func_fit

    # %%
    def fitness(self):
        return \
            self.func_fit(
                self.func_to_opt(
                    *self.func_in(self.gencode)))

    # %%
    def generate_random_gencode(self):
        gencode = {}
        for key, limits in self.varbounds.items():
            if self.vartypes[key] == list:

                new_i_pos = random.randint(0, len(self.varbounds[key])-1)
                new_val = self.varbounds[key][new_i_pos]
                gencode.update({key: new_val})

            if self.vartypes[key] == float:

                if len(limits) == 2:
                    gencode.update(
                        {key: limits[0] + (limits[-1] - limits[0]) * random.random()})
                if len(limits) == 3:
                    new_val = limits[0] + (((limits[-1] - limits[0])
                                           * random.random()) // limits[1]) * limits[1]
                    gencode.update({key: new_val})

            if self.vartypes[key] == int:
                gencode.update(
                    {key: limits[0] + int((limits[-1] - limits[0]) * random.random())})

        self.gencode = gencode

    # %%
    def mutate(self):
        new_gencode = {}
        MUTATE_CONSTANT = 0.1
        LIST_CHOSSE_ANOTHER = 0.3

        for key, val in self.gencode.items():

            if self.vartypes[key] == list:

                if random.random() < LIST_CHOSSE_ANOTHER:

                    new_i_pos = random.randint(0, len(self.varbounds[key])-1)
                    new_val = self.varbounds[key][new_i_pos]
                    while new_val == self.gencode[key]:
                        new_i_pos = random.randint(
                            0, len(self.varbounds[key])-1)
                        new_val = self.varbounds[key][new_i_pos]

            if self.vartypes[key] == int:

                MUTATE_CONSTANT = 0.5

                new_val = \
                    self.gencode[key] + int(
                        (random.random() - 0.5) *
                        (self.varbounds[key][-1] - self.varbounds[key][0]) * MUTATE_CONSTANT)

                if new_val < self.varbounds[key][0]:
                    new_val = int(self.varbounds[key][0])
                if new_val > self.varbounds[key][-1]:
                    new_val = int(self.varbounds[key][-1])

                new_gencode.update({key: new_val})

            if self.vartypes[key] == float:

                # Only uniform mutation.
                if len(self.varbounds[key]) == 2:
                    new_val = \
                        self.gencode[key] + \
                        (random.random() - 0.5) * \
                        (self.varbounds[key][-1] -
                         self.varbounds[key][0]) * MUTATE_CONSTANT

                if len(self.varbounds[key]) == 3:
                    new_val = \
                        self.gencode[key] + ((
                            (random.random() - 0.5) *
                            (self.varbounds[key][-1] - self.varbounds[key][0]) * MUTATE_CONSTANT)
                            // self.varbounds[key][1]) * self.varbounds[key][1]

                if new_val < self.varbounds[key][0]:
                    new_val = self.varbounds[key][0]

                if new_val > self.varbounds[key][-1]:
                    new_val = self.varbounds[key][-1]

                new_gencode.update({key: new_val})

            else:
                new_gencode.update({key: self.gencode[key]})

        mutant = Individual(
            self.func_to_opt,
            self.func_in,
            self.func_fit,
            self.vartypes,
            self.varbounds)

        mutant.gencode = new_gencode
        return mutant

    # %%
    def crossover(self, other):
        # Uniform crossover
        new_gencode = {}
        for key, val in self.gencode.items():

            if random.random() > 0.5:
                new_val = self.gencode[key]
            else:
                new_val = other.gencode[key]

            new_gencode.update({key: new_val})

        cross_over_idv = Individual(
            self.func_to_opt,
            self.func_in,
            self.func_fit,
            self.vartypes,
            self.varbounds)

        cross_over_idv.gencode = new_gencode
        return cross_over_idv

    def elitechild(self):

        cross_over_idv = Individual(
            self.func_to_opt,
            self.func_in,
            self.func_fit,
            self.vartypes,
            self.varbounds)

        cross_over_idv.gencode = self.gencode
        return cross_over_idv

    # %%
    def func_out(self):
        return \
            self.func_to_opt(
                *self.func_in(self.gencode))

    # %%
    def signature(self):
        return self.gencode

# %%


class Evolver():

    # %%
    def __init__(
            self,
            var_type=None,
            var_bounds=None,
            f=None,
            f_in=None,
            f_fit=None,
            individual_class=Individual,
            metaoptimize=False):

        self.configuration = {
            "population_size": 100,
            "maximal_iterations": 20,
            "elitismus_proportion": 0.2,
            "crossover_proportion": 0.4,
            "new_blood_proporiton": 0,
            "probability_of_mutation": 0.2,
            "log_results": False}

        self.individual_class = individual_class
        self.f_to_opt = f
        self.f_in = f_in
        self.f_fit = f_fit
        self.individuals = []
        self.vartypes = var_type
        self.varbounds = var_bounds

    # %%
    def generate_first_gen(self):

        for _ in range(self.configuration["population_size"]):
            new_idv = self.individual_class(
                self.f_to_opt,
                self.f_in,
                self.f_fit,
                self.vartypes,
                self.varbounds)
            new_idv.generate_random_gencode()

            self.individuals.append(new_idv)

    # %%
    def sort_the_generation(self):

        self.sorted_gen = [[x, x.fitness()] for x in self.individuals.copy()]
        self.sorted_gen.sort(key=lambda x: x[1])

    # %%
    def generate_next_gen(self):

        prob_of_select = self.sorted_gen.copy()
        overall_fitness = 0

        max_val = 0
        for _, i in prob_of_select:
            if i > max_val:
                max_val = i
        max_val = 1.01 * max_val

        for i, _ in enumerate(prob_of_select):
            prob_of_select[i].append(max_val - prob_of_select[i][1])

        for _, _, fit in prob_of_select:
            overall_fitness += fit

        for i, _ in enumerate(prob_of_select):
            prob_of_select[i][2] = prob_of_select[i][2] / overall_fitness

        def select_random_parent():
            """
            Return a parent individual from list with probabilities
            of selection
            """
            nonlocal prob_of_select
            random_no = random.random()

            prob_sum = 0
            for no, [pos_parent, _, prob] in enumerate(prob_of_select):

                if prob_sum + prob > random_no:
                    parent = pos_parent
                    break
                else:
                    prob_sum += prob

            return parent

        # Elitismus
        next_gen = []
        for i in range(int(self.configuration["elitismus_proportion"] * self.configuration["population_size"])):
            next_gen.append(prob_of_select[i][0].elitechild())

        # Crossover
        no_of_crossover = int(
            self.configuration["population_size"] *
            self.configuration["crossover_proportion"])

        MAX_ITER = 10
        for _ in range(no_of_crossover):

            new_child_special = False
            i = 0

            while (not new_child_special) and (i <= MAX_ITER):

                parent_first = select_random_parent()
                parent_second = select_random_parent()
                child = parent_first.crossover(parent_second)

                del parent_first, parent_second
                if random.random() < self.configuration["probability_of_mutation"]:
                    child.mutate()

                if not (child.signature() in [i.signature() for i in next_gen]):
                    new_child_special = True
                    next_gen.append(child)
                    break

                i += 1

        # Mutation
        no_of_mutations = int(
            self.configuration["population_size"] * (1 -
                                                     self.configuration["elitismus_proportion"] -
                                                     self.configuration["crossover_proportion"] -
                                                     self.configuration["new_blood_proporiton"]))

        assert (no_of_mutations > 0), "Negative number of mutations!"

        MAX_ITER = 10
        for _ in range(no_of_mutations):

            new_child_special = False
            i = 0
            while (not new_child_special) and (i <= MAX_ITER):

                idv_to_mutate_from = select_random_parent()
                mutant = idv_to_mutate_from.mutate()

                if not (mutant.signature() in [i.signature() for i in next_gen]):
                    new_child_special = True
                    next_gen.append(mutant)
                    break

                i += 1

        self.individuals = next_gen.copy()

    # %%
    def log_preambel(self):

        x = datetime.datetime.now()
        x = str(x)
        x = x[0:19]
        x = x.replace(" ", "__")
        x = x.replace("-", "_")
        x = x.replace(":", "_")

        file_name = x + "_" + self.f_to_opt.__name__
        seperator = "=====================================================================================================\n"

        with open("logging/{}.txt".format(file_name), "a") as of:

            of.write("{}\n".format(str(datetime.datetime.now())))
            of.write("{}\n".format(self.f_to_opt.__name__))
            of.write(seperator)
            of.write("CONFIGURATION\n")
            of.write(seperator)
            for key, val in self.configuration.items():
                of.write("config: {},\t value: {}\n".format(key, val))

            of.write(seperator)
            of.write("VARIABLE BOUNDARIES\n")
            of.write(seperator)
            for key, val in self.varbounds.items():
                of.write("variable: {},\t boundaries: {}\n".format(key, val))

            of.write(seperator)

        return file_name

    # %%
    def log_time(self, file_name, time_start):

        time_end = timer()
        s = time_end - time_start
        h = s // 3600
        m = (s - h*3600) // 60
        s = np.round(s - h*3600 - m*60, 3)

        with open("logging/{}.txt".format(file_name), "a") as of:
            of.write("\nElapsed time: \t{} h. {} min. {} sec.\n".format(
                int(h), int(m), s))

    # %%
    def log_generation(self, file_name, gen_no, top):

        with open("logging/{}.txt".format(file_name), "a") as of:

            of.write("\nGeneration {}, top {} solutions:\n".format(gen_no, top))
            of.write("\tFit\t\t")
            for key, _ in self.varbounds.items():
                of.write("\t" + key)
            of.write("\toutput_values\n")

            for i in range(top):

                of.write("\t{}".format(np.round(self.sorted_gen[i][1], 6)))
                for _, val in self.sorted_gen[i][0].gencode.items():
                    of.write("\t{}".format(val))

                of.write("\t{}\n".format(self.sorted_gen[i][0].func_out()))

    # %%
    def run(self, verbose=False):

        WRITE_INTER_GEN = True

        if self.configuration["log_results"]:
            file_name = self.log_preambel()
            time_start = timer()

        self.generate_first_gen()

        for gen_no in range(self.configuration["maximal_iterations"]):
            self.sort_the_generation()

            if WRITE_INTER_GEN and self.configuration["log_results"]:
                self.log_time(file_name, time_start)
                self.log_generation(file_name, gen_no, top=3)
            if verbose:
                print("Gen. {} calculated.".format(gen_no + 1))

            self.generate_next_gen()

        self.sort_the_generation()

        if self.configuration["log_results"]:
            self.log_time(file_name, time_start)
            self.log_generation(file_name, gen_no, top=20)

    # %%
    def bestvars(self):
        return self.f_in(self.sorted_gen[0][0].gencode)

    # %%
    def bestoutput(self):
        return self.sorted_gen[0][0].func_out()

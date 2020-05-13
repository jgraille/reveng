from deap import gp, tools, base, creator, algorithms
import operator
import math
import random
import itertools
import numpy
from utils.utils import protectedDiv, update_indexes


class GeneticProgramming:

    def __init__(self, x_train, y_train, excludes, header, config_file):
        """
        https://deap.readthedocs.io/en/master/examples/gp_spambase.html
        inputing features indexes are given after removing the target feature
        """
        self.feature_names = update_indexes(original_list=config_file['inputing_features_gp'],
                                            excludes=excludes,
                                            header=header)
        self.x_train = x_train[:, self.feature_names]
        self.y_train = y_train
        self.position_target = numpy.size(self.x_train, 1)
        self.pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, self.position_target), float, "IN")
        self.primitive_functions()
        # Minimizing the error between the response of the genetic formula and the true response. MSE
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        self.data = numpy.insert(self.x_train, numpy.size(self.x_train, 1), self.y_train, axis=1).tolist()
        self.toolbox = base.Toolbox()
        self.register_functions()

    def primitive_functions(self):
        try:
            self.pset.addPrimitive(operator.add, [float, float], float)
            self.pset.addPrimitive(operator.sub, [float, float], float)
            self.pset.addPrimitive(operator.mul, [float, float], float)
            self.pset.addPrimitive(protectedDiv, [float, float], float)
            self.pset.addPrimitive(operator.neg, [float], float)
            #self.pset.addPrimitive(math.exp, [float], float)
            #self.pset.addPrimitive(math.pow, [float, float], float)
            self.pset.addEphemeralConstant("randuniform", lambda: random.uniform(-1, 1), float)
            self.pset.addTerminal(False, bool)
            self.pset.addTerminal(True, bool)
        except Exception as e:
            print('\nMethod GeneticProgramming.primitive_funtions did not work', e.__repr__())

    def evalFunction(self, individual):
        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual)
        data_sample = random.sample(self.data, 500)
        sqerrors = ((func(*l[:self.position_target]) - l[self.position_target]) ** 2 for l in data_sample)
        result = math.fsum(sqerrors) / len(data_sample)
        return result,

    def register_functions(self):
        """
        gp.genHalfAndHAlf returns an expression with gp.genGrow() and gp.genFull (50%/50%)
        gp.genGrow returns expression where each leaf with different dept btw min and max
        gp.genFull returns expression where each leaf has the same depth between min and max
        """
        try:
            self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
            self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            self.toolbox.register("compile", gp.compile, pset=self.pset)

            self.toolbox.register("evaluate", self.evalFunction)
            """
            tools.selTournament get here 3 individuals are selected randomly from the pop.
            https://github.com/DEAP/deap/issues/214
            """
            self.toolbox.register("select", tools.selTournament, tournsize=3)
            self.toolbox.register("mate", gp.cxOnePoint)
            self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
            self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        except Exception as e:
            print('\nMethod GeneticProgramming.register_functions did not work: ', e.__repr__())

    def calculate(self):
        try:
            random.seed(10)
            pop = self.toolbox.population(n=500)
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", numpy.mean)
            stats.register("std", numpy.std)
            stats.register("min", numpy.min)
            stats.register("max", numpy.max)

            algorithms.eaSimple(pop, self.toolbox, 0.5, 0.2, 50, stats, halloffame=hof)

            return pop, stats, hof
        except Exception as e:
            print('\nMethod GeneitcProgramming.calculate did not work: ', e.__repr__())





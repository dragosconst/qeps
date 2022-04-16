from typing import List, Tuple, Dict
import random

import numpy as np

from qiea import qiea_membrane, QbitIndividual

class Region:
    def __init__(self, clauses_no: int, var_no: int, variable_clauses_mapping: Dict[int, int]):
        self.var_no = var_no
        self.clauses_no = clauses_no
        self.mapping = variable_clauses_mapping
        self.population = []
        self.best = None
        self.best_fit = 10 ** 4

    def qiea(self):
        g = np.random.randint(low=1, high=11)
        for i in range(g):
            self.population, best, fit = qiea_membrane(self.var_no, self.clauses_no, 10 ** 6, self.population,
                                                                 self.best, self.mapping)
            if fit == 0:
                self.best = best
                self.best_fit = fit
                return


    def set_population(self, qpop: List[QbitIndividual]):
        self.population = qpop

    def set_best(self, best: QbitIndividual):
        self.best = best


class Membrane:
    def __init__(self, var_no: int, clauses_no: int, regions_no: int, populations_no: int,
                 variable_clauses_mapping: Dict[int, int]):
        self.var_no = var_no
        self.clauses_no = clauses_no
        self.regions_no = regions_no
        self.regions = [Region(clauses_no, var_no, variable_clauses_mapping) for r in range(self.regions_no)]
        self.population = np.asarray([QbitIndividual(var_no) for i in range(populations_no)])

    def alloc(self):
        indexes = np.asarray([i for i in range(len(self.population))])
        random.shuffle(indexes)
        for r in range(self.regions_no):
            regions_left = len(self.regions) - r - 1
            idx = np.random.randint(low=1, high=len(indexes) - regions_left + 2)
            self.regions[r].set_population(self.population[indexes[:idx]])
            indexes = indexes[idx:]

    def qiea_and_propagate(self):
        best = None
        best_fit = 10 ** 4
        for r in self.regions:
            r.qiea()
            if r.best is not None and r.best_fit < best_fit:
                best = r.best
                best_fit = r.best_fit
                if best_fit == 0:
                    return True

        # propagate best to all regions
        for r in self.regions:
            r.set_best(best)
        return False


variable_clauses_mapping = dict()
def get_expression(input_str: str) -> Tuple[int, int]:
    global variable_clauses_mapping

    """

    :param input_str:
    :return: tuple of clauses number and unique variable number
    """
    clauses = input_str.split("^")
    biggest_var = None

    for idx, clause in enumerate(clauses):
        clause = clause.split("v")
        for variable in clause:
            variable = int(variable)
            if biggest_var is None or biggest_var < abs(variable):
                biggest_var = abs(variable)
            if variable in variable_clauses_mapping:
                variable_clauses_mapping[variable].add(idx)
            else:
                variable_clauses_mapping[variable] = set([idx])
    return len(clauses), biggest_var

if __name__ == "__main__":
    with open("input", "r") as f:
        lines = f.readlines()
        clauses_no, var_no = get_expression(lines[0][:-1])
        membrane = Membrane(var_no, clauses_no, 3, 4, variable_clauses_mapping)
        for t in range(10 ** 6):
            membrane.alloc()
            if membrane.qiea_and_propagate():
                break

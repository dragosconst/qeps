from typing import List, Tuple, Union, Dict
import copy

import numpy as np

class Qbit:
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def __iter__(self):
        return iter((self.alpha, self.beta))

    def get_quadrant(self):
        if self.alpha * self.beta >= 0:
            return 1
        return 2


class QbitIndividual:
    def __init__(self, l):
        self.l = l
        self.qbits = [Qbit(1/np.sqrt(2), 1/np.sqrt(2)) for i in range(l)]

    def __getitem__(self, idx):
        return self.qbits[idx]

    def __setitem__(self, idx, value):
        self.qbits[idx] = value

    def get(self):
        p = []
        for qbit in self.qbits:
            ai = qbit.alpha ** 2
            bi = qbit.beta ** 2
            pi = np.random.choice([0,1], p=(ai, bi))
            p.append(pi)
        return p

best_individuals = []
best = None
best_fit = None
variable_clauses_mapping = dict()
POP_SIZE = 3
MIG_PERIOD = 200

def fitness(x: List[int], clauses_no: int, from_membrane: bool=False,
            membrane_mapping: Dict[int, int]=None) -> int:
    global variable_clauses_mapping

    mapping = variable_clauses_mapping
    if from_membrane is True:
        mapping = membrane_mapping
    clauses_true = set()
    for idx, xi in enumerate(x):
        idx = idx + 1
        if xi and idx in mapping:
            for clause in mapping[idx]:
                clauses_true.add(clause)
        elif not xi:
            if -idx not in mapping:
                continue
            for clause in mapping[-idx]:
                clauses_true.add(clause)
    return clauses_no - len(clauses_true)


def q_gate(q: QbitIndividual, x: List[int], b: List[int], best_fit: int, clauses_no: int) -> QbitIndividual:
    x_fit = fitness(x, clauses_no)

    new_q = QbitIndividual(q.l)
    for idx, qi in enumerate(q):
        new_q[idx] = qi
    if not (x_fit <= best_fit):
        for idx, (xi, bi) in enumerate(zip(x, b)):
            cos_theta = 1
            sin_theta = 0
            if q[idx].get_quadrant() == 1:
                sign = 1
            else:
                sign = -1
            if xi == 0 and bi == 1:
                cos_theta = np.cos(sign * np.pi * 0.01)
                sin_theta = np.sin(sign * np.pi * 0.01)
            elif xi == 1 and bi == 0:
                cos_theta = np.cos(sign * np.pi * -0.01)
                sin_theta = np.sin(sign * np.pi * -0.01)
            alpha, beta = q[idx]
            new_q[idx].alpha = alpha * cos_theta - beta * sin_theta
            new_q[idx].beta = alpha * sin_theta + beta * cos_theta
    return new_q

def collapse_qbit_individuals(qpop: List[QbitIndividual], clauses_no: int) -> List[List[int]]:
    global best, best_fit

    pop = []
    # collapse qbits
    for qind in qpop:
        pind = qind.get()
        pind_fit = fitness(pind, clauses_no)
        if best_fit is None or best_fit > pind_fit:
            best_fit = pind_fit
            best = pind
        pop.append(pind)
    return pop

def migrate(var_no: int, clauses_no: int):
    global best, best_fit, best_individuals

    coin_flip = np.random.rand(1)
    # global migration
    if coin_flip >= 0.5:
        for idi in range(len(best_individuals)):
            best_individuals[idi] = best
    # local migration
    else:
        best_local = None
        best_local_fit = None
        # find best individual in population
        for ind in best_individuals:
            if best_local_fit is None or fitness(ind, clauses_no) > best_local_fit:
                best_local = ind
                best_local_fit = fitness(ind, clauses_no)
        # replace some variables with the best one
        replace_num = np.random.randint(low=1, high=POP_SIZE + 1)
        replace_idx = np.random.choice([i for i in range(POP_SIZE)], size=replace_num, replace=False)
        for idr in replace_idx:
            best_individuals[idr] = best_local

def qiea(var_no: int, clauses_no: int, time_steps: int):
    global best, best_fit, best_individuals, POP_SIZE

    qpop = [QbitIndividual(var_no) for i in range(POP_SIZE)]
    pop = collapse_qbit_individuals(qpop, clauses_no)
    best_individuals = copy.deepcopy(pop)
    print(f"best pop is {best} and score is {best_fit}")

    for t in range(1, time_steps):
        # get population
        pop = collapse_qbit_individuals(qpop, clauses_no)
        # update qbit individuals
        for idx, (qind, pind) in enumerate(zip(qpop, pop)):
            new_q = q_gate(qind, pind, best, best_fit, clauses_no)
            qpop[idx] = new_q
        # update best individuals
        for idx, (pind, bind) in enumerate(zip(pop, best_individuals)):
            pind_fitness = fitness(pind, clauses_no)
            bind_fitness = fitness(bind, clauses_no)

            if pind_fitness < bind_fitness:
                best_individuals[idx] = pind
        # do migrations
        if idx % MIG_PERIOD == 0:
            migrate(var_no, clauses_no)

        # check terminating condition
        if t % 1000 == 0:
            print(f"best pop is {best} and score is {best_fit}")
        if best_fit == 0:
            print(t)
            print(f"best pop is {best} and score is {best_fit}")
            print("am rezolvat")
            break

def collapse_qbit_individuals_membrane(qpop: List[QbitIndividual], clauses_no: int, best: QbitIndividual,
                                       best_fit: int, mapping: Dict[int, int]) \
                                       -> Tuple[List[List[int]], QbitIndividual, int]:
    pop = []
    # collapse qbits
    for qind in qpop:
        pind = qind.get()
        pind_fit = fitness(pind, clauses_no, True, mapping)
        if best_fit is None or best_fit > pind_fit:
            best_fit = pind_fit
            best = pind
        pop.append(pind)
    return pop, best, best_fit

def migrate_membrane(var_no: int, clauses_no: int, pop_size: int, best: QbitIndividual, best_fit: int, best_individuals: List[List[int]]):
    coin_flip = np.random.rand(1)
    # global migration
    if coin_flip >= 0.5:
        for idi in range(len(best_individuals)):
            best_individuals[idi] = best
    # local migration
    else:
        best_local = None
        best_local_fit = None
        # find best individual in population
        for ind in best_individuals:
            if best_local_fit is None or fitness(ind, clauses_no) > best_local_fit:
                best_local = ind
                best_local_fit = fitness(ind, clauses_no)
        # replace some variables with the best one
        replace_num = np.random.randint(low=1, high=pop_size + 1)
        replace_idx = np.random.choice([i for i in range(pop_size)], size=replace_num, replace=False)
        for idr in replace_idx:
            best_individuals[idr] = best_local

def q_gate_membrane(q: QbitIndividual, x: List[int], b: List[int], best_fit: int, clauses_no: int,
                    mapping: Dict[int, int]) -> QbitIndividual:
    x_fit = fitness(x, clauses_no, True, mapping)

    new_q = QbitIndividual(q.l)
    for idx, qi in enumerate(q):
        new_q[idx] = qi
    if not (x_fit <= best_fit):
        for idx, (xi, bi) in enumerate(zip(x, b)):
            cos_theta = 1
            sin_theta = 0
            if q[idx].get_quadrant() == 1:
                sign = 1
            else:
                sign = -1
            if xi == 0 and bi == 1:
                cos_theta = np.cos(sign * np.pi * 0.01)
                sin_theta = np.sin(sign * np.pi * 0.01)
            elif xi == 1 and bi == 0:
                cos_theta = np.cos(sign * np.pi * -0.01)
                sin_theta = np.sin(sign * np.pi * -0.01)
            alpha, beta = q[idx]
            new_q[idx].alpha = alpha * cos_theta - beta * sin_theta
            new_q[idx].beta = alpha * sin_theta + beta * cos_theta
    return new_q


def qiea_membrane(var_no: int, clauses_no: int, time_steps: int, qpop: List[QbitIndividual], best: QbitIndividual,
                  mapping: Dict[int, int])\
                  -> Tuple[List[QbitIndividual],
                  QbitIndividual, int]:
    best_fit = 10 ** 4
    if best is not None:
        best_fit = fitness(best, clauses_no, True, mapping)
    pop, best, best_fit = collapse_qbit_individuals_membrane(qpop, clauses_no, best, best_fit, mapping)
    best_individuals = copy.deepcopy(pop)
    print(f"best pop is {best} and score is {best_fit}")

    for t in range(1, time_steps):
        # get population
        pop, best, best_fit = collapse_qbit_individuals_membrane(qpop, clauses_no, best, best_fit, mapping)
        # update qbit individuals
        for idx, (qind, pind) in enumerate(zip(qpop, pop)):
            new_q = q_gate_membrane(qind, pind, best, best_fit, clauses_no, mapping)
            qpop[idx] = new_q
        # update best individuals
        for idx, (pind, bind) in enumerate(zip(pop, best_individuals)):
            pind_fitness = fitness(pind, clauses_no, True, mapping)
            bind_fitness = fitness(bind, clauses_no, True, mapping)

            if pind_fitness < bind_fitness:
                best_individuals[idx] = pind
        # do migrations
        if idx % MIG_PERIOD == 0:
            migrate_membrane(var_no, clauses_no, len(qpop), best, best_fit, best_individuals)

        # check terminating condition
        if t % 1000 == 0:
            print(f"best pop is {best} and score is {best_fit}")
        if best_fit == 0:
            print(t)
            print(f"best pop is {best} and score is {best_fit}")
            print("am rezolvat")
            break
    return qpop, best, fitness(best, clauses_no, True, mapping)

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
        qiea(var_no, clauses_no, 20000)
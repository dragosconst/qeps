from typing import List, Tuple
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

    def get(self):
        p = []
        for qbit in self.qbits:
            ai = qbit.alpha ** 2
            bi = qbit.beta ** 2
            pi = np.random.choice([0,1], p=(ai, bi), size=1)
            p.append(pi)
        return p

best_individuals = []
best = None
best_fit = None
variable_clauses_mapping = dict()
POP_SIZE = 1

def fitness(x: List[int], clauses_no: int) -> int:
    clauses_true = set()
    for idx, xi in enumerate(x):
        idx = idx + 1
        if xi:
            for clause in variable_clauses_mapping[idx]:
                clauses_true.add(clause)
        else:
            if -idx not in variable_clauses_mapping:
                continue
            for clause in variable_clauses_mapping[-idx]:
                clauses_true.add(clause)
    return clauses_no - len(clauses_true)


def q_gate(q: QbitIndividual, x: List[int], b: List[int], best_fit: int, clauses_no: int) -> QbitIndividual:
    x_fit = fitness(x, clauses_no)

    new_q = QbitIndividual(q.l)
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
    # collapse qbits for the first time
    for qind in qpop:
        pind = qind.get()
        pind_fit = fitness(pind, clauses_no)
        if best_fit is None or best_fit > pind_fit:
            best_fit = pind_fit
            best = pind
        pop.append(pind)
    return pop

def qiea(var_no: int, clauses_no: int, time_steps: int):
    global best, best_fit, best_individuals, POP_SIZE

    qpop = [QbitIndividual(var_no) for i in range(POP_SIZE)]
    pop = collapse_qbit_individuals(qpop, clauses_no)
    best_individuals = copy.deepcopy(pop)

    for t in range(1, time_steps):
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
        # check terminating condition
        if t % 1000 == 0:
            print(f"best pop is {best} and score is {best_fit}")
        if best_fit == 0:
            print(t)
            print(f"best pop is {best} and score is {best_fit}")
            print("am rezolvat")
            break


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
            if biggest_var is None or biggest_var < variable:
                biggest_var = variable
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
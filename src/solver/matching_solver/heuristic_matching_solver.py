from typing import Optional, List

from mapfmclient import Problem

from src.solver.epeastar.epeastar import EPEAStar
from src.solver.epeastar.independence_detection import IDSolver
from src.util.agent import Agent
from src.util.coordinate import Coordinate
from src.util.grid import Grid
from src.util.path import Path


class HeuristicMatchingSolver:
    """
    Solves a problem using heuristic matching.
    With heuristic matching, the A* heuristic for an individual agent is the distance to the closest goal of the
    same color.
    """

    def __init__(self, problem: Problem, independence_detection=True):
        """
        Constructs a HeuristicMatchingSolver instance
        :param problem:                The MAPFM problem that has to be solved
        :param independence_detection: Whether Independence Detection (ID) should be used
        """
        self.problem = problem
        self.independence_detection = independence_detection
        agents = [Agent(Coordinate(s.x, s.y), s.color, i) for i, s in enumerate(problem.starts)]
        self.grid = Grid(problem.width, problem.height, problem.grid, agents, problem.goals)
        if self.independence_detection:
            self.solver = IDSolver(self.grid)
        else:
            self.solver = EPEAStar(self.grid)

    def solve(self) -> Optional[List[Path]]:
        """
        Solves the problem with which the solver instance was instantiated
        :return:    Solution if it was found
        """
        return self.solver.solve()[0]
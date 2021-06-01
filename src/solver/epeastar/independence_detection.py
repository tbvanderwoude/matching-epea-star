from copy import copy
from typing import List, Tuple, Optional

from src.solver.epeastar.epeastar import EPEAStar
from src.solver.epeastar.mapf_problem import MAPFProblem
from src.util.agent import Agent
from src.util.cat import CAT
from src.util.path import Path
from src.util.path_set import PathSet


def find_conflict(paths: List[Path]) -> Optional[Tuple[int, int]]:
    """
    Checks if there are vertex and edge conflicts between paths
    :param paths:   Paths to check conflicts with
    :return:        Agent identifiers of first conflicting paths
    """
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if paths[i].conflicts(paths[j]):
                return paths[i].identifier, paths[j].identifier
    return None


class IDSolver:
    """
    Solver that uses Independence Detection:
    First solve for all agents individually. If paths are conflicting, merge the agents and solve for the new group
    """

    def __init__(self, problem: MAPFProblem, agents: List[Agent], cat: Optional[CAT], max_value=float('inf')):
        """
        Constructs an IDSolver instance
        :param problem:         MAPF problem instance that needs to be solved
        :param agents:          Agents that belong to the problem
        :param cat:             Additional Collision Avoidance Table that should be used in calculating the result
        :param max_value:       Maximum allowed value of the solver. Stop the solver if the value is exceeded
        """
        self.paths = []
        self.problem = problem
        self.agents = agents
        self.cost = 0
        self.max_value = max_value
        self.path_set = PathSet(self.agents, self.problem.heuristic)
        self.cats = []
        if cat is not None:
            self.cats.append(cat)
        self.cats.append(self.path_set.cat)

    def solve(self) -> Optional[Tuple[list, int]]:
        """
        Solves the original problem
        :return:    Solution with paths for every agent and the cost of the solution
        """
        agents = copy(self.agents)
        groups = []

        # Solve for every group
        total_cost = sum(self.problem.heuristic[agent.color][agent.coord.y][agent.coord.x] for agent in self.agents)
        for agent in agents:
            self.agents = [agent]
            total_cost -= self.problem.heuristic[agent.color][agent.coord.y][agent.coord.x]
            solver = EPEAStar(self.problem, self.agents, self.cats, max_cost=self.max_value - total_cost)
            solution = solver.solve()
            if solution is None:
                return None
            agent_paths, cost = solution
            self.path_set.update(agent_paths)
            total_cost += cost

            groups.append(([agent], cost))

            self.paths.append(agent_paths[0])

        self.cost = total_cost

        # Find and resolve conflicts until solution is conflict-free
        conflict = find_conflict(self.paths)
        while conflict is not None:
            a, b = conflict
            groups = self.merge_groups(groups, a, b, self.cats)
            if groups is None:
                return None
            conflict = find_conflict(self.paths)
        return self.paths, self.cost

    def merge_groups(self,
                     groups: List[Tuple[List[Agent], int]],
                     agent_a_id: int,
                     agent_b_id: int,
                     cats: List[CAT]) -> Optional[List[Tuple[List[Agent], int]]]:
        """
        Merge two groups with agents with conflicting paths
        :param groups:      List of all groups
        :param agent_a_id:  Identifier of the first conflicting agent
        :param agent_b_id:  Identifier of the second conflicting agent
        :param cats:        List of Collision Avoidance Tables
        :return:            New list of groups
        """
        group_a = None
        group_b = None
        group_a_id = 0
        for i, group in enumerate(groups):
            for agent in group[0]:
                if agent.identifier == agent_a_id:
                    group_a = group
                    group_a_id = i
                    break
                if agent.identifier == agent_b_id:
                    group_b = group
                    break
        assert group_a is not None
        assert group_b is not None

        self.cost -= group_a[1] + group_b[1]

        # Combine groups a and b
        new_agents = group_a[0] + group_b[0]

        # Try to solve new group
        self.agents = new_agents
        solver = EPEAStar(self.problem, self.agents, cats, self.max_value - self.cost)

        solution = solver.solve()
        if solution is None:
            return None
        group_paths, cost = solution

        self.path_set.update(group_paths)

        groups[group_a_id] = (new_agents, cost)
        groups.remove(group_b)

        self.cost += cost

        for agent, path in zip(new_agents, group_paths):
            self.paths[agent.identifier] = path
        return groups

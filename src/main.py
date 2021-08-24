import subprocess
from typing import Optional

from mapf_branch_and_bound.bbsolver import solve_bb
from mapfmclient import Solution, MapfBenchmarker, Problem, BenchmarkDescriptor

from src.solver.algorithm_descriptor import AlgorithmDescriptor, Algorithm
from src.solver.solver import Solver


def solve(problem: Problem, algorithm: AlgorithmDescriptor) -> Solution:
    """
    Solves the given problem and returns a solution
    :param problem:     Multi-Agent Pathfinding problem with Matching
    :param algorithm:   Descriptor of the algorithm version that should be used to solve the problem
    :return:            Solution
    """
    solver = Solver(problem, algorithm)
    solution, tracker = solver.solve()
    return Solution.from_paths(solution)

def solve_subroutine(starting_problem: Problem, upper_bound: Optional[int]):
    algorithm = AlgorithmDescriptor(Algorithm.HeuristicMatching,
                                               independence_detection=True)
    if not upper_bound:
        upper_bound = float("inf")
    else:
        upper_bound += len(starting_problem.starts)
    solver = Solver(starting_problem, algorithm)
    solution, tracker = solver.solve(upper_bound=upper_bound + 1)
    if solution is None:
        print("Failed to find solution")
        return None
    return Solution.from_paths(solution)

def solve_branch_and_bound(problem: Problem):
    return solve_bb(problem, solve_subroutine)

def get_version(is_debug, current_version) -> str:
    """
    Generates a version number string
    :param is_debug:        Indicates if the benchmark is run in debug mode
    :param current_version: Current version number
    :return:                Git commit hash if debug, else real version number
    """
    if not is_debug:
        return current_version
    git_hash = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    return f"{git_hash}"


def run_online_benchmarker():
    """
    Runs the online mapfmclient benchmarker.
    """
    version = '1.0.1'
    debug = False
    api_token = open('../apitoken.txt', 'r').read().strip()
    benchmarker = MapfBenchmarker(api_token, BenchmarkDescriptor(93), "EPEA* with branch-and-bound",
                                  get_version(debug, version), debug,
                                  solver=solve_branch_and_bound,
                                  cores=1)
    benchmarker.run()


if __name__ == '__main__':
    run_online_benchmarker()

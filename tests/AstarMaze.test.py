import pytest
import numpy as np
from your_astar_module import AStarMaze  # replace with actual module name

# Basic 4x4 maze
simple_maze = np.array([
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 1, 0, 0]
])

def test_simple_path():
    solver = AStarMaze(simple_maze)
    path = solver.solve_shortest_path((0, 0), (3, 3))
    assert path[0] == (0, 0)
    assert path[-1] == (3, 3)
    assert all(simple_maze[x, y] == 0 for x, y in path)

def test_no_path():
    blocked_maze = np.array([
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 0]
    ])
    solver = AStarMaze(blocked_maze)
    with pytest.raises(ValueError, match="No path found"):
        solver.solve_shortest_path((0, 0), (2, 2))

def test_start_on_obstacle():
    solver = AStarMaze(simple_maze)
    with pytest.raises(ValueError, match="Start position.*invalid or on an obstacle"):
        solver.solve_shortest_path((0, 1), (3, 3))

def test_end_on_obstacle():
    solver = AStarMaze(simple_maze)
    with pytest.raises(ValueError, match="End position.*invalid or on an obstacle"):
        solver.solve_shortest_path((0, 0), (1, 3))

def test_out_of_bounds():
    solver = AStarMaze(simple_maze)
    with pytest.raises(ValueError, match="Start position.*invalid or on an obstacle"):
        solver.solve_shortest_path((-1, 0), (3, 3))

    with pytest.raises(ValueError, match="End position.*invalid or on an obstacle"):
        solver.solve_shortest_path((0, 0), (10, 10))

def test_non_numpy_maze():
    with pytest.raises(TypeError):
        AStarMaze([[0, 1], [1, 0]])

def test_non_2d_maze():
    with pytest.raises(ValueError):
        AStarMaze(np.array([[[0]]]))

def test_invalid_start_type():
    solver = AStarMaze(simple_maze)
    with pytest.raises(TypeError):
        solver.solve_shortest_path("start", (3, 3))

def test_invalid_end_type():
    solver = AStarMaze(simple_maze)
    with pytest.raises(TypeError):
        solver.solve_shortest_path((0, 0), "end")

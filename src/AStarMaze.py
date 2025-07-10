import logging
import heapq
import numpy as np
from typing import Any, List, Optional, Tuple

class AStarMaze:
    """
    AStarMaze resolves the A* shortest path search on a 2D numpy maze.
    The maze should be a 2D numpy array with 0 for free cells and any other value for obstacles.

    Attributes:
        maze (np.ndarray): The 2D numpy array representing the maze grid.
        rows (int): The number of rows in the maze.
        cols (int): The number of columns in the maze.
        logger (logging.Logger): Logger instance for this class.

    Example:
        >>> maze = np.zeros((5, 5), dtype=int)
        >>> astar = AStarMaze(maze)
        >>> path = astar.solve_shortest_path((0, 0), (4, 4))
    """
    class Node:
        """
        Node for use in the A* algorithm.

        Attributes:
            position (Tuple[int, int]): (x, y) position in the maze.
            parent (Optional[AStarMaze.Node]): Parent node in the path.
            g (float): Cost from start to current node.
            h (float): Heuristic cost from current node to end.
            f (float): Total cost (g + h).
        """
        def __init__(self, position: Tuple[int, int], parent: Optional["AStarMaze.Node"] = None) -> None:
            self.position: Tuple[int, int] = position
            self.parent: Optional["AStarMaze.Node"] = parent
            self.g: float = 0
            self.h: float = 0
            self.f: float = 0

        def __lt__(self, other):
            return self.f < other.f

        def __eq__(self, other):
            return self.position == other.position

    def __init__(self, maze: np.ndarray, log_level: int = logging.WARNING) -> None:
        """
        Initializes the AStarMaze instance.

        Args:
            maze (np.ndarray): 2D numpy array representing the maze.
            log_level (int): Logging level for this instance.
        """
        if not isinstance(maze, np.ndarray):
            raise TypeError("Maze must be a numpy ndarray.")
        if maze.ndim != 2:
            raise ValueError("Maze must be a 2D numpy array.")
        self.maze: np.ndarray = maze
        self.rows, self.cols = maze.shape

        self.logger = logging.getLogger(f"{__name__}.AStarMaze")
        self.logger.setLevel(log_level)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)

    def _is_valid_position(self, position: Tuple[int, int]) -> bool:
        """
        Checks if a position is within the bounds of the maze and not an obstacle.

        Args:
            position (Tuple[int, int]): The (x, y) position to check.

        Returns:
            bool: True if position is valid and free, False otherwise.
        """
        x, y = position
        return (
            0 <= x < self.rows and
            0 <= y < self.cols and
            self.maze[x, y] == 0
        )

    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Returns the squared Euclidean distance heuristic between two points.

        Args:
            pos1 (Tuple[int, int]): First position.
            pos2 (Tuple[int, int]): Second position.

        Returns:
            float: The squared Euclidean distance.
        """
        return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2

    def solve_shortest_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Solves for the shortest path from start to end using the A* algorithm.

        Args:
            start (Tuple[int, int]): Start position (x, y).
            end (Tuple[int, int]): End position (x, y).

        Returns:
            Optional[List[Tuple[int, int]]]: The shortest path as a list of positions, or None if no path exists.
        """
        if not (isinstance(start, tuple) and isinstance(end, tuple)):
            raise TypeError("Start and end must be tuples (x, y).")

        if not self._is_valid_position(start):
            raise ValueError(f"Start position {start} is invalid or on an obstacle.")

        if not self._is_valid_position(end):
            raise ValueError(f"End position {end} is invalid or on an obstacle.")

        start_node = self.Node(start)
        end_node = self.Node(end)

        open_list = []
        heapq.heappush(open_list, start_node)
        closed_set = set()

        while open_list:
            current_node = heapq.heappop(open_list)
            self.logger.debug(f"Exploring node: {current_node.position} with f={current_node.f}")

            if current_node == end_node:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                self.logger.info(f"Shortest Path found: {path[::-1]}")
                return path[::-1]

            closed_set.add(current_node.position)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor_pos = (current_node.position[0] + dx,
                                current_node.position[1] + dy)

                if not self._is_valid_position(neighbor_pos):
                    continue
                if neighbor_pos in closed_set:
                    continue

                neighbor = self.Node(neighbor_pos, current_node)
                neighbor.g = current_node.g + 1
                neighbor.h = self._heuristic(neighbor_pos, end_node.position)
                neighbor.f = neighbor.g + neighbor.h

                in_open = False
                for node in open_list:
                    if neighbor == node and neighbor.g >= node.g:
                        in_open = True
                        break

                if not in_open:
                    heapq.heappush(open_list, neighbor)
                    self.logger.debug(f"Added to open list: {neighbor.position} with f={neighbor.f}")

        self.logger.warning(f"No shortest path found between: start={start}, end={end}")
        return None

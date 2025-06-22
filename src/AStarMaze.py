import numpy as np
import heapq
import logging

class AStarMaze:
    class Node:
        def __init__(self, position, parent=None):
            self.position = position
            self.parent = parent
            self.g = 0
            self.h = 0
            self.f = 0

        def __lt__(self, other):
            return self.f < other.f

        def __eq__(self, other):
            return self.position == other.position

    def __init__(self, maze: np.ndarray, log_level=logging.WARNING):
        if not isinstance(maze, np.ndarray):
            raise TypeError("Maze must be a numpy ndarray.")
        if maze.ndim != 2:
            raise ValueError("Maze must be a 2D numpy array.")
        self.maze = maze
        self.rows, self.cols = maze.shape

        logging.basicConfig(
            level=log_level,
            format="%(levelname)s:%(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def _is_valid_position(self, position):
        x, y = position
        return (
            0 <= x < self.rows and
            0 <= y < self.cols and
            self.maze[x, y] == 0
        )

    def _heuristic(self, pos1, pos2): # -> List[] | None
        # Euclidean distance (squared)
        return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2

    def solve_shortest_path(self, start, end):
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

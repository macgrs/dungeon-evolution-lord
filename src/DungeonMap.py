import itertools
from copy import deepcopy
from random import randrange,randint

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import logging

from typing import Any, Dict, List, Optional, Tuple

from .Genotype import Genotype
from .AStarMaze import AStarMaze


class DungeonMap:
    """
    Core class for growing, previewing, and computing features of a playable dungeon from a Genotype.

    This class supports:
    + dungeon generation, using growing regions algorithm from the genotype Points Of Interest (POI) and
    linkage of the rooms using extended minimum spanning tree between the POIs 
    + dungeon preview utilities, with different plots
    + dungeon gameplay features such as path validation, room connectivity, placement of POIs

    Attributes:
        genotype (Genotype): The genotype from which this dungeon is grown.
        dungeon (np.ndarray): 2D numpy array representing the dungeon layout.
        features (Dict[str, Any]): Computed features and statistics about the dungeon.
        logger (logging.Logger): Logger instance for this class.
    """
    def __init__(self, dimension: int, genotype: Genotype) -> None:
        """
        Initialize a DungeonMap instance.

        Args:
            dimension (int): The width and height of the dungeon (dungeon is always square).
            genotype (Genotype): The Genotype instance encoding points of interest and parameters.
        """
        self.dungeon_shape: Tuple[int, int] = (dimension, dimension)
        self.set_dungeon_Genotype(genotype=genotype)
        self.dungeonmap: Optional[np.ndarray] = None
        self.shortest_path: Optional[Any] = None
        self.shortest_path_rooms: Optional[Any] = None
        self.feature_dict: dict = {}
        self.genotype: Genotype = genotype

        self.logger = logging.getLogger(f"{__name__}.DungeonMap")
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
            self.logger.addHandler(handler)

    # GENOTYPE
    # --------
    def set_dungeon_Genotype(self, genotype: Genotype) -> None:
        """
        Assigns a new genotype to the dungeon.

        Args:
            genotype (Genotype): The genotype to assign.

        Raises:
            ValueError: If the genotype is not valid for this dungeon's dimensions.
        """
        if genotype is not None:
            if genotype.is_valid(xy_bounds=self.dungeon_shape):        
                self.genotype=genotype
            else:
                raise ValueError(f"Provided Genotype is not valid regarding DungeonMap spec, maybe some pois out of bounds ?")

    def get_dungeon_entrypoints_positions(self, upscaled: bool) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Returns the (entry, exit) positions from the genotype.

        Args:
            upscaled (bool): If True, multiplies entry/exit coordinates by 2 as the playable/upscaled dungeon is twice the size
            of the genotype shape

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]]: (entry, exit) positions as (x, y) tuples.

        Raises:
            ValueError: If the genotype does not have POIs set.
        """
        if self.genotype.pois == None:
            raise ValueError(f"Dungeon' genotype does not have pois: {self.genotype.pois}")
        d_in = self.genotype.pois[np.where(self.genotype.pois[:,2] == 2)][0,:2]
        d_out = self.genotype.pois[np.where(self.genotype.pois[:,2] == 3)][0,:2]
        if upscaled is True:
            return tuple(d_in*2), tuple(d_out*2)
        else:
            return tuple(d_in), tuple(d_out)

    # DUNGEON FEATURES
    # ----------------
    def extract_dungeon_features(self) -> Optional[dict]:
        """
        Extracts and computes various features/statistics of the dungeon:

        - corridorness (float): The corridor/room tile ratio for the whole dungeon.
        - nolliness (float): The ratio of empty to walkable spaces.
        - spath_length (int): The length of the shortest path.
        - spath_rooms_types_completeness_ratio (float): Ratio of room types crossed by the shortest path over possible room types.
        - spath_rooms_diversity_ratio (float): Diversity of rooms crossed by the shortest path.
        - spath_corridorness (float): Corridor/room tile ratio for the shortest path.

        Returns:
            Optional[dict]: The feature dictionary if extraction succeeded, None otherwise.

        Raises:
            ValueError: If the dungeon map has not been computed/generated yet.
        """

        ## Dungeon map
        feature_dict = {}
        if not isinstance(self.dungeonmap, np.ndarray):
            raise ValueError(f"Compute the dungeon map before extracting any feature, dungeonmap={self.dungeonmap}")
        
        ### Corrdiorness
        tiles_count=(self.dungeonmap.shape[0] * self.dungeonmap.shape[1])
        corridors_count = np.count_nonzero(self.dungeonmap == 1)
        wall_count = np.count_nonzero(self.dungeonmap == 0)
        feature_dict["corridorness"] = corridors_count / (tiles_count - wall_count)
    
        ### Nolliness,the ratio of empty to walkable spaces. 1=full walkable
        arr_walkable = self._get_walkable_dungeon_surfaces()
        feature_dict["nolliness"] = 1 - np.count_nonzero(arr_walkable) / (arr_walkable.shape[0]*arr_walkable.shape[1])

        ## Shortest path
        try:
            if not self.shortest_path is None:
                logging.info(f"Shortest path already computed, shortest_path={self.shortest_path}. Continue.")
            shortest_path = self._solve_shortestpath_in_dungeon()
            self.shortest_path = shortest_path
        except ValueError as ve:
            logging.error(f"Dungeon shortest path failed, with {ve}")
        
        ### Features in shortest path
        if self.shortest_path is None:
            logging.error(f"No shortest path found")
            return None
        
        feature_dict["spath_length"] = len(self.shortest_path)

        ## Rooms in Shortest Path
        spath_rooms=[]
        for pt in self.shortest_path:
            spath_rooms.append(self.dungeonmap[pt[0],pt[1]])
        spath_rooms_types_uniques=set(spath_rooms)
        # spath_rooms_sequence_of_rooms = [k for k, g in itertools.groupby(spath_rooms)]
        spath_rooms_sequence_of_rooms_corridorrless = [k for k, g in itertools.groupby(spath_rooms) if k != 1]
        # logging.debug(f"spath_rooms_sequence_of_rooms={spath_rooms_sequence_of_rooms}, corridorless={spath_rooms_sequence_of_rooms_corridorrless}")

        ### Completeness, the ratio of the types of rooms crossed by the shortest path, on the number of possible rooms let by the Genotype. If =1, all the Genotype' room types are crossed
        feature_dict["spath_completeness"] = len(spath_rooms_types_uniques) / 6       # With 6 the number of rooms types (entrance, corridor, monster, exit...) 

        ### Diversity, ratio that measures the variety of rooms crossed with a score computed on the good alternance of rooms types (not always the same)
        score=0
        for i in range(len(spath_rooms_sequence_of_rooms_corridorrless)-1):
            if spath_rooms_sequence_of_rooms_corridorrless[i] != spath_rooms_sequence_of_rooms_corridorrless[i+1]:
                score+=1
        if (len(spath_rooms_sequence_of_rooms_corridorrless)-1) <= 0:
            feature_dict["spath_diversity"] = 0
        else:
            feature_dict["spath_diversity"] = score/(len(spath_rooms_sequence_of_rooms_corridorrless)-1)

        ### spath_corridorness, 
        spath_corridors_count = spath_rooms.count(1)
        # print(f"nb_corridors={spath_rooms.count(1)} nb_others={len(spath_rooms)-spath_rooms.count(1)}")
        feature_dict["spath_corridorness"] = spath_corridors_count / len(spath_rooms)
        
        return feature_dict

    def _get_walkable_dungeon_surfaces(self) -> np.ndarray:
        """
        Returns a binary array of the dungeonmap where walkable tiles are marked as 1, others as 0.

        Returns:
            np.ndarray: Walkable surface mask (same shape as dungeonmap).
        """
        if not isinstance(self.dungeonmap, np.ndarray):
            raise ValueError(f"No dungeon map has been provided, dungeonmap={self.dungeonmap}")
        convert_to_ones_and_zeros = np.vectorize(lambda x: 1 if x == 0 else 0, otypes=[int])
        return convert_to_ones_and_zeros(self.dungeonmap)
    
    def _solve_shortestpath_in_dungeon(self) -> Optional[List[Tuple[int, int]]]:
        """
        Computes the shortest path in the dungeon using A* implemented with AStarMaze class.

        Returns:
            Optional[List[Tuple[int, int]]]: The shortest path object (format depends on implementation).

        Raises:
            ValueError: If pathfinding fails or entry/exit are missing.
        """
        arr_walkable = self._get_walkable_dungeon_surfaces()
        walkable_dungeon_as_maze = AStarMaze(maze=arr_walkable)
        d_in, d_out = self.get_dungeon_entrypoints_positions(upscaled=True)
        shortest_path=walkable_dungeon_as_maze.solve_shortest_path(start=d_in, end=d_out)
        return shortest_path


    # Generate
    def grow_dungeonmap_from_genotype(self) -> None:
        """
        Grows (generates) the dungeon map (`self.dungeonmap`) as a 2D numpy array
        from the current genotype's POIs and growth procedure.

        This method applies deterministic procedural rules - growing regions, defining a circulation graph between
        POIs using minimum spanning tree - to produce a playable dungeon map, with rooms, corridors, and connections from the genotype.

        Raises:
            ValueError: If the genotype is not valid or not set.
        """
        if self.genotype.pois == None:
            raise ValueError(f"Dungeon' genotype does not have pois, can't grow a dungeon from nothing: {self.genotype.pois}")
        logging.info(f"--- Growing dungeonmap with {len(self.genotype.pois)} pois ---")
        if not isinstance(self.genotype, Genotype):
            raise ValueError(f"No valid Genotype found, can't grow a dungeon from {type(self.genotype)}")

        # Initialize the dungeonmap, as a np array with the phenotypal POIs
        arr_phenotypal = np.zeros(self.dungeon_shape)
        for poi in self.genotype.pois:
            arr_phenotypal[poi[0]][poi[1]] = poi[2]

        # Draw the dungeon rooms by growing each POI'region
        arr_grownrooms = self._get_dungeonrooms_by_growing_regions(arr_dungeonmap=arr_phenotypal, iteration_limit=self.genotype.growth_iterator_limit)
        # self._preview_dungeon(arr=arr_grownrooms)

        # Draw the dungeon paths, from the POI minimum spanning tree
        G_delaunay, T_minspan = self._build_utils_graphs_from_genotype()
        g_circu = self._build_circulation_graph(G_delaunay=G_delaunay, T_minspan=T_minspan, add_edges=1)
        logging.debug(f"Dungeon graphs edges: delaunay={len(G_delaunay.edges)}, minimum span tree={len(T_minspan.edges)}, circulation={len(g_circu.edges)}")
        arr_dungeonpaths = self._get_dungeonpath_from_circulation_graph(G_circulation=g_circu, dungeon_shape=self.dungeon_shape, pois=self.genotype.pois)
        self.dungeonmap = self._get_upscaled_dungeon_with_rooms_and_paths(arr_rooms=arr_grownrooms, G_circulation=g_circu)
        logging.debug(f"Upscaled to shape {arr_grownrooms.shape}")

    def _get_upscaled_dungeon_with_rooms_and_paths(self, arr_rooms, G_circulation):
        def upscale_dungeonroom_array(arr_rooms):
            upscaled = np.full(np.dot(arr_rooms.shape, 2), -1)
            for x in range(arr_rooms.shape[0]):
                for y in range(arr_rooms.shape[1]):
                    upscaled[x*2][y*2] = arr_rooms[x][y]
            return upscaled
        
        if self.genotype.pois == None:
            raise ValueError(f"Dungeon' genotype does not have pois, can't upscale the dungeon: {self.genotype.pois}")

        up_base  = upscale_dungeonroom_array(arr_rooms=arr_rooms)
        up_final = np.copy(up_base)
        up_pois  = np.hstack([np.dot(self.genotype.pois[:,:2],2), self.genotype.pois[:,2:]])
        
        up_paths = self._get_dungeonpath_from_circulation_graph(dungeon_shape=up_base.shape, G_circulation=G_circulation, pois=up_pois)

        for x in range(0, up_base.shape[0], 2):
            for y in range(0, up_base.shape[1]-1):
                if up_final[x][y] == -1:
                    # if up_paths[x][y] == 1:
                    #     up_final[x][y] = 10
                    if up_final[x][y-1] == up_final[x][y+1]:
                        # print(f'equal {up_base_pathed[x][y-1]}')
                        up_final[x][y] = up_final[x][y-1]

        for x in range(0, up_base.shape[0]-1):
            for y in range(up_base.shape[1]):
                if up_final[x][y] == -1:
                    # if up_paths[x][y] == 1:
                    #     up_final[x][y] = 10
                    if up_final[x-1][y] == up_final[x+1][y]:
                        # print(f'equal {up_base_pathed[x][y-1]}')
                        up_final[x][y] = up_final[x-1][y]  
        up_rooms_and_paths=np.maximum(up_final,up_paths)
        return up_rooms_and_paths

    ## Utils, rooms
    def _get_dungeonrooms_by_growing_regions(self, arr_dungeonmap: np.ndarray, iteration_limit=100) -> np.ndarray:
        def grow_regions(arr_dungeonmap: np.ndarray):
            if not len(arr_dungeonmap.shape) == 2:
                raise ValueError(f"Array map shall be flat 2 dimensional, actual shape is {arr_dungeonmap.shape}")

            nplus_map = np.zeros(arr_dungeonmap.shape)
            adj_coords = {(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0,-1), (1,-1)}

            for x in range(arr_dungeonmap.shape[0]):
                for y in range(arr_dungeonmap.shape[1]):
                    if arr_dungeonmap[x][y] > 0:
                        nplus_map[x][y] = arr_dungeonmap[x][y]
                        for dx, dy in adj_coords:
                            if 0 <= x+dx < arr_dungeonmap.shape[0] and 0 <= y+dy < arr_dungeonmap.shape[1]:
                                if arr_dungeonmap[x+dx][y+dy] == 0:
                                    nplus_map[x+dx][y+dy] = arr_dungeonmap[x][y]
            return nplus_map
        
        count = 0
        grown_map = np.copy(arr_dungeonmap)
        while 0 in grown_map and count < iteration_limit:
            logging.info(f"Iteration {count}, actual empty spaces={np.count_nonzero(grown_map == 0)}")
            logging.debug(f"Growing pois regions, iteration={count}")
            grown_map = grow_regions(arr_dungeonmap=grown_map)
            count+=1
        return grown_map

    def _get_phenotypal_dungeon(self) -> np.ndarray:
        if not isinstance(self.genotype, Genotype):
            raise ValueError(f"No valid Genotype found, can't grow a dungeon from {type(self.genotype)}")   
        if self.genotype.pois == None:
            raise ValueError(f"Dungeon' genotype does not have pois: {self.genotype.pois}")
           
        arr = np.zeros(self.dungeon_shape)
        for poi in self.genotype.pois:
            arr[poi[0]][poi[1]] = poi[2]
        return arr
    
    ## Utils, paths
    def _get_dungeonpath_from_circulation_graph(self, dungeon_shape, G_circulation, pois):
        arr_dungeonpath = np.zeros(dungeon_shape)
        for e in G_circulation.edges:
            p0 = pois[e[0]][:2]
            p1 = pois[e[1]][:2]
            new_line = self._line_get_angle(p0[0], p0[1], p1[0], p1[1], False)
            logging.debug(e, new_line)
            for pt in new_line:
                arr_dungeonpath[pt[0]][pt[1]] = 1
        return arr_dungeonpath

    def _build_utils_graphs_from_genotype(self) -> Tuple[nx.Graph, nx.Graph]:
        """
        Given an array-like of 2D points, return a NetworkX graph of the Delaunay triangulation,
        with edges weighted by cityblock (Manhattan) distance.
        """
        pois = np.asarray(self.genotype.pois, dtype=int)
        pos = pois[:, :2]  # only x, y
        tri = Delaunay(pos)

        # Extract edges from Delaunay triangles
        edges = set()
        for simplex in tri.simplices:
            for i, j in itertools.combinations(simplex, 2):
                edges.add(tuple(sorted((int(i), int(j)))))

        # Compute weights using cityblock distance
        e1, e2 = zip(*edges)
        weights = cdist(pos[list(e1)], pos[list(e2)], metric='cityblock')
        weighted_edges = [(i, j, w) for (i, j), w in zip(edges, weights.diagonal())]

        # Build graph
        G_delaunay = nx.Graph()
        G_delaunay.add_weighted_edges_from(weighted_edges)
        T_minspan=nx.minimum_spanning_tree(G_delaunay)
        return G_delaunay, T_minspan

    def _build_circulation_graph(self, G_delaunay, T_minspan, add_edges=0):
        G = deepcopy(T_minspan)
        g_diff = nx.difference(G_delaunay, T_minspan)
        if add_edges > len(g_diff.edges):
            logging.debug(f"add_edges={add_edges}, the Delaunay Graph is returned")
            return G_delaunay
        
        listof_new_edges = list(g_diff.edges)
        logging.debug(f"adding edges {listof_new_edges[:add_edges]} to the minimum spanning tree")
        G.add_edges_from(listof_new_edges[:add_edges])
        return G

    def _debug_plot_graph(self, G, pos, node_size=30, font_size=8):
        """
        Visualizes the graph with edge weights and labels.
        """
        plt.figure(figsize=(3, 3))
        nx.draw_networkx_edges(G, pos, width=2)
        nx.draw_networkx_labels(G, pos, font_size=font_size)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=font_size)
        nx.draw_networkx_nodes(G, pos, node_size=node_size)

        plt.axis("off")
        plt.tight_layout()
        plt.show()

    ### Line Drawing
    def _line_get_bresenham(self, x0,y0,x1,y1):
        points_line = []
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            logging.debug(x0, y0)
            points_line.append((x0, y0))
            e2 = 2 * error
            if e2 >= dy:
                if x0 == x1:
                    break
                error = error + dy
                x0 = x0 + sx
            if e2 <= dx:
                if y0 == y1:
                    break
                error = error + dx
                y0 = y0 + sy
        return points_line

    def _line_get_angle(self, x0, y0, x1, y1, l_shaped_angle=True):
        if x0 == x1:  # vertical
            return [(x0, y) for y in range(min(y0, y1), max(y0, y1)+1)]
        
        if y0 == y1:  # horizontal
            return [(x, y0) for x in range(min(x0, x1), max(x0, x1)+1)]
        
        if l_shaped_angle:
            # First vertical, then horizontal
            vline = [(x0, y) for y in range(min(y0, y1), max(y0, y1)+1)]
            hline = [(x, y1) for x in range(min(x0, x1), max(x0, x1)+1)]
            logging.debug(f"L-shape: vline={vline}, hline={hline}")
            return vline + vline[1:]
        else:
            # First horizontal, then vertical
            hline = [(x, y0) for x in range(min(x0, x1), max(x0, x1)+1)]
            vline = [(x1, y) for y in range(min(y0, y1), max(y0, y1)+1)]
            logging.debug(f"reverse-L-shape: hline={hline}, vline={vline}")
            return hline + vline[1:]

    def _debug_preview_line_draw(self, line_get_function):
        endpoints=[(1,1),(1,6),(6,1),(6,8)]
        for e in itertools.islice(itertools.combinations(endpoints, 2),6):
            print(e)
            points_in_line = line_get_function(e[0][0],e[0][1],e[1][0],e[1][1], False)
            preview_line_arr=np.zeros((10,10))
            for p in points_in_line:
                preview_line_arr[p[0]][p[1]] = 1
            self._preview_dungeon(preview_line_arr)

    # Previews
    def _preview_dungeon(self, arr: np.ndarray, palette=None):
        """
        Displays a 2D numpy array using a custom palette for values from 0 to 10.
        """
        if palette is None:
            # Default: tab10 colors (10 colors max)
            palette = [
                "#ffffff", "#b4b4b4", "#8ad98a", "#145221", "#c9c9c9",
                "#c13e3e", "#dbbd24", "#973636", "#bcbd22", "#17becf", "#000000"
            ]

        cmap = ListedColormap(palette[:11])  # Use only the first 11 colors

        fig, ax = plt.subplots()
        cax = ax.imshow(arr, vmin=0, vmax=10, cmap=cmap, interpolation='nearest', aspect='equal')

        # Colorbar with discrete ticks
        # cbar = plt.colorbar(cax, ticks=range(11))
        # cbar.set_label("Tile Type")

        # Ticks & layout
        ax.set_xticks(np.arange(arr.shape[1]))
        ax.set_yticks(np.arange(arr.shape[0]))
        ax.invert_yaxis()
        # ax.set_title("Dungeon Preview")

        plt.tight_layout()
        plt.show()
        
    def preview_full_dungeon(self) -> None:
        if not isinstance(self.dungeonmap, np.ndarray):
            raise ValueError(f"No dungeon map has been provided, dungeonmap={self.dungeonmap}")
        self._preview_dungeon(self.dungeonmap)
    
    def preview_phenotypal_dungeon(self) -> None:
        arr=self._get_phenotypal_dungeon()
        self._preview_dungeon(arr)
    
    def preview_walkable_dungeon(self) -> None:
        walkable_palette = [
                "#ffffff", "#000000", "#8ad98a", "#145221", "#c9c9c9",
                "#c13e3e", "#dbbd24", "#973636", "#bcbd22", "#17becf", "#000000"
            ]
        arr=self._get_walkable_dungeon_surfaces()
        self._preview_dungeon(arr, palette=walkable_palette)

    def preview_speedrunnable_dungeon(self) -> None:
        walkable_palette = [
                "#ffffff", "#000000", "#8ad98a", "#145221", "#c9c9c9",
                "#c13e3e", "#dbbd24", "#973636", "#bcbd22", "#17becf", "#000000"
            ]
        arr=self._get_walkable_dungeon_surfaces()

        if self.shortest_path is None:
            raise ValueError(f"Compute the dungeon features first")
        for pt in self.shortest_path:
            arr[pt[0],pt[1]] = 4
        d_in, d_out = self.get_dungeon_entrypoints_positions(upscaled=True)
        arr[d_in[0],d_in[1]] = 2
        arr[d_out[0],d_out[1]] = 3
        self._preview_dungeon(arr, palette=walkable_palette)
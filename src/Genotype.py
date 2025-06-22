import random
import numpy as np

from typing import Tuple

import logging

class Genotype:
    """
        Types of POIs:
            + `-1`: Blocked
            + `0`,`1`: <Reserved>
            + `2`: Entry  (neutral room)
            + `3`: Exit   (neutral room)
            + `4`: Neutral
            + `5`: Monster
            + `6`: Treasure
        
    """
    def __init__(self):
        self.pois = None
        self.number_of_pois = 5
        self.dungeon_shape = (16, 16)
        self.growth_iterator_limit = 0
    
    def is_valid(self, xy_bounds:Tuple[int,int]) -> bool:
        # test if: (a) the pois locations are within xy bounds, (b) there is one and only one entry and exit, (c) there are no reserved values such as `0`,`1`
        return True
    
    def generate_genotype(self, dungeon_shape:Tuple[int,int], number_of_pois, growth_iterator_limit) -> None:
        if number_of_pois < 3:
            raise ValueError(f"There must be at least 3 number_of_pois, actual={number_of_pois}")
        pois = [[random.randrange(dungeon_shape[0]), random.randrange(dungeon_shape[1]), 2],[random.randrange(dungeon_shape[0]), random.randrange(dungeon_shape[1]), 3]]
        additional_pois = [[random.randrange(dungeon_shape[0]), random.randrange(dungeon_shape[1]), random.randint(4, 6)] for n in range(number_of_pois-2)]
        self.pois=np.asarray(pois+additional_pois, dtype=int)
        self.number_of_pois = number_of_pois
        self.dungeon_shape = dungeon_shape
        self.growth_iterator_limit = growth_iterator_limit

    def mutate_from_genotype(self, genotomut, global_mutation_strength:float = 1.0):
        def random_move_within_manhattan_np(x, y, x_max, y_max, max_distance):
            """
            Efficiently computes a random move from (x, y) within a specified Manhattan distance
            on a bounded grid using NumPy.

            Args:
                x (int): Current x-coordinate.
                y (int): Current y-coordinate.
                x_max (int): Maximum x bound (inclusive).
                y_max (int): Maximum y bound (inclusive).
                max_distance (int): Maximum Manhattan distance for movement.

            Returns:
                tuple: Randomly selected valid new (x, y) coordinates.
            """

            # Create a grid of offsets within the max_distance
            dx = np.arange(-max_distance, max_distance + 1)
            dy = np.arange(-max_distance, max_distance + 1)
            dx_grid, dy_grid = np.meshgrid(dx, dy)

            # Compute Manhattan distance for all offset pairs
            manhattan_dist = np.abs(dx_grid) + np.abs(dy_grid)
            mask = (manhattan_dist <= max_distance) & (manhattan_dist > 0)

            # Apply mask to get valid offsets
            dx_valid = dx_grid[mask]
            dy_valid = dy_grid[mask]

            # Compute all candidate new positions
            new_x = x + dx_valid
            new_y = y + dy_valid

            # Bound checking
            within_bounds = (0 <= new_x) & (new_x < x_max) & (0 <= new_y) & (new_y < y_max)
            new_x = new_x[within_bounds]
            new_y = new_y[within_bounds]

            if len(new_x) == 0:
                return (x, y)  # No valid moves

            idx = np.random.randint(len(new_x))
            return int(new_x[idx]), int(new_y[idx])

        # Mutate poi
        dungeon_min_dimension = min(genotomut.dungeon_shape[0],genotomut.dungeon_shape[1]) 
        mutation_strength_poi_move = round(global_mutation_strength * (dungeon_min_dimension / 4)) # 2 is a bit strong
        mutation_strength_poi_change_chance = 0.5 * global_mutation_strength
        pois_existing_mutated = []
        for poi in genotomut.pois:
            x, y = random_move_within_manhattan_np(x=poi[0], y=poi[1], x_max=genotomut.dungeon_shape[0], y_max=genotomut.dungeon_shape[1], max_distance=mutation_strength_poi_move)
            if poi[2] > 3:
                if random.random() + mutation_strength_poi_change_chance > 1:
                    pois_existing_mutated.append([x, y, random.choice([4,5,6])])
                else:
                    pois_existing_mutated.append([x, y, poi[2]])
            else:
                pois_existing_mutated.append([x, y, poi[2]])
        logging.debug(pois_existing_mutated)

        # Mutate poi count
        mutated_pois = []
        mutation_strength_poi_number_dx = 1.5 * global_mutation_strength
        mutated_number_of_pois = round(random.uniform(max(3, genotomut.number_of_pois - mutation_strength_poi_number_dx), genotomut.number_of_pois + mutation_strength_poi_number_dx))

        if mutated_number_of_pois > genotomut.number_of_pois:
            pois_difference = abs(mutated_number_of_pois-genotomut.number_of_pois)
            # Add new random pois
            new_pois = [[random.randrange(genotomut.dungeon_shape[0]), random.randrange(genotomut.dungeon_shape[1]), random.randint(4, 6)] for n in range(pois_difference)]
            mutated_pois = pois_existing_mutated + new_pois
        elif mutated_number_of_pois < genotomut.number_of_pois:
            pois_difference = abs(mutated_number_of_pois-genotomut.number_of_pois)
            arr_pois = np.asarray(genotomut.pois)
            potential_pois = arr_pois[(arr_pois[:, 2] != 2) & (arr_pois[:, 2] != 3)].tolist()
            endpoints_pois = arr_pois[(arr_pois[:, 2] == 2) | (arr_pois[:, 2] == 3)].tolist()
            mutated_pois = endpoints_pois + random.sample(potential_pois, k=pois_difference)
        else:
            mutated_pois = pois_existing_mutated

        # Mutate grow range
        mutation_strength_growth_dx = 1.5 * global_mutation_strength
        mutated_growth_iterator_limit = round(random.uniform(max(1, genotomut.growth_iterator_limit - mutation_strength_growth_dx), genotomut.growth_iterator_limit + mutation_strength_growth_dx))

        def __repr__(self) -> str:
            return str(f"pois={self.pois}\n shape={self.dungeon_shape}\n growth_limit={self.growth_iterator_limit}")
        
        ## ASSIGN
        self.pois = np.asarray(mutated_pois)
        self.number_of_pois = mutated_number_of_pois
        self.dungeon_shape = self.dungeon_shape
        self.growth_iterator_limit = mutated_growth_iterator_limit

    def __repr__(self) -> str:
        return str(f"pois={self.pois}, nbpois={self.number_of_pois}, growth={self.growth_iterator_limit}")
<a name="readme-top"></a>

<div align="center">

<h1 align="center">PCG02: Dungeon Evolution</h1>
*PCG02* explores procedural content generation for roguelike dungeons using evolutionary algorithms.

</div>

#### TOC
- [🔦 Project Summary](#-project-summary)
- [✨ Features Overview](#-features-overview)
- [🚀 Quick Start](#-quick-start)
- [📑 References](#-references)
- [Credits](#credits)

## 🔦 Project Summary
*PCG02* is a compact answer to the lab exercise of the third chapter of the [PCG Book](https://pcgbook.com/), entitled ["The search-based approach"](https://www.pcgbook.com/chapter02.pdf) and written by J.Togelius & N.Shaker.

### Exercise
> Roguelike games are a type of games that use PCG for level generation; in fact, the runtime generation and thereafter the infinite supply of levels is a key feature of this genre. As in the original game Rogue from 1980, a roguelike typically lets you control an agent in a labyrinthine dungeon, collecting treasures, fighting monsters and levelling up. A level in such a game thus consists of rooms of different sizes containing monsters and items and connected by corridors.

> The purpose of this exercise is to allow you to understand the search-based approach through implementing a search-based dungeon generator. Your generator should evolve playable dungeons for an imaginary roguelike. The phenotype of the dungeons should be 2D matrices (e.g. size 50 × 50) where each cell is one of the following: free space, wall, starting point, exit, monster, treasure. It is up to you whether to add other possible types of cell content, such as traps, teleporters, doors, keys, or different types of treasures and monsters. One of your tasks is to explore different content representations and quality measures in the context of dungeon generation.

### Implementation
The main challenge in PCG is always representation. This project defines a genotype as a set of points of interest (POIs) on the map. Dungeons are grown deterministically around these POIs, and circulation is ensured using a minimum spanning tree. This structure can easily be extended.

A basic genetic algorithm is implemented using the [μ + λ strategy](https://algorithmafternoon.com/strategies/mu_plus_lambda_evolution_strategy/), with adaptive mutation rates applied to: (a) The number, placement and types of POIs ; (b) Region-growing iterations ; (c) Circulation

Various fitness functions were tested, with a focus on producing longer dungeons featuring diverse rooms and pathways.

## ✨ Features Overview
- Evolutionary generation of dungeon maps (with Genotype encoding)
- Fast A* pathfinding and feature extraction
- Visualization for dungeon structure, walkability, and diversity
- Tools for measuring corridor/room ratios, room diversity, and completeness regarding

## 🚀 Quick Start
1. **Clone the repo**  
   `git clone https://github.com/macgrs/PCG02-dungeon_evolution.git`
2. **Install requirements**  
   `pip install -r requirements.txt`
3. **Run a notebook example**  
   Open and execute any Jupyter notebook in the repository.

## 📑 References
- [Procedural Content Generation in Games (PCG Book)](https://pcgbook.com/)
- [Evolving Playable Content](https://doi.org/10.1145/1234567)  
- [A* Pathfinding Algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)
- [RogueBasin: Map Generation](https://www.roguebasin.com/index.php/Map_Generation)
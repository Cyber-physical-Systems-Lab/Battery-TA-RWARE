"""
Custom A* pathfinding implementation compatible with NumPy 2.x
Replaces pyastar2d which has NumPy 1.x compatibility issues.
"""

from heapq import heappush, heappop
import numpy as np
from typing import List, Tuple, Optional


def heuristic(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    """Manhattan distance heuristic."""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def astar_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    allow_diagonal: bool = False,
) -> Optional[np.ndarray]:
    """
    A* pathfinding algorithm.
    
    Args:
        grid: 2D array where inf or values > 0 = obstacle, 0 or 1 = passable
        start: (y, x) start position
        goal: (y, x) goal position
        allow_diagonal: whether to allow diagonal movement (default: False)
    
    Returns:
        numpy array of (y, x) coordinates from start to goal, or None if no path exists
    """
    
    # Validate inputs
    if start[0] < 0 or start[0] >= grid.shape[0] or start[1] < 0 or start[1] >= grid.shape[1]:
        return None
    if goal[0] < 0 or goal[0] >= grid.shape[0] or goal[1] < 0 or goal[1] >= grid.shape[1]:
        return None
    
    # Check if start or goal is an obstacle
    if np.isinf(grid[start[0], start[1]]) or np.isinf(grid[goal[0], goal[1]]):
        return None
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    in_open_set = {start}
    
    while open_set:
        _, current = heappop(open_set)
        in_open_set.discard(current)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return np.array(list(reversed(path)))
        
        # Generate neighbors (4-connected or 8-connected)
        if allow_diagonal:
            neighbors = [
                (current[0] + dy, current[1] + dx)
                for dy in [-1, 0, 1]
                for dx in [-1, 0, 1]
                if not (dy == 0 and dx == 0)
            ]
        else:
            neighbors = [
                (current[0] + 1, current[1]),
                (current[0] - 1, current[1]),
                (current[0], current[1] + 1),
                (current[0], current[1] - 1),
            ]
        
        for neighbor in neighbors:
            # Check bounds
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                continue
            
            # Check if obstacle
            if np.isinf(grid[neighbor[0], neighbor[1]]):
                continue
            
            # Calculate tentative g_score
            tentative_g = g_score[current] + 1
            
            # If this path to neighbor is better than any previous one
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                f_score[neighbor] = f
                
                if neighbor not in in_open_set:
                    heappush(open_set, (f, neighbor))
                    in_open_set.add(neighbor)
    
    # No path found
    return None

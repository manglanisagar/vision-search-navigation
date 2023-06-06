import numpy as np
import heapq

def split_bev_into_grid(bev_image, grid_size):

  height, width = bev_image.shape
  a, b = grid_size
  
  cell_height = height // a
  cell_width = width // b
  
  # Reshape the BEV image into a grid-like structure
  grid_image = bev_image[:a * cell_height, :b * cell_width]
  grid_image = grid_image.reshape((a, cell_height, b, cell_width))
  
  # Calculate the average cost for each grid cell
  grid = np.mean(grid_image, axis=(1, 3))
  
  return grid

def get_neighbors(cell, rows, cols):
    row, col = cell
    neighbors = []
    
    if row > 0:
        neighbors.append((row - 1, col))
    if row < rows - 1:
        neighbors.append((row + 1, col)) 
    if col > 0:
        neighbors.append((row, col - 1))  
    if col < cols - 1:
        neighbors.append((row, col + 1))  
    
    return neighbors

def manhattan_distance(cell, destination):
    return abs(cell[0] - destination[0]) + abs(cell[1] - destination[1])

def uniform_cost_search(grid, source, destination):
    rows, cols = grid.shape
    
    cost_matrix = np.full_like(grid, fill_value=10**8)
    cost_matrix[source[0], source[1]] = 0
    
    queue = [(0, source)]
    
    parent_matrix = np.zeros_like(grid, dtype=np.ndarray)
    parent_matrix[source[0], source[1]] = None
    
    while queue:
        current_cost, current_cell = heapq.heappop(queue)
        
        if current_cell == destination:
            break
        
        neighbors = get_neighbors(current_cell, rows, cols)
        
        for neighbor in neighbors:
            neighbor_cost = current_cost + grid[neighbor[0], neighbor[1]]
            
            if neighbor_cost < cost_matrix[neighbor[0], neighbor[1]]:
                cost_matrix[neighbor[0], neighbor[1]] = neighbor_cost
                parent_matrix[neighbor[0], neighbor[1]] = current_cell
                heapq.heappush(queue, (neighbor_cost, neighbor))
    
    if cost_matrix[destination[0], destination[1]] == np.inf:
        return None
    
    path = []
    current_cell = destination
    
    while current_cell is not None:
        path.append(current_cell)
        current_cell = parent_matrix[current_cell[0], current_cell[1]]
    
    path.reverse()
    
    return path

def a_star_search(grid, source, destination):
    rows, cols = grid.shape
    
    cost_matrix = np.full_like(grid, fill_value=10**8)
    cost_matrix[source[0], source[1]] = 0
    
    queue = [(0, source)]
    
    parent_matrix = np.zeros_like(grid, dtype=np.ndarray)
    parent_matrix[source[0], source[1]] = None
    
    while queue:
        current_cost, current_cell = heapq.heappop(queue)
        
        if current_cell == destination:
            break
        
        neighbors = get_neighbors(current_cell, rows, cols)
        
        for neighbor in neighbors:
            neighbor_cost = current_cost + grid[neighbor[0], neighbor[1]]
            
            if neighbor_cost < cost_matrix[neighbor[0], neighbor[1]]:
                cost_matrix[neighbor[0], neighbor[1]] = neighbor_cost
                parent_matrix[neighbor[0], neighbor[1]] = current_cell
                total_cost = neighbor_cost + manhattan_distance(neighbor, destination)
                heapq.heappush(queue, (total_cost, neighbor))
    
    if cost_matrix[destination[0], destination[1]] == np.inf:
        return None
    
    path = []
    current_cell = destination
    
    while current_cell is not None:
        path.append(current_cell)
        current_cell = parent_matrix[current_cell[0], current_cell[1]]
    
    path.reverse()
    
    return path
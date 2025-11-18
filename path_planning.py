import numpy as np

def simple_rrt_cartesian(start, goal, n_waypoints=15, max_iterations=500, step_size=0.1):

    print("Running RRT path planning in Cartesian space...")
    
    # Simple obstacle check (you can extend this)
    def is_valid(point):
        # Check workspace bounds
        if point[2] < 0.1:  # NO going below z=0.1
            return False
        if np.linalg.norm(point[:2]) > 1.0:  # Stay within radius
            return False
        return True
    
    # RRT Tree
    tree = [start]
    parent = {0: None}
    
    for iteration in range(max_iterations):
        # Random sample (bias toward goal)
        if np.random.random() < 0.1:  # 10% goal bias
            random_point = goal
        else:
            random_point = np.array([
                np.random.uniform(-0.6, 0.6),
                np.random.uniform(-0.6, 0.6),
                np.random.uniform(0.1, 1.0)
            ])
        
        # Find nearest node in tree
        distances = [np.linalg.norm(node - random_point) for node in tree]
        nearest_idx = np.argmin(distances)
        nearest = tree[nearest_idx]
        
        # Extend toward random point
        direction = random_point - nearest
        distance = np.linalg.norm(direction)
        if distance > step_size:
            direction = direction / distance * step_size
        
        new_point = nearest + direction
        
        # Check if valid
        if is_valid(new_point):
            new_idx = len(tree)
            tree.append(new_point)
            parent[new_idx] = nearest_idx
            
            # Check if we reached goal
            if np.linalg.norm(new_point - goal) < step_size:
                # Reconstruct path
                path = [goal]
                current = new_idx
                while current is not None:
                    path.append(tree[current])
                    current = parent[current]
                path.reverse()
                
                # Resample to get desired number of waypoints
                path_array = np.array(path)
                t_original = np.linspace(0, 1, len(path))
                t_new = np.linspace(0, 1, n_waypoints)
                
                resampled_path = np.zeros((n_waypoints, 3))
                for dim in range(3):
                    resampled_path[:, dim] = np.interp(t_new, t_original, path_array[:, dim])
                
                print(f"  RRT found path in {iteration+1} iterations with {len(path)} nodes")
                return resampled_path
    
    print("  RRT failed to find path, using direct interpolation")
    return np.linspace(start, goal, n_waypoints)


def generate_cartesian_path(start, end, n_points, path_type="linear"):
    if path_type == "linear":
        print("Using linear path in Cartesian space")
        return np.linspace(start, end, n_points)
    
    elif path_type == "arc":
        print("Using smooth arc path (avoiding obstacles)")
        t = np.linspace(0, 1, n_points)
        linear_path = np.outer(1-t, start) + np.outer(t, end)
        mid_height = 0.2
        vertical_offset = 4 * mid_height * t * (1 - t)
        arc_path = linear_path.copy()
        arc_path[:, 2] += vertical_offset
        return arc_path
    
    elif path_type == "circular":
        print("Using circular path")
        center = (start + end) / 2
        radius = np.linalg.norm(end - start) / 2
        direction = end - start
        direction = direction / np.linalg.norm(direction)
        up = np.array([0, 0, 1])
        perp = np.cross(direction, up)
        if np.linalg.norm(perp) < 0.01:
            perp = np.array([1, 0, 0])
        perp = perp / np.linalg.norm(perp)
        angles = np.linspace(0, np.pi, n_points)
        path = []
        for angle in angles:
            offset = radius * (np.cos(angle) * direction + np.sin(angle) * perp)
            point = center + offset
            path.append(point)
        return np.array(path)
    
    elif path_type == "parabolic":
        print("Using parabolic path with lateral movement")
        t = np.linspace(0, 1, n_points)
        linear_path = np.outer(1-t, start) + np.outer(t, end)
        variation_y = 0.1 * np.sin(np.pi * t)
        variation_z = 0.15 * (4 * t * (1 - t))
        parabolic_path = linear_path.copy()
        parabolic_path[:, 1] += variation_y
        parabolic_path[:, 2] += variation_z
        return parabolic_path
    
    elif path_type == "rrt":
        return simple_rrt_cartesian(start, end, n_points)
    
    else:
        raise ValueError(f"Unknown path type: {path_type}")
# This is a sample Python script.
import numpy as np
import trimesh
import open3d as o3d

# original_pcd is one numpy array (Data of one .off file)
def visualize_point_cloud(filtered_array, original_array):
    # Create Open3D point cloud objects
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_array)

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_array)

    # Visualize point clouds
    # Visualize original point cloud
    o3d.visualization.draw_geometries([original_pcd], window_name='Original Point Cloud')

    # Visualize filtered point cloud
    o3d.visualization.draw_geometries([filtered_pcd], window_name='Filtered Point Cloud')


## SECOND PART (FUNCTIONS FOR GENERATING POINT CLOUD OUT OF MESH)
def read_OFF_mesh(file_path):
    """
    Read vertex information from an OFF file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    vertices = []
    faces = []
    for line in lines[2:]:
        data = list(map(float, line.strip().split()))
        if len(data) == 3:  # vertex
            vertices.append(data)
        elif len(data) == 4:  # face
            faces.append(data[1:])

    # Returns the vertices and faces as Python lists
    return vertices, faces


def generate_mesh(vertices, faces):
    """
    Generate a mesh from vertices and faces using trimesh.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def generate_point_cloud_from_mesh(file_path):
    """
    Generate point cloud data from a mesh.
    """
    # Read vertex information from the OFF file
    vertices, faces = read_OFF_mesh(file_path)

    # Number of points in the point cloud (Can be len(faces) * n for more dense point cloud)
    num_points = len(faces)

    # Generate a mesh from vertices
    mesh = generate_mesh(vertices, faces)

    # Generate point cloud data from the mesh
    point_cloud = mesh.sample(num_points)

    # Output the generated point cloud data from mesh and only the vertexes
    # visualize_point_cloud(np.array(point_cloud), np.array(vertices))
    return np.array(point_cloud)


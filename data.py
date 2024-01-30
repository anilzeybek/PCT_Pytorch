import os
import glob
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, cdist, squareform

# Skips the first 2 lines of the .off file and reads the rest
def read_off(off_path):
    with open(off_path, 'r') as f:
        if 'OFF' != f.readline().strip():
            return []

        n_verts, _, _ = tuple([int(s) for s in f.readline().strip().split(' ')])
        vertices = []
        for i in range(n_verts):
            vert = [float(s) for s in f.readline().strip().split(' ')]
            vertices.append(vert)

        return vertices

# Downloads the ModelNet dataset to the data folder and unzips it
def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'ModelNet10')):
        www = 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_labels = []
    for off_path in glob.glob(os.path.join(DATA_DIR, 'ModelNet10', f'*/{partition}/*.off')):
        pcd = read_off(off_path)
        all_data.append(np.array(pcd))
        all_labels.append(off_path.split('/')[-3])
    return all_data, all_labels

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

# original_pcd is one numpy array (Data of one .off file)
def filter_point_cloud(original_pcd):
    # Choose a random point
    random_point = np.random.choice(original_pcd, size=1, replace=False)
    radius = calculate_radius(original_pcd, 0.05)

    # Find points outside the sphere
    distances = cdist(original_pcd, random_point)
    points_outside_sphere = original_pcd[distances > radius]

    return points_outside_sphere, original_pcd

# original_pcd is one numpy array (Data of one .off file)
def visualize_point_cloud(original_pcd, filtered_array):
    import open3d as o3d
    # Create Open3D point cloud objects
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_pcd)

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_array)

    # Visualize point clouds
    o3d.visualization.draw_geometries([original_pcd, filtered_pcd])

# original_pcd is one numpy array (Data of one .off file)
# if distance_percentage is 0.05, it means the 5% of the maximum distance
def calculate_radius(original_pcd, distance_percentage):
    # Compute pairwise distances between points
    pairwise_distances = squareform(pdist(original_pcd))
    
    # Find the farthest two points
    max_distance = np.max(pairwise_distances)
    
    # Calculate radius as a percentage of the maximum distance
    radius = distance_percentage * max_distance
    
    return radius

class ModelNet10(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item: int):
        # TODO
        # 1) Point cloud'in icinden random bir point sec, ve onun etrafindaki n point'i yok et (belli bir radius)
        # 2) Return values: 1->kirpilmis point cloud       2->full point cloud
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    train = ModelNet10(1024)
    test = ModelNet10(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)

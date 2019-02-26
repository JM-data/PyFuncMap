import numpy as np

def read_off(file_dir):
    file = open(file_dir, 'r')
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    V = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    F = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]

    F = np.array(F)
    V = np.array(V)

    tmp = file_dir.split('/')
    tmp = tmp[-1]
    mesh_name = tmp.split('.')
    return V, F, mesh_name[0]

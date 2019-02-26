import MeshProcess
import MESH_IO
import numpy as np
from scipy.sparse.linalg import eigsh


class surface:
    def __init__(self, V, F):
        self.X = V[:,0]
        self.Y = V[:,1]
        self.Z = V[:,2]
        self.TRIV = F


class mesh:
    def __init__(self, V, F, mesh_name=''):
        self.surface = surface(V,F)
        self.VERT = V
        self.TRIV = F
        self.nv = V.shape[0]
        self.nf = F.shape[0]
        self.name = mesh_name
        self.W = None               # cotangent Laplacian matrix
        self.A = None               # area weight matrix (sparse!)
        self.evals = None           # LB operator - eigenfunctions
        self.evecs = None           # LB operator - eigenvalues
        self.normals_vtx = None     # per-vertex normal (unit length)
        self.normals_face = None    # per-face normals (unit length)

    def cotangent_laplacian(self):
        [self.W, self.A] = MeshProcess.cotangent_laplacian(self)
        return self

    def compute_laplacian_basis(self, numEigs=50):
        '''
        Generalized Eigen-decomposition of the Laplacian-Beltrami operator
        :param numEigs: the number of eigen-functions to be computed
        :return: the eigen-functions, and the corresponding eigen-values
        '''
        if self.W is None:
            self.cotangent_laplacian()
        evals, evecs = eigsh(self.W, numEigs, self.A, 1e-6)
        self.evecs = np.array(evecs, ndmin=2)
        self.evals = np.array(evals, ndmin=2).T
        return self

    def compute_vertex_and_face_normals(self):
        if self.normals_vtx is None:
            [self.normals_vtx, self.normals_face] = MeshProcess.compute_vertex_and_face_normals(self)
        return self

    def info(self):
        '''
        print the mesh information
        '''
        print('------------- MESH INFO --------------')
        if self.name != '':
            print('Mesh name: %s'%(self.name))
        print('The number of vertices: %d'%(self.nv))
        print('The number of faces: %d'%(self.nf))
        print('--------------------------------------')


def mesh_load_and_preprocess(mesh_dir, numEigs=50):
    [V, F, mesh_name] = MESH_IO.read_off(mesh_dir)
    S = mesh(V, F, mesh_name)
    S.compute_laplacian_basis(numEigs)
    S.compute_vertex_and_face_normals()
    return S

import numpy as np
import dolfinx as dlfx
from basix.ufl import element, mixed_element
import ufl
import source.helperFunc as hf
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
class Region:
    # filename
    meshInputFile: str

    # characteristic values
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    hmin: float
    hmax: float
    dim: float
    fdim: float

    # further instance variables:
    # facet_tags, region 
    # and functionspaces: SCAL, VEC, TEN, SOLspace (mixed element for u and s)

    def __init__(self, meshInputFile: str):
        """initializes Region (mesh) object: opens file, reads important values and defines functionspaces

        Parameters
        ----------
        meshInputFile : str
            mesh input filename
        """
        self.meshInputFile = meshInputFile

        # read mesh
        with dlfx.io.XDMFFile(comm, meshInputFile, 'r') as mesh_inp:
            self.region = mesh_inp.read_mesh()
            self.dim = self.region.topology.dim 
            self.fdim = self.dim-1
            self.region.topology.create_connectivity(self.fdim, self.dim)
            self.facet_tags = mesh_inp.read_meshtags(self.region, name='Facet tags')

        # find characteristic values
        all_elements = dlfx.mesh.locate_entities(self.region, self.dim, hf.all)
        self.coordsX = self.region.geometry.x[:,0]
        self.coordsY = self.region.geometry.x[:,1]
        self.hmax = comm.allreduce(np.max(self.region.h(self.dim, all_elements)), op=MPI.MAX)
        self.hmin = comm.allreduce(np.min(self.region.h(self.dim, all_elements)), op=MPI.MIN)
        self.xmin = comm.allreduce(np.min(self.coordsX), op=MPI.MIN)
        self.xmax = comm.allreduce(np.max(self.coordsX), op=MPI.MAX)
        self.ymin = comm.allreduce(np.min(self.coordsY), op=MPI.MIN)
        self.ymax = comm.allreduce(np.max(self.coordsY), op=MPI.MAX)
        self.left_cells = dlfx.mesh.locate_entities(self.region, self.dim, lambda x: x[0] <= (self.xmax- self.xmin)/2 + 1e-12)
        self.right_cells = dlfx.mesh.locate_entities(self.region, self.dim, lambda x: x[0] >= (self.xmax- self.xmin)/2 - 1e-12)

        # define functionspaces
        self.SCALe = element('Lagrange', self.region.basix_cell(), 1)
        self.SCAL = dlfx.fem.functionspace(self.region, self.SCALe)
        self.VECe = element("Lagrange", self.region.basix_cell(), 1, shape=(self.dim,)) 
        self.VEC = dlfx.fem.functionspace(self.region,self.VECe)
        self.TENe = element('DG', self.region.basix_cell(), 0, shape=(self.dim, self.dim, ))
        self.TEN = dlfx.fem.functionspace(self.region, self.TENe)
        self.SOLspace = dlfx.fem.functionspace(self.region, mixed_element([self.VECe,self.SCALe]))
        self.VEC2, self.VEC2Map = self.SOLspace.sub(0).collapse()
        self.SCAL2, self.SCAL2Map = self.SOLspace.sub(1).collapse()

        # define normal vector and integral on boundary
        self.nVec = ufl.FacetNormal(self.region)
        self.ds_BC = ufl.Measure("ds",domain=self.region, subdomain_data=self.facet_tags)  

        # define Tags
        self.inTag = 2
        self.botTag = 3
        self.outTag = 4
        self.lakeTag = 5
        self.leftBottomCorner = 6 #tags for benchmark
        self.rightBottomCorner = 7 #tags for benchmark 
        self.unevenbotTag = 9 #tag for weakly imposed dirichlet boundary condition

        # calculate max and min x coordinates of lake
        dofsLake = dlfx.fem.locate_dofs_topological(self.SCAL,self.fdim, self.facet_tags.find(self.lakeTag))
        xCoordsLake = self.SCAL.tabulate_dof_coordinates()[dofsLake][:,0]
        try: 
            lakeXMin = np.min(xCoordsLake)
        except:
            lakeXMin = np.inf
        try: 
            lakeXMax = np.max(xCoordsLake)
        except:
            lakeXMax = 0

        self.lakeXMin = comm.allreduce(lakeXMin, op=MPI.MIN)
        self.lakeXMax = comm.allreduce(lakeXMax, op=MPI.MAX)
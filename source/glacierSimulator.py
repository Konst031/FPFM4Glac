import dolfinx as dlfx

from mpi4py import MPI
from petsc4py import PETSc
from source.reg import Region
from source.boundaryCondition import Boundary
from source.validKeys import validKeysTime, validKeysGlacier

from basix.ufl import element, mixed_element
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import source.helperFunc as hf
import ufl 
import numpy as np
import sys
import matplotlib.pyplot as plt
from dataclasses import dataclass


# initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@dataclass
class TimeStepControl:
    """
        This class defines default parameters for the adaptive time stepping 
    """
    tend: float = 1       
    max_iters: int = 8
    min_iters: int = 4
    dt_scale_down: float = 0.5
    dt_scale_up: float = 1.5
    dt_max: float = 0.05 * tend            
    dt_start: float = 0.05 * tend

    def getInstanceVariables(self):
        return {var: getattr(self, var) for var in validKeysTime}
        
@dataclass
class GlacierSimulator:

    filename: str = 'default.xdmf'
    # material constants
    nu: float = 0.325                       # []
    rho_ice: float = 918.0                  # [kg/m^3]
    rho_water: float = 1000.0               # [kg/m^3]
    K_IC : float = 95.35 * 1e3              # [Pa m^(1/2)]    
    E: float = 1.0e9                        # [Pa]
    GcVal: float = K_IC**2/E                # [Pa*m]

    # physical constants
    g: float = 9.81                 # [m/s^2]
    
    # for mici
    freeBoard: float = 100          # [m]

    # numeric parameters
    M: float = 1.0                  # [m^3/Js]
    etaVal: float = 0.001
    maxSteps: int = np.inf
    l_0Val: float = 20.0                 

    # necessary objects
    mesh: Region = None
    bc: Boundary = None
    timeControl: TimeStepControl = None

    # checks
    simulationDone: bool = False
    solverSetup: bool = False
    crackDepthSetup: bool = False
    hasTopLeftCorner: bool = False

    def preSimulation(self):
        """
        Simulates purely elastic deformation according to boundary condition.
        Deformation then taken as initial condition for further crack simulation.
        (only applied when elDefoRight or elDefoBoth are chosen as BCs)
        """
        # setup BCs for elastic simulation of glacier deformation
        if self.bc.bcType == 'elDefoRight':
            self.bcPreSim = Boundary(mesh=self.mesh,bcType='preSimElDefoRight')
            self.bcPreSim.defineStrongBCs()
        if self.bc.bcType == 'elDefoBoth':
            self.bcPreSim = Boundary(mesh=self.mesh,bcType='preSimElDefoBoth')
            self.bcPreSim.defineStrongBCs()

        # weak form of continuity equation
        a = ufl.inner(hf.sigma(u=self.uPre, lam=self.lam, dim=self.mesh.dim, mu=self.mu), hf.eps(self.duPre)) * ufl.dx
        L = ufl.dot(self.f, self.duPre) * ufl.dx

        weakDirichlet = self.bc.penalty_factor*ufl.dot(self.duPre,self.mesh.nVec)*self.mesh.ds_BC(self.mesh.unevenbotTag)
        L = L + weakDirichlet

        if self.bc.waterOn == 1:
            L = L + ufl.dot(self.p_water * self.mesh.nVec, self.duPre) * self.mesh.ds_BC(self.mesh.lakeTag)


        self.preProblem =  dlfx.fem.petsc.LinearProblem(a, L, bcs=self.bcPreSim.bcs, 
                                                        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'})
        solution = self.preProblem.solve()

        self.bc.uout.x.array[:] = solution.x.array[0::solution.x.block_size]    # extract only x values
        if self.bc.bcType == 'elDefoBoth':
            self.bc.uin.x.array[:] = solution.x.array[0::solution.x.block_size] 

        # write result of pre simulation to xdmf (for debugging)
        # solution.name ='u-Pre'
        # with dlfx.io.XDMFFile(comm,f'{self.filename}PreSim.xdmf', 'w') as xdmfout:
        #     xdmfout.write_mesh(self.mesh.region)
        #     xdmfout.write_function(solution)     
    
    def simulate(self):
        """
            runs the whole simulation; runs all setup functions and solves numerical problem
        """
        # check if simulation is possible
        if  self.mesh == None or self.bc == None:
            print('Simulator must be setup prior to simulate')
            return
        if not(self.bc.bcValid):
            print('Invalid boundary conditions, unable to simulate')
            return
        
        # initialize necessary constants, functions, lists and stopwatch
        stopwatch = dlfx.common.Timer()
        stopwatch.start()        
        self.setupConstants()
        self.setupFunctions()
        self.setupCrackDepth()
        xdmfFilename = self.filename + '.xdmf'

        if self.bc.preSimNeeded:
            self.preSimulation()
        
        # Possible log levels: ERROR, INFO, OFF, WARNING
        dlfx.log.set_log_level(dlfx.log.LogLevel.WARNING)
        
        print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
        sys.stdout.flush()

        with dlfx.io.XDMFFile(comm, xdmfFilename, 'w') as xdmf_out:
            xdmf_out.write_mesh(self.mesh.region)

        # split into deformation-field (u) and phase-field (s)
        u, s = ufl.split(self.sol)
        uPrev, sPrev = ufl.split(self.solPrev)
        du, ds = ufl.split(self.dsol)

        # Mapping between DOFs of the mixed function space
        U, Umap = self.mesh.SOLspace.sub(0).collapse()
        S, Smap = self.mesh.SOLspace.sub(1).collapse()

        # define residuum to solve
        res = self.setupResiduum(u=u, s=s, du=du, ds=ds, sPrev=sPrev)

        # prepare expressions for further output
        self.setupExpressions(u=u, s=s)

        # define problem and solver
        self.nl_problem = dlfx.fem.petsc.NonlinearProblem(res, self.sol, self.bc.bcs)
        self.nl_solver = dlfx.nls.petsc.NewtonSolver(comm, self.nl_problem)
        self.solverSetup = True
        self.configureSolver()

        # initialize s=1, smin = 1
        self.solPrev.sub(1).x.array[:] = np.ones_like(self.solPrev.sub(1).x.array[:])
        self.solRestart.x.array[:] = self.solPrev.x.array[:]

        # initialize initial u
        uInit = np.zeros(self.mesh.coordsX.size)
        uInitNormalized = np.zeros(self.mesh.coordsX.size)
        if self.bc.preSimNeeded:
            uInit[:] = self.bc.uout.x.array[:]
            uInitMax = comm.allreduce(np.max(uInit), op=MPI.MAX)
            uInitNormalized[:] = uInit[:]/uInitMax

        # initialize time
        t = 0.0
        trestart = 0.0
        i = 0

        # computation loop
        while t <= self.timeControl.tend and i < self.maxSteps:

            # Stoppuhr fuer Zeitschritt
            stopwatch_step = stopwatch.elapsed()[0]
            # pull on glacier for the first half of the simulation
            if self.bc.bcType != 'mici':
                if not(self.bc.preSimNeeded):
                    if t <= 0.5 * self.timeControl.tend:
                        self.bc.uout.value = self.bc.u_right * (t/self.timeControl.tend) * 2 + self.bc.u_start
                        self.bc.uin.value = self.bc.u_left * (t/self.timeControl.tend) * 2
                    else:
                        self.bc.uout.value = self.bc.u_right + self.bc.u_start
                        self.bc.uin.value = self.bc.u_left
                else:
                    if t <= 0.5 * self.timeControl.tend:
                        self.bc.uout.x.array[:] = (t/self.timeControl.tend) * 2 * uInitNormalized[:] * self.bc.u_right
                        self.bc.uin.x.array[:] = (t/self.timeControl.tend) * 2 * uInitNormalized[:] * self.bc.u_left

            
            # information on current step
            if rank == 0:
                print('')
                print(f'Step {i+1}:')
                print('Computing solution at t = {0:.4e}'.format(t))
                print('Current time step dt = {0:.7e}'.format(self.dt.value))
                sys.stdout.flush()
        
            # adaptive time step control
            converged = False
            iters = 0
            try:
                (iters, converged) = self.nl_solver.solve(self.sol)
            except RuntimeError:
                self.dt.value = self.timeControl.dt_scale_down*self.dt.value
                
            if converged and iters < self.timeControl.min_iters and t > np.finfo(float).eps:
                self.dt.value = min(self.timeControl.dt_scale_up*self.dt.value, self.timeControl.dt_max)
                if rank == 0:
                    print('!FAST CONVERGENCE => dt = {0:.4e}'.format(self.dt.value))
                    sys.stdout.flush()
                
            # write new values into u, s and further expressions; write values to xdmf/h5
            if converged:
                # split mixed solution into deformation and phase-field parts
                u = self.sol.sub(0).collapse()
                s = self.sol.sub(1).collapse()
                u.name = 'u'
                s.name = 's'                
                # save previous values of s
                sPrev = self.solPrev.sub(1).collapse()

                # ensure that s >= 0 and that crack can't heal (= s increasing)
                s.x.array[:] = np.minimum(s.x.array[:], sPrev.x.array[:])
                s.x.array[:] = np.maximum(s.x.array[:], 0)

                # process solution and write to file
                self.interpolateExpressions()
                self.computeCrackDepth(s=s,u=u,t=t)
                self.writeFunctionsToXDMF(xdmfFilename,u,s,t)                
            
                # update solPrev and solRestart
                self.solPrev.x.array[Umap] = u.x.array[:] 
                self.solPrev.x.array[Smap] = s.x.array[:] 
                self.solRestart.x.array[:] = self.solPrev.x.array[:]
                trestart = t
                # make sure that the step at tend is actually computed but only once
                t = np.min([t+self.dt.value, self.timeControl.tend])
                t = self.timeControl.tend + 1 if t == trestart else t   
                i = i+1
            else:
                # set solution space back to restart solution (= previous steps solution)
                if rank == 0:
                    print('!NO CONVERGENCE  =>  dt = {0:.4e}'.format(self.dt.value))
                    sys.stdout.flush()
                # make sure that t = 0 is actually computed
                if i == 0:
                    t = trestart
                else:
                    t = trestart + self.dt.value
                
                self.sol.x.array[:] = self.solRestart.x.array[:]
                self.sol.x.scatter_forward()
                iters = 8

            # Write out information of current step
            if rank == 0:    
                print('No. of iterations:', iters)
                print('Converged:        ', converged)
                print('Runtime step/total {0:.2e}/{1:.2e}'.format(stopwatch.elapsed()[0]-stopwatch_step, stopwatch.elapsed()[0]))
                sys.stdout.flush()
            
        stopwatch.stop()
        self.simulationDone = True

################################################################################
###### Methods that set up simulation


    def configureSolver(self):
        """
        configures the solver with chosen options and prints tolerances
        """
        if not(self.solverSetup):
            print("Solver not setup yet. Can't configure solver")
        self.nl_solver.max_it = self.timeControl.max_iters
        self.nl_solver.convergence_criterion = 'incremental'
        lin_solver = self.nl_solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = lin_solver.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly" 
        opts[f"{option_prefix}pc_type"] = "lu" 
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        lin_solver.setFromOptions()
        print("Absolute tolerance (atol):", self.nl_solver.atol)
        print("Relative tolerance (rtol):", self.nl_solver.rtol)

    def setupConstants(self):
        """
        sets up all necessary fem.constants for simulation
        """
        if self.mesh == None:
            print('No mesh found, please enter a mesh first!')
            return
        reg = self.mesh.region
        muVal = self.E/(2*(1+self.nu))                          # [Pa]
        lamVal = (self.E*self.nu)/((1+self.nu)*(1-2*self.nu))   # [Pa]
        KVal = lamVal + ((2/self.mesh.dim) * muVal)             # [Pa]

        self.dt = dlfx.fem.Constant(reg, self.timeControl.dt_start)
        # lamé constants mu and lambda
        self.lam = dlfx.fem.Constant(reg, lamVal)               # lamé's first constant
        self.mu = dlfx.fem.Constant(reg, muVal)                 # lamé's second constant       
        self.K = dlfx.fem.Constant(reg, KVal)                   # bulk modulus
        self.fConst = dlfx.fem.Constant(reg, PETSc.ScalarType([0.0, - self.rho_ice * self.g])) # body force due to gravity [N/m^3]
        self.nuConst = dlfx.fem.Constant(reg,self.nu)           # Poisson's ratio
        self.EConst = dlfx.fem.Constant(reg, self.E)            # Young's modulus
        self.G_c = dlfx.fem.Constant(reg, self.GcVal)           # critical energy release rate
        self.M_inv = dlfx.fem.Constant(reg, 1/self.M)           # inverse of mobility constant
        self.eta = dlfx.fem.Constant(reg, self.etaVal)          # Steifigkeitsreduktion
        self.l_0 = dlfx.fem.Constant(reg, self.l_0Val)          # regularistion parameter    

    def setupFunctions(self):
        """
        sets up all necessary fem.functions for simulation
        """
        if self.mesh == None:
            RuntimeError('No mesh found, please enter a mesh first!')
        if self.bc == None:
            RuntimeError('No boundary condition defined')
        
        # solution space and test function
        self.sol =  dlfx.fem.Function(self.mesh.SOLspace)
        self.solPrev =  dlfx.fem.Function(self.mesh.SOLspace)
        self.solRestart =  dlfx.fem.Function(self.mesh.SOLspace)
        self.dsol = ufl.TestFunction(self.mesh.SOLspace)

        # functions needed for pre Simulation
        if self.bc.preSimNeeded:
            self.uPre = ufl.TrialFunction(self.mesh.VEC2)
            self.duPre = ufl.TestFunction(self.mesh.VEC2)

        # functions for nicer output
        self.stress = dlfx.fem.Function(self.mesh.TEN)
        self.strain = dlfx.fem.Function(self.mesh.TEN)
        self.energy_P_Lo = dlfx.fem.Function(self.mesh.SCAL)
        self.energy_M_Lo = dlfx.fem.Function(self.mesh.SCAL)
        self.strain_dev = dlfx.fem.Function(self.mesh.TEN)
        self.strain_vol = dlfx.fem.Function(self.mesh.TEN)


        # functions needed for further calculation
        # definition of approximate rectangular function for smooth transition of pressure function
        slope = 0.05
        maxDeviationAtLake = 0.001
        n = np.log(maxDeviationAtLake)/(slope*self.mesh.xmax)
        rectangularFunction = 1/(1+np.exp(n*(self.mesh.coordsX[:]-self.mesh.lakeXMin+slope*self.mesh.xmax))) - \
            1/(1+np.exp(n*(self.mesh.coordsX[:]-self.mesh.lakeXMax-slope*self.mesh.xmax)))
        # define water pressure over mesh area (for water inside of cracks under lake)
        self.p_water = dlfx.fem.Function(self.mesh.SCAL2)
        self.p_water.x.array[:] = self.rho_water * self.g * (self.mesh.ymax - self.mesh.coordsY[:]) * rectangularFunction[:]      
        self.p_water.name = "p_water"

        # water pressure for mici
        x = ufl.SpatialCoordinate(self.mesh.region)
        self.p_front = self.rho_water * self.g * ufl.conditional(ufl.le(x[1],self.mesh.ymax - self.freeBoard), x[1] - (self.mesh.ymax - self.freeBoard), 0)
        
        # body force vector
        self.f = dlfx.fem.Function(self.mesh.VEC2)
        self.f.interpolate(dlfx.fem.Expression(self.fConst, self.mesh.VEC2.element.interpolation_points()))
        self.f.name = "bodyForce"

    def setupExpressions(self, u, s):
        """sets up expressions for visualization

        Parameters
        ----------
        u : form
            deformation function
        s : form
            s-field
        """
        self.stress_expr = dlfx.fem.Expression(
            hf.stress(eps=hf.eps(u), s=s, eta=self.eta, nu=self.nu, E=self.E, lam=self.lam, mu=self.mu),
            self.mesh.TEN.element.interpolation_points())
        self.strain_expr = dlfx.fem.Expression(hf.eps(u), self.mesh.TEN.element.interpolation_points())
        self.energy_M_Lo_expr = dlfx.fem.Expression(
            hf.psielM_Lo(eps=hf.eps(u), lam=self.lam, mu=self.mu, nu=self.nuConst, E=self.EConst), 
            self.mesh.SCAL.element.interpolation_points())
        self.energy_P_Lo_expr = dlfx.fem.Expression(
            hf.psielP_Lo(eps=hf.eps(u), lam=self.lam, mu=self.mu, nu=self.nuConst, E=self.EConst), 
            self.mesh.SCAL.element.interpolation_points())
        self.stress.name = 'stress'
        self.strain.name = 'strain'
        self.energy_M_Lo.name = 'energy_M_Lo'
        self.energy_P_Lo.name = 'energy_P_Lo'

    def setupCrackDepth(self):
        """creates lists for crack data and finds dofs of cells to read values from
        """
        if self.mesh == None:
            print('No mesh found, please enter a mesh first!')
            return
        # initialize lists
        if rank == 0:
            self.crackDepthList = []
            self.crackDepthUnderLakeList = []
            self.stressLeftMeanList = []
            self.stressLeftMedianList = []
            self.meanDisplacementList = []
            self.highestXCrackList = []
            self.timeStepList = []
        
        # get x and y coordinates in same order as Dofs
        self.xCoordsSCAL = self.mesh.SCAL2.tabulate_dof_coordinates()[:,0]
        self.yCoordsSCAL = self.mesh.SCAL2.tabulate_dof_coordinates()[:,1]

        # locate dofs
        self.displacementDofsLeft = dlfx.fem.locate_dofs_topological(self.mesh.VEC2, self.mesh.fdim, self.mesh.facet_tags.find(self.mesh.inTag))
        self.displacementDofsRight = dlfx.fem.locate_dofs_topological(self.mesh.VEC2, self.mesh.fdim, self.mesh.facet_tags.find(self.mesh.outTag))
        
        # find Dofs close to left edge (close means closer than 2*the lowest x value of a triangles coordinate)
        dofCoordsTenX = self.mesh.TEN.tabulate_dof_coordinates()[:,0]
        self.stressDofsLeft = np.argwhere(dofCoordsTenX < (dofCoordsTenX.min() * 2))[:,0]

        self.crackDepthSetup = True

    def setupResiduum(self, u, s, du, ds, sPrev):
        """sets up variational form

        Parameters
        ----------
        u : form
            deformation field
        s : form
            s-field (phase field parameter)
        du : test function
            test function related to deformation field
        ds : test function
            test function related to phase field parameter field
        sPrev : form
            s-field of previous time step

        Returns
        -------
        form
            mathematical problem
        """
            
        # problem using Lo's scheme:
        pot = (hf.degrad(s,self.eta)*hf.psielP_Lo(hf.eps(u),self.lam, self.mu, self.nuConst, self.EConst) +
                hf.psielM_Lo(hf.eps(u),self.lam, self.mu, self.nuConst, self.EConst) + hf.psifrac(s, self.G_c, self.l_0) - hf.psigrav(u, self.f))*ufl.dx

        weakDirichlet, weakNeumann = self.bc.defineWeakBCs(u,s,du,self.p_front,self.p_water,self.eta)

        equi = ufl.derivative(pot,u,du)
        sdrive = ufl.derivative(pot, s, ds)
        rate = self.M_inv*(s-sPrev)/self.dt*ds*ufl.dx
        res = rate+sdrive+equi+weakDirichlet + ufl.derivative(weakNeumann, u, du)

        return res

################################################################################
###### Methods that are necessary during simulation

    def interpolateExpressions(self):
        """calculates output data (stress, strain...) and interpolates it to functions
        """
        self.stress.interpolate(self.stress_expr)
        self.strain.interpolate(self.strain_expr)
        self.energy_M_Lo.interpolate(self.energy_M_Lo_expr)
        self.energy_P_Lo.interpolate(self.energy_P_Lo_expr)
                                                        
    def computeCrackDepth(self, s, u, t):
        """computes crack depth as difference of lowest and highest s parameter within threshhold

        Parameters
        ----------
        s : form
            phase field parameter field
        u : form
            displacement field
        """
        if not(self.crackDepthSetup):
            print('Please set up Crack depth before')
            return
        
        ## compute Crack depth ##
        # find all Y coords where s has value < 2*eta
        mask_crack = np.where(s.x.array < 2*self.eta.value, True, False)
        mask_lake = np.where((self.xCoordsSCAL >= self.mesh.lakeXMin) & (self.xCoordsSCAL <= self.mesh.lakeXMax), True, False)
        mask_crackUnderLake = mask_crack & mask_lake

        crackCoordsX = self.xCoordsSCAL[mask_crack]
        crackCoordsY = self.yCoordsSCAL[mask_crack]
        underLakeCrackCoordsY = self.yCoordsSCAL[mask_crackUnderLake]

        # find max and min Y and max X of crack
        if(crackCoordsY.size > 0):
            # necessary as crackCoordsY could be empty -> error from np.min/max
            crackMinY = np.min(crackCoordsY)
            crackMaxY = np.max(crackCoordsY)
            crackMaxX = np.max(crackCoordsX)
        else:
            crackMinY = np.inf
            crackMaxY = -np.inf
            crackMaxX = 0

        crackMinY = comm.allreduce(crackMinY, op=MPI.MIN)
        crackMaxY = comm.allreduce(crackMaxY, op=MPI.MAX)
        crackMaxX = comm.allreduce(crackMaxX, op=MPI.MAX)

        # compute crackDepth (keeping in mind that crackMinY/MaxY could be inf)
        if np.isinf(crackMinY) or np.isinf(crackMaxY):
            crackDepth = 0
        else:
            crackDepth = crackMaxY - crackMinY 

        # same for crack under lake
        if(underLakeCrackCoordsY.size > 0):
            # necessary as crackCoordsY could be empty -> error from np.min/max
            underLakeCrackMinY = np.min(underLakeCrackCoordsY)
            underLakeCrackMaxY = np.max(underLakeCrackCoordsY)
        else:
            underLakeCrackMinY = np.inf
            underLakeCrackMaxY = -np.inf

        underLakeCrackMinY = comm.allreduce(underLakeCrackMinY, op=MPI.MIN)
        underLakeCrackMaxY = comm.allreduce(underLakeCrackMaxY, op=MPI.MAX)

        # compute crackDepth (keeping in mind that crackMinY/MaxY could be inf)
        if np.isinf(underLakeCrackMinY) or np.isinf(underLakeCrackMaxY):
            underLakeCrackDepth = 0
        else:
            underLakeCrackDepth = underLakeCrackMaxY - underLakeCrackMinY 
            

        # read displacement and stress        
        xDisplacementLeft = []
        xDisplacementRight = []
        stressXLeft = []

        if self.displacementDofsLeft.size != 0:
            xDisplacementLeft = u.x.array[2*self.displacementDofsLeft]

        if self.displacementDofsRight.size != 0:
            xDisplacementRight = u.x.array[2*self.displacementDofsRight]

        if self.stressDofsLeft.size != 0:
            stressXLeft = self.stress.x.array[4*self.stressDofsLeft]

        # gather values onto root process
        xDisplacementLeftTot = comm.gather(xDisplacementLeft, root=0)
        xDisplacementRightTot = comm.gather(xDisplacementRight, root=0)
        stressXLeftTot = comm.gather(stressXLeft, root=0)


        ## write computed/read values to Lists
        if rank == 0:   
            # get array of numbers not array of arrays
            xDisplacementLeftTot = np.concatenate(xDisplacementLeftTot)
            xDisplacementRightTot = np.concatenate(xDisplacementRightTot)
            stressXLeftTot = np.concatenate(stressXLeftTot)
            
            self.timeStepList.append(t)
            self.highestXCrackList.append(crackMaxX)
            self.crackDepthList.append(crackDepth)
            self.crackDepthUnderLakeList.append(underLakeCrackDepth)  
            self.stressLeftMeanList.append(np.mean(stressXLeftTot))
            self.stressLeftMedianList.append(np.median(stressXLeftTot))
            meanDispRight = np.mean(xDisplacementRightTot)
            meanDispLeft = np.mean(xDisplacementLeftTot)
            self.meanDisplacementList.append(meanDispRight - meanDispLeft)
            print(f"Risstiefe: {crackDepth}")
            
    def writeFunctionsToXDMF(self, xdmfFilename,u,s,t):
        """writes functions calculated at a timestep to xdmf

        Parameters
        ----------
        xdmfFilename : string
            name of output xdmf file
        u : form
            displacement field
        s : form
            phase field parameter field
        t : float
            time where functions are evaluated at
        """
        with dlfx.io.XDMFFile(comm, xdmfFilename, 'a') as xdmfout:
                xdmfout.write_function(u, t)
                xdmfout.write_function(s, t)
                xdmfout.write_function(self.stress, t)
                xdmfout.write_function(self.strain, t)
                xdmfout.write_function(self.energy_P_Lo, t)
                xdmfout.write_function(self.energy_M_Lo, t)
                xdmfout.write_function(self.p_water, t)

################################################################################
###### Methods for end of simulation
        
    def plotCrackDepthInto(self, plotNum: int):
        """plots computed crack depth against mean deformation

        Parameters
        ----------
        plotNum : int
            number of matplotlib figure to plot into
        """
        if not(self.simulationDone):
            print('Simulation has not run yet. Unable to plot anything')
            return
        if rank == 0:
            plt.figure(plotNum)
            plt.plot(self.meanDisplacementList, self.crackDepthList)  

    def saveCrackDepth(self):
        """saves computed crack depth and stress/deformation values as numpy arrays (.npy)
        """
        if not(self.simulationDone):
            print('Simulation has not run yet. Unable to plot anything')
            return
        if rank == 0:
            np.save(self.filename + 'meanStress', self.stressLeftMeanList)
            np.save(self.filename + 'medianStress', self.stressLeftMedianList)
            np.save(self.filename + 'crackDepth', self.crackDepthList)
            np.save(self.filename + 'meanDisp', self.meanDisplacementList)
            np.save(self.filename + 'crackUnderLake', self.crackDepthUnderLakeList)
            np.save(self.filename + 'highestXCrack', self.highestXCrackList)
            np.save(self.filename + 'time', self.timeStepList)

    def getInstanceVariables(self):
        """reads all used parameters for simulation (of all classes)

        Returns
        -------
        dict
            dictionary with all parameter values used in simulation
        """
        dictTime = self.timeControl.getInstanceVariables()
        dictBoundary = self.bc.getInstanceVariables()
        dictGlacier = {var: getattr(self, var) for var in validKeysGlacier}
        return {**dictTime, **dictBoundary, **dictGlacier}
        
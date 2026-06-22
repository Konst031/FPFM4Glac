import dolfinx as dlfx
import numpy as np
import ufl
from petsc4py import PETSc
import source.helperFunc as hf
from source.validKeys import validKeysBoundary

from source.reg import Region      
from dataclasses import dataclass
@dataclass
class Boundary:
    mesh: Region = None
    u_right: float = 0.0
    u_left: float = 0.0
    u_start: float = 0.0
    bcType: int = None
    preSimNeeded: bool = False
    bcValid: bool = False

    waterOn: int = 0                    # 0 means off, 1 means on
    tau_b: float = 0.0                  # [Pa]
    load_benchmark : float = 0.0        # [Pa]

    penalty_factor: float = 1e10        # for weakly imposed Dirichlet Boundary condition

    tauAntisymmetric: int = 0


    # further instance variable:
    # bcs: list of all boundary conditions
    # uout: for bcType uniform pull: fem.Constant for u

    def defineStrongBCs(self):
        """creates dirichletBCs according to bcType using dolfinx built in dirichletbc method
        """
        if self.mesh == None:
            print('No mesh found, please enter a mesh first!')
            return
        match self.bcType:
            case 'uniformPull':     
                # left: pushing in x with u_left, right: pulling in x with u_right, bottom: no u in y
                fdim = self.mesh.fdim
                self.uout = dlfx.fem.Constant(self.mesh.region, self.u_right)
                self.uin = dlfx.fem.Constant(self.mesh.region, self.u_left)
                indofs_x = dlfx.fem.locate_dofs_topological(self.mesh.SOLspace.sub(0).sub(0), fdim, self.mesh.facet_tags.find(self.mesh.inTag))
                outdofs_x = dlfx.fem.locate_dofs_topological(self.mesh.SOLspace.sub(0).sub(0), fdim, self.mesh.facet_tags.find(self.mesh.outTag))
                botdofs_y = dlfx.fem.locate_dofs_topological(self.mesh.SOLspace.sub(0).sub(1), fdim, self.mesh.facet_tags.find(self.mesh.botTag))

                inBC_x = dlfx.fem.dirichletbc(self.uin, indofs_x, self.mesh.SOLspace.sub(0).sub(0))
                outBC_x = dlfx.fem.dirichletbc(self.uout, outdofs_x, self.mesh.SOLspace.sub(0).sub(0))
                botBC_y = dlfx.fem.dirichletbc(0.0, botdofs_y, self.mesh.SOLspace.sub(0).sub(1))

                self.bcs = [inBC_x, outBC_x, botBC_y]

            case 'elDefoRight':     
                # left: pushing in x, right: pulling in x with prior elastic deformation on right boundary, bottom: no u in y
                fdim = self.mesh.fdim
                Q = self.mesh.SOLspace.sub(0).collapse()[0]
                Qx = Q.sub(0).collapse()[0]
                self.uout = dlfx.fem.Function(Qx)
                self.uin = dlfx.fem.Function(Qx)
                indofs_x = dlfx.fem.locate_dofs_topological((self.mesh.SOLspace.sub(0).sub(0), Qx), fdim, self.mesh.facet_tags.find(self.mesh.inTag))
                outdofs_x = dlfx.fem.locate_dofs_topological((self.mesh.SOLspace.sub(0).sub(0), Qx), fdim, self.mesh.facet_tags.find(self.mesh.outTag))
                botdofs_y = dlfx.fem.locate_dofs_topological(self.mesh.SOLspace.sub(0).sub(1), fdim, self.mesh.facet_tags.find(self.mesh.botTag))

                inBC_x = dlfx.fem.dirichletbc(self.uin, indofs_x, self.mesh.SOLspace.sub(0).sub(0))
                outBC_x = dlfx.fem.dirichletbc(self.uout, outdofs_x, self.mesh.SOLspace.sub(0).sub(0))
                botBC_y = dlfx.fem.dirichletbc(0.0, botdofs_y, self.mesh.SOLspace.sub(0).sub(1))
                self.bcs = [inBC_x, outBC_x, botBC_y]
                self.preSimNeeded = True

            case 'preSimElDefoRight':     
                # helper for elastic deformation right (-> vector-valued function space): 
                # left no u in x, right: nothing, bottom: no u in y
                fdim = self.mesh.fdim
                indofs_x = dlfx.fem.locate_dofs_topological(self.mesh.VEC2.sub(0), fdim, self.mesh.facet_tags.find(self.mesh.inTag))
                botdofs_y = dlfx.fem.locate_dofs_topological(self.mesh.VEC2.sub(1), fdim, self.mesh.facet_tags.find(self.mesh.botTag))

                inBC_x = dlfx.fem.dirichletbc(0.0, indofs_x, self.mesh.VEC2.sub(0))
                botBC_y = dlfx.fem.dirichletbc(0.0, botdofs_y, self.mesh.VEC2.sub(1))

                self.bcs = [inBC_x, botBC_y]

            case 'elDefoBoth':     
                # left: no u in x, right: pulling in x with prior elastic deformation right and left, bottom: no u in y
                fdim = self.mesh.fdim
                Q = self.mesh.SOLspace.sub(0).collapse()[0]
                Qx = Q.sub(0).collapse()[0]
                self.uin = dlfx.fem.Function(Qx)
                self.uout = dlfx.fem.Function(Qx)
                indofs_x = dlfx.fem.locate_dofs_topological((self.mesh.SOLspace.sub(0).sub(0), Qx), fdim, self.mesh.facet_tags.find(self.mesh.inTag))
                outdofs_x = dlfx.fem.locate_dofs_topological((self.mesh.SOLspace.sub(0).sub(0), Qx), fdim, self.mesh.facet_tags.find(self.mesh.outTag))
                botdofs_y = dlfx.fem.locate_dofs_topological(self.mesh.SOLspace.sub(0).sub(1), fdim, self.mesh.facet_tags.find(self.mesh.botTag))

                inBC_x = dlfx.fem.dirichletbc(self.uin, indofs_x, self.mesh.SOLspace.sub(0).sub(0))
                outBC_x = dlfx.fem.dirichletbc(self.uout, outdofs_x, self.mesh.SOLspace.sub(0).sub(0))
                botBC_y = dlfx.fem.dirichletbc(0.0, botdofs_y, self.mesh.SOLspace.sub(0).sub(1))
                self.bcs = [inBC_x, outBC_x, botBC_y]
                self.preSimNeeded = True

            case 'preSimElDefoBoth':    
                # helper for elastic deformation both sides (-> vector-valued function space): 
                # left nothing, right: nothing, bottom: no u in y, no u in x at center point
                fdim = self.mesh.fdim
                botdofs_y = dlfx.fem.locate_dofs_topological(self.mesh.VEC2.sub(1), fdim, self.mesh.facet_tags.find(self.mesh.botTag))
                bot_middle_point_x = dlfx.mesh.locate_entities_boundary(self.mesh.region, fdim-1, marker=lambda x: np.logical_and(np.isclose(x[0], self.mesh.xmax * 0.5),np.isclose(x[1], 0)))
                botdofs_middle_point_x = dlfx.fem.locate_dofs_topological(self.mesh.VEC2.sub(0), fdim-1, bot_middle_point_x)

                botBC_x = dlfx.fem.dirichletbc(0.0, botdofs_middle_point_x, self.mesh.VEC2.sub(0))
                botBC_y = dlfx.fem.dirichletbc(0.0, botdofs_y, self.mesh.VEC2.sub(1))

                self.bcs = [botBC_x, botBC_y]

            case 'benchmark':
                fdim = self.mesh.fdim

                self.uout = dlfx.fem.Constant(self.mesh.region, self.u_right)
                self.uin = dlfx.fem.Constant(self.mesh.region, self.u_left)

                #leftBottomCorner_dofs_x = dlfx.fem.locate_dofs_topological(self.mesh.SOLspace.sub(0).sub(0), fdim-1, self.mesh.facet_tags.find(self.mesh.leftBottomCorner))
                #leftBottomCorner_dofs_y = dlfx.fem.locate_dofs_topological(self.mesh.SOLspace.sub(0).sub(1), fdim-1, self.mesh.facet_tags.find(self.mesh.leftBottomCorner))
                #rightBottomCorner_dofs_y = dlfx.fem.locate_dofs_topological(self.mesh.SOLspace.sub(0).sub(1), fdim-1, self.mesh.facet_tags.find(self.mesh.rightBottomCorner))

                leftBottomCorner = dlfx.mesh.locate_entities_boundary(self.mesh.region, fdim-1, marker=lambda x: np.logical_and(np.isclose(x[0], self.mesh.xmin),np.isclose(x[1], self.mesh.ymin)))
                rightBottomCorner = dlfx.mesh.locate_entities_boundary(self.mesh.region, fdim-1, marker=lambda x: np.logical_and(np.isclose(x[0], self.mesh.xmax),np.isclose(x[1], self.mesh.ymin)))

                leftBottomCorner_dofs_x = dlfx.fem.locate_dofs_topological(self.mesh.SOLspace.sub(0).sub(0), fdim-1, leftBottomCorner)
                leftBottomCorner_dofs_y = dlfx.fem.locate_dofs_topological(self.mesh.SOLspace.sub(0).sub(1), fdim-1, leftBottomCorner)
                rightBottomCorner_dofs_y = dlfx.fem.locate_dofs_topological(self.mesh.SOLspace.sub(0).sub(1), fdim-1, rightBottomCorner)

                botBC_x_leftCorner = dlfx.fem.dirichletbc(0.0, leftBottomCorner_dofs_x, self.mesh.SOLspace.sub(0).sub(0))
                botBC_y_leftCorner = dlfx.fem.dirichletbc(0.0, leftBottomCorner_dofs_y, self.mesh.SOLspace.sub(0).sub(1))
                botBC_y_rightCorner = dlfx.fem.dirichletbc(0.0, rightBottomCorner_dofs_y, self.mesh.SOLspace.sub(0).sub(1))

                self.bcs = [botBC_x_leftCorner, botBC_y_leftCorner, botBC_y_rightCorner]

            case 'mici':
                # left: no u in x, right: 
                fdim = self.mesh.fdim
                indofs_x = dlfx.fem.locate_dofs_topological(self.mesh.SOLspace.sub(0).sub(0), fdim, self.mesh.facet_tags.find(self.mesh.inTag))
                botdofs_y = dlfx.fem.locate_dofs_topological(self.mesh.SOLspace.sub(0).sub(1), fdim, self.mesh.facet_tags.find(self.mesh.botTag))

                inBC_x = dlfx.fem.dirichletbc(0.0, indofs_x, self.mesh.SOLspace.sub(0).sub(0))
                botBC_y = dlfx.fem.dirichletbc(0.0, botdofs_y, self.mesh.SOLspace.sub(0).sub(1))

                self.bcs = [inBC_x, botBC_y]

            case _:     # default
                print('Invalid Boundary condition type')
                self.bcValid = False
                return
            
        self.bcValid = True

    def defineWeakBCs(self, u, s, du, p_front, p_water, eta):
        """defines dirichlet bcs that can't be added via dolfinx built in function and von Neumann bcs in weak form

        Parameters
        ----------
        u : form
            displacement field
        s : form
            phasefield parameter field
        du : test function
            test function related to displacement
        p_front : _type_
            pressure function needed for mici calculation
        p_water : _type_
            pressure function needed for water pressure in lake/cracks
        eta : float
            small numerical value for degradation function

        Returns
        -------
        form
            weak boundary conditions (need to be differentiated w.r.t. u)
        """
        weakDirichlet = self.penalty_factor*ufl.dot(self.mesh.nVec,u)*ufl.dot(du,self.mesh.nVec)*self.mesh.ds_BC(self.mesh.unevenbotTag)
        
        # applicaple weak Neumann condititions added to weak Neumann step by step
        weakNeumann = 0 * self.mesh.ds_BC(self.mesh.inTag) # this is a zero valued ufl form -> can add to it

        # add shear stress to ground
        if self.tau_b != 0:
            self.tauVec = dlfx.fem.Constant(self.mesh.region, PETSc.ScalarType([-self.tau_b, 0]))
            self.posTauVec = dlfx.fem.Constant(self.mesh.region, PETSc.ScalarType([self.tau_b, 0]))
            self.tauFunc = dlfx.fem.Function(self.mesh.VEC2)

            if self.tauAntisymmetric == 0:
                self.tauFunc.interpolate(dlfx.fem.Expression(self.tauVec, self.mesh.VEC.element.interpolation_points()))
            else:
                self.tauFunc.interpolate(dlfx.fem.Expression(self.tauVec, self.mesh.VEC.element.interpolation_points()), self.mesh.right_cells)
                self.tauFunc.interpolate(dlfx.fem.Expression(self.posTauVec, self.mesh.VEC.element.interpolation_points()), self.mesh.left_cells)
                self.tauFunc.x.scatter_forward()

            weakNeumann = weakNeumann + ufl.dot(self.tauFunc, u) * self.mesh.ds_BC(self.mesh.botTag) + ufl.dot(self.tauFunc, u) * self.mesh.ds_BC(self.mesh.unevenbotTag)
        
        match self.bcType:
            case 'mici':
                # add water pressure to front   
                weakNeumann = weakNeumann - ufl.dot(p_front * self.mesh.nVec, u) * self.mesh.ds_BC(self.mesh.outTag)
            case 'benchmark':
                load_left = ufl.dot(self.load_benchmark * self.mesh.nVec, u) * self.mesh.ds_BC(self.mesh.inTag)
                load_right = ufl.dot(self.load_benchmark * self.mesh.nVec, u) * self.mesh.ds_BC(self.mesh.outTag)
                weakNeumann = weakNeumann - load_left - load_right
            case _:
                if self.waterOn == 1:
                    # impose water pressure to crack and lake
                    waterTerms = hf.degrad(s,eta)*ufl.dot(p_water * self.mesh.nVec, u) * self.mesh.ds_BC(self.mesh.lakeTag) \
                    + hf.degrad(s,eta)*hf.psiwater(u,p_water)*ufl.dx \
                    - hf.degrad(s,eta)*ufl.dot(p_water * self.mesh.nVec, u) * self.mesh.ds_BC
                    weakNeumann = weakNeumann + waterTerms
        
        return weakDirichlet, weakNeumann

    def getInstanceVariables(self):
        return {var: getattr(self, var) for var in validKeysBoundary}
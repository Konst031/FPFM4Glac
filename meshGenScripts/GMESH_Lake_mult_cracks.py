import numpy as np
import dolfinx as dlfx
import gmsh
from mpi4py import MPI
import sys
import csv


mesh_name = "../mesh/00_Test"
gdim = 2 
h_size = 13 # denser mesh if h_size is smaller


gmsh.initialize()
gmsh.clear()
gmsh.model.add(mesh_name)

# Geometry
L = 4000            # length of mesh
H = 1000            # height of mesh
S = 1500            # Distance from leftmost point of lake to left edge
h = 50              # Depth of lake
B = L-2*S           # width of lake (for definition of crack position)

# ----------- Points -----------

point1 = gmsh.model.occ.add_point(x=0, y=0, z=0) 
# ensure that there is a point in the middle of the bedrock. Handy for some BCs
point1a = gmsh.model.occ.add_point(x=L/2, y=0, z=0)
point2 = gmsh.model.occ.add_point(x=L, y=0, z=0) 
point3 = gmsh.model.occ.add_point(x=L, y=H, z=0) 
point4 = gmsh.model.occ.add_point(x=0, y=H, z=0) 

point_lake_right  = gmsh.model.occ.add_point(x=L-S, y=H, z=0) 
point_lake_left = gmsh.model.occ.add_point(x=S, y=H, z=0) 
point_lake_mid = gmsh.model.occ.add_point(x=L/2, y=H-h, z=0) 
point_bottom_mid = gmsh.model.occ.add_point(x=L/2, y=0, z=0) 

point_lake_mid_right = gmsh.model.occ.add_point(x=L/2 + (B/4), y=H-h, z=0) 
point_lake_mid_right_crack_down = gmsh.model.occ.add_point(x=L/2 + (B/4), y=(H-h)-(2*h), z=0) 
point_lake_mid_right_epsilon = gmsh.model.occ.add_point(x=(L/2 + (B/4))-2, y=H-h, z=0) 

point_lake_mid = gmsh.model.occ.add_point(x=L/2, y=H-h, z=0) 
point_lake_mid_crack = gmsh.model.occ.add_point(x=L/2, y=(H-h)-(0.2*h), z=0) 
point_lake_mid_epsilon = gmsh.model.occ.add_point(x=L/2 -1 , y=H-h, z=0) 
point_bottom_mid = gmsh.model.occ.add_point(x=L/2, y=0, z=0) 

point_lake_mid_left = gmsh.model.occ.add_point(x=L/2 - (B/4), y=H-h, z=0) 
point_lake_arc_right = gmsh.model.occ.add_point(x=L/2 + (B/4)  + (B/8) , y=(H-((4*h)/5)) , z=0) 
point_lake_arc_left = gmsh.model.occ.add_point(x=L/2 - (B/4)  - (B/8) , y=(H-((4*h)/5)), z=0) 

# ----------- Lines -----------
arc_lake_right = gmsh.model.occ.add_circle_arc(point_lake_right, point_lake_arc_right ,point_lake_mid_right, center = False)
crack_lake_right_down = gmsh.model.occ.add_line(point_lake_mid_right, point_lake_mid_right_crack_down)
crack_lake_right_up = gmsh.model.occ.add_line( point_lake_mid_right_crack_down, point_lake_mid_right_epsilon)
line_lake_right = gmsh.model.occ.add_line(point_lake_mid_right_epsilon, point_lake_mid)
crack_lake_mid_down = gmsh.model.occ.add_line( point_lake_mid, point_lake_mid_crack)
crack_lake_mid_up = gmsh.model.occ.add_line( point_lake_mid_crack, point_lake_mid_epsilon)
line_lake_left = gmsh.model.occ.add_line(point_lake_mid_epsilon, point_lake_mid_left)
arc_lake_left = gmsh.model.occ.add_circle_arc(point_lake_mid_left, point_lake_arc_left ,point_lake_left, center = False)

line1 = gmsh.model.occ.add_line(point1, point1a)
line1a = gmsh.model.occ.add_line(point1a, point2) 
line2 = gmsh.model.occ.add_line(point2, point3)
line3 = gmsh.model.occ.add_line(point3, point_lake_right) 
line4 = gmsh.model.occ.add_line(point_lake_left, point4) 
line5 = gmsh.model.occ.add_line(point4, point1) 
lineRefinment = gmsh.model.occ.add_line(point_bottom_mid, point_lake_mid) 

lake_lines = [arc_lake_right, crack_lake_right_down, crack_lake_right_up, line_lake_right,
         crack_lake_mid_down, crack_lake_mid_up, line_lake_left, arc_lake_left]

loop = gmsh.model.occ.add_curve_loop([line1,line1a,line2,line3,arc_lake_right,crack_lake_right_down,crack_lake_right_up, line_lake_right,crack_lake_mid_down,crack_lake_mid_up, line_lake_left, arc_lake_left, line4,line5])
area = gmsh.model.occ.add_plane_surface([loop])
gmsh.model.occ.synchronize()


# ----------- Tags -----------
gmsh.model.addPhysicalGroup(dim=gdim, tags=[area], tag=1)               # area
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=[line5], tag=2)            # left edge 
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=[line1, line1a], tag=3)    # bottom 
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=[line2], tag=4)            # right edge 
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=lake_lines, tag=5)         # lake


# ----------- Mesh refinement -----------
fine_line = [lineRefinment]
fine_points = [point_lake_mid]
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", fine_line)

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", h_size/3)
gmsh.model.mesh.field.setNumber(2, "SizeMax", h_size)
gmsh.model.mesh.field.setNumber(2, "DistMin", 270)
gmsh.model.mesh.field.setNumber(2, "DistMax", 350)

gmsh.model.mesh.field.setAsBackgroundMesh(2)
gmsh.model.mesh.generate(dim=gdim)



mesh, cell_tags, facet_tags = dlfx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=gdim)

# save mesh as XDMF
with dlfx.io.XDMFFile(MPI.COMM_WORLD, mesh_name+'.xdmf', 'w') as xdmfout:
    xdmfout.write_mesh(mesh)
    xdmfout.write_meshtags(cell_tags, mesh.geometry) 
    xdmfout.write_meshtags(facet_tags, mesh.geometry) 
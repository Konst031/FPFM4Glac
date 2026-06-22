import numpy as np
import dolfinx as dlfx
import gmsh
from mpi4py import MPI

mesh_name = "../mesh/00_Test"
gdim = 2 
h_size = 0.02 # denser mesh if h_size is smaller

gmsh.initialize()
gmsh.clear()
gmsh.model.add(mesh_name)

# Geometry
L = 5500            # length of mesh
H = 500             # height of mesh
S = 800             # Distance from leftmost point of lake to left edge
h = 50              # Depth of lake
d = 0              # Depth of initial crack (0 means no crack)
pos_crack = 0.2     # fractional position within width of lake (0 -> left edge of lake; 1 -> right edge of lake)
crack_width = 20     
lake_width = 1200 
point1 = gmsh.model.occ.add_point(x=0, y=0, z=0) 
point2 = gmsh.model.occ.add_point(x=L, y=0, z=0) 
point3 = gmsh.model.occ.add_point(x=L, y=H, z=0) 
point4 = gmsh.model.occ.add_point(x=0, y=H, z=0) 
# ensure that there is a point in the middle of the bedrock. Handy for some BCs
point_mid_bottom = gmsh.model.occ.add_point(x=L/2,y=0,z=0) 

point_lake_right  = gmsh.model.occ.add_point(x=S+lake_width, y=H, z=0) 
point_lake_left = gmsh.model.occ.add_point(x=S, y=H, z=0) 
x_lake_cen = S+lake_width/2
y_lake_circ_cen = H+((lake_width/2)**2 - (h*h))/(2*h)
center_circle_arc = gmsh.model.occ.add_point(x=x_lake_cen, y=y_lake_circ_cen, z=0) # center of circle arch

line1a = gmsh.model.occ.add_line(point1, point_mid_bottom)
line1b = gmsh.model.occ.add_line(point_mid_bottom, point2)
line2 = gmsh.model.occ.add_line(point2, point3)
line3 = gmsh.model.occ.add_line(point3, point_lake_right) 
line4 = gmsh.model.occ.add_line(point_lake_left, point4) 
line5 = gmsh.model.occ.add_line(point4, point1) 

# add crack
if d > 0:
    if (pos_crack * lake_width - crack_width/2 < 0):
        raise ValueError('Initial crack out of bounds of lake: Choose larger pos_crack or smaller crack_width')
    if (pos_crack*lake_width + crack_width/2 > lake_width):
        raise ValueError('Initial crack out of bounds of lake: Choose smaller pos_crack or smaller crack_width')
    
    x_crack_cen = S + lake_width * pos_crack
    x_crack_left = x_crack_cen - crack_width/2
    x_crack_right = x_crack_cen + crack_width/2

    radius_lake_circ = y_lake_circ_cen - H + h
    get_y = lambda x_pos: y_lake_circ_cen - np.sqrt(radius_lake_circ**2- np.abs(x_lake_cen - x_pos)**2)

    point_crack_center = gmsh.model.occ.add_point(x=x_crack_cen, y=get_y(x_crack_cen)-d, z=0)
    point_crack_left = gmsh.model.occ.add_point(x=x_crack_left, y=get_y(x_crack_left), z=0) 
    point_crack_right = gmsh.model.occ.add_point(x=x_crack_right, y=get_y(x_crack_right), z=0) 

    line_lake_1 = gmsh.model.occ.add_circle_arc(startTag=point_lake_right, middleTag=center_circle_arc, endTag=point_crack_right, center=True)
    line_lake_2 = gmsh.model.occ.add_circle_arc(startTag=point_crack_left, middleTag=center_circle_arc, endTag=point_lake_left, center=True)
    line_crack_1 = gmsh.model.occ.add_line(point_crack_right, point_crack_center)
    line_crack_2 = gmsh.model.occ.add_line(point_crack_center, point_crack_left)
    lines_lake = [line_lake_1, line_lake_2, line_crack_1, line_crack_2]

    loop = gmsh.model.occ.add_curve_loop([line1a,line1b,line2,line3, line_lake_1, line_crack_1, line_crack_2, line_lake_2, line4,line5])

else:
    # without crack
    line_lake = gmsh.model.occ.add_circle_arc(startTag=point_lake_right, middleTag=center_circle_arc, endTag=point_lake_left, center=True)
    lines_lake = [line_lake]
    loop = gmsh.model.occ.add_curve_loop([line1a,line1b,line2,line3,line_lake,line4,line5])

area = gmsh.model.occ.add_plane_surface([loop])
gmsh.model.occ.synchronize()


# ------------ Tags ------------

gmsh.model.addPhysicalGroup(dim=gdim, tags=[area], tag=1)               # area
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=[line5], tag=2)            # left edge
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=[line1a,line1b], tag=3)    # bottom 
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=[line2], tag=4)            # right edge 
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=lines_lake, tag=5)         # lake


gmsh.option.setNumber('Mesh.CharacteristicLengthFactor',h_size)
gmsh.model.mesh.generate(dim=gdim)


mesh, cell_tags, facet_tags = dlfx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=gdim)

# save as XDMF
with dlfx.io.XDMFFile(MPI.COMM_WORLD, mesh_name+'.xdmf', 'w') as xdmfout:
    xdmfout.write_mesh(mesh)
    xdmfout.write_meshtags(cell_tags, mesh.geometry) 
    xdmfout.write_meshtags(facet_tags, mesh.geometry) 
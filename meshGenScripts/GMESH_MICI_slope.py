import numpy as np
import dolfinx as dlfx
import gmsh
from mpi4py import MPI
import sys
import csv

width = 3000
height = 1000
meshFilename = '../mesh/Mesh_Mici_slope_test.xdmf'
data = [
    {'id': "1", 'xcoord': 0, 'ycoord': 0},
    {'id': "2", 'xcoord': width, 'ycoord': 100},
    {'id': "3", 'xcoord': width, 'ycoord': height},
    {'id': "4", 'xcoord': 0, 'ycoord': height},
]

with open('../mesh/MICIMesh.csv', 'w', newline='') as csvfile:
    fieldnames = ['id', 'xcoord', 'ycoord']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)


gdim = 2 #Dimension


with open('../mesh/MICIMesh.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

#'Auslesen der CSV'
gmsh.initialize()
gmsh.clear()
gmsh.model.add('MICIMesh')

length = len(data)

#Geometrie erstellen
points = [gmsh.model.occ.add_point(x=float(data[i][1]), y=float(data[i][2]), z=0) for i in range(1,length) ]
#avg_point = gmsh.model.occ.add_point(x=np.mean([float(data[i][1]) for i in range(1,length)]), y=np.mean([float(data[i][2]) for i in range(1,length)]), z=0)
lines = [gmsh.model.occ.add_line(points[i], points[i+1]) for i in range(0,len(points)-1)]
lines.append(gmsh.model.occ.add_line(points[len(points)-1], points[0]))
loop = gmsh.model.occ.add_curve_loop(lines)
area = gmsh.model.occ.add_plane_surface([loop])
gmsh.model.occ.synchronize()

#Tags
gmsh.model.addPhysicalGroup(dim=gdim, tags=[area], tag=1)       # Tag for area
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=range(4,5), tag=2) # Tag for inflow
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=range(1,2), tag=9) # Tag for bottom
gmsh.model.addPhysicalGroup(dim=gdim-1, tags=range(2,3), tag=4) # Tag for cliff

# Netz 
h_size = float(input("Enter h_size: ").strip() or "50")
gmsh.option.setNumber('Mesh.CharacteristicLengthMax',h_size)

# Optional refinement
# h_size_fine = float(input("Enter h_size_fine: ").strip() or "20") #verfeinerte Elementgröße

# fine_lines = [2] # Liste von Linien, um die verfeinert werden soll
# dist_min = 1200
# dist_max = 1750

# gmsh.model.mesh.field.add("Distance", 1)

# gmsh.model.mesh.field.setNumbers(1, "CurvesList", fine_lines)
# gmsh.model.mesh.field.setNumber(1,"Sampling",100)

# gmsh.model.mesh.field.add("Threshold", 2)
# gmsh.model.mesh.field.setNumber(2, "InField", 1)
# gmsh.model.mesh.field.setNumber(2, "SizeMin", h_size_fine)
# gmsh.model.mesh.field.setNumber(2, "SizeMax", h_size)
# gmsh.model.mesh.field.setNumber(2, "DistMin", dist_min)
# gmsh.model.mesh.field.setNumber(2, "DistMax", dist_max)


gmsh.model.mesh.field.setAsBackgroundMesh(2)
gmsh.model.mesh.generate(dim=gdim)



mesh, cell_tags, facet_tags = dlfx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=gdim)
# save as XDMF
with dlfx.io.XDMFFile(MPI.COMM_WORLD, meshFilename, 'w') as xdmfout:
    xdmfout.write_mesh(mesh)
    xdmfout.write_meshtags(cell_tags, mesh.geometry) 
    xdmfout.write_meshtags(facet_tags, mesh.geometry) 
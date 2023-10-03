import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis as cfv
from calfem.utils import save_mesh, save_geometry
import calfem.editor as editgeo
import os
import numpy as np
from tkinter import filedialog

# Simulation points in cm
# rectangles
directory = filedialog.askdirectory()
os.chdir(directory)
mesh_size_file = open('Mesh_sizes.txt')
foil_el_size = float(mesh_size_file.readline().split('=')[-1].split(' ')[1])
water_el_size = float(mesh_size_file.readline().split('=')[-1].split(' ')[1])
mesh_size_file.close()
physical_size_file = open('Physical_sizes.txt')
L_foil = float(physical_size_file.readline().split('=')[-1].split(' ')[1])
W_foil = float(physical_size_file.readline().split('=')[-1].split(' ')[1])
H_foil = float(physical_size_file.readline().split('=')[-1].split(' ')[1])
L_water = float(physical_size_file.readline().split('=')[-1].split(' ')[1])
W_water = float(physical_size_file.readline().split('=')[-1].split(' ')[1])
H_water = float(physical_size_file.readline().split('=')[-1].split(' ')[1])
physical_size_file.close()
pass
w_foil = W_foil / 2.0
h_foil = H_foil / 2.0
l_foil = L_foil / 2.0

w_max = W_water / 2.0
h_max = H_water / 2.0
l_max = L_water / 2.0
# A Geometry-object
# This creates a Geometry object which will be used to described our geometry
g = cfg.Geometry()
# g = editgeo.edit_geometry(g)

# foil points

g.point([0, 0, 0], 0, el_size=foil_el_size)
g.point([w_foil, 0, 0], 1, el_size=foil_el_size)
g.point([0, h_foil, 0], 2, el_size=foil_el_size)
g.point([0, 0, l_foil], 3, el_size=foil_el_size)
g.point([0, h_foil, l_foil], 4, el_size=foil_el_size)
g.point([w_foil, h_foil, 0], 5, el_size=foil_el_size)
g.point([w_foil, 0, l_foil], 6, el_size=foil_el_size)
g.point([w_foil, h_foil, l_foil], 7, el_size=foil_el_size)
# foil lines
g.line([0, 1], 1)
g.line([0, 2], 2)
g.line([0, 3], 3)
g.line([2, 4], 4)
g.line([2, 5], 5)
g.line([1, 5], 6)
g.line([1, 6], 7)
g.line([3, 6], 8)
g.line([3, 4], 9)
g.line([4, 7], 11)
g.line([5, 7], 12)
g.line([6, 7], 13)
# foil surface
g.surface([1, 2, 5, 6], ID=1)
g.surface([2, 3, 9, 4], ID=2)
g.surface([1, 3, 8, 7], ID=3)
g.surface([6, 7, 13, 12], ID=4)
g.surface([8, 13, 11, 9], ID=5)
g.surface([5, 12, 11, 4], ID=6)
# foil volume
g.volume([1, 2, 3, 4, 5, 6], ID=1, marker=10)

# water points
g.point([w_max, 0, 0], 8)
g.point([0, h_max, 0], 9)
g.point([0, 0, l_max], 10)
g.point([0, h_max, l_max], 11)
g.point([w_max, 0, l_max], 12)
g.point([w_max, h_max, 0], 13)
g.point([w_max, h_max, l_max], 14)
# water lines
g.line([1, 8], 14)
g.line([2, 9], 15)
g.line([3, 10], 16)
g.line([8, 12], 17)
g.line([8, 13], 18)
g.line([9, 11], 19)
g.line([9, 13], 20)
g.line([10, 11], 21)
g.line([10, 12], 22)
g.line([11, 14], 23)
g.line([12, 14], 24)
g.line([13, 14], 25)
# water surface
g.surface([6, 5, 15, 20, 18, 14], ID=7)
g.surface([4, 9, 16, 21, 19, 15], ID=8)
g.surface([8, 7, 14, 17, 22, 16], ID=9)
g.surface([19, 23, 25, 20], ID=10)
g.surface([18, 25, 24, 17], ID=11)
g.surface([21, 23, 24, 22], ID=12)
# water volume
g.volume([6, 5, 4, 7, 8, 9, 10, 11, 12], ID=2, marker=100)

# display our geometry, we use the calfem.vis module
cfv.drawGeometry(g)
cfv.showAndWait()

# To create a mesh we need to create a GmshMesh object and initialize this with our geometry
mesh = cfm.GmshMesh(g)

# set some desired properties on our mesh
mesh.elType = 4  # Degrees of freedom per node.
mesh.dofsPerNode = 2  # Factor that changes element sizes.
mesh.elSizeFactor = water_el_size  # Element size Factor

# To generate the mesh and at the same time get the needed data structures for use with CALFEM we call the .create() method of the mesh object
coords, edof, dofs, bdofs, elementmarkers = mesh.create()
tetras = mesh.topo
# triangles = mesh.topo
points_fringe_dict = mesh.nodesOnCurve
points_surface_dict = mesh.nodesOnSurface
points_volume_dict = mesh.nodesOnVolume

'''

        coords - Element coordinates

        edof - Element topology

        dofs - Degrees of freedom per node

        bdofs - Boundary degrees of freedom. Dictionary containing the dofs for each boundary marker. More on markers in the next section.

        elementmarkers - List of integer markers. Row i contains the marker of element i. Can be used to determine what region an element is in.

'''

# To display the generated mesh we can use the drawMesh() function of the calfem.vis module
cfv.figure()
# Draw the mesh.

cfv.drawMesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofsPerNode,
    el_type=mesh.elType,
    filled=True,
    title="Example 01"
)
cfv.showAndWait()

try:
    os.chdir('Mesh')
except:
    os.mkdir('Mesh')
    os.chdir('Mesh')
np.savetxt('Mesh_points.csv', np.array(coords))  # coordinates of the nodes points
np.savetxt('Mesh_elements.csv', np.array(tetras) - 1, fmt='%d')  # corresponance of tetraeder numbers and points numbers
np.savetxt('Mesh_markers.csv', np.array(elementmarkers), fmt='%d')
# print(np.array(coords)[points_surface_dict[8]])

Mesh_1_sector_surface_nodes = np.concatenate([points_surface_dict[i] for i in [1, 7]])  # surface where v_z==0
Mesh_2_sector_surface_nodes = np.concatenate([points_surface_dict[i] for i in [2, 8]])  # surface where v_x==0
Mesh_3_sector_surface_nodes = np.concatenate([points_surface_dict[i] for i in [3, 9]])  # surface where v_y==0
np.savetxt('Mesh_1_sector_surface_nodes.csv', Mesh_1_sector_surface_nodes, fmt='%d')
np.savetxt('Mesh_2_sector_surface_nodes.csv', Mesh_2_sector_surface_nodes, fmt='%d')
np.savetxt('Mesh_3_sector_surface_nodes.csv', Mesh_3_sector_surface_nodes, fmt='%d')

np.savetxt('Mesh_foil_points.csv', points_volume_dict[1], fmt='%d')
np.savetxt('Mesh_water_points.csv', points_volume_dict[2], fmt='%d')
np.savetxt('Mesh_1_sector_surface_water.csv', points_surface_dict[7], fmt='%d')  # surface for report
np.savetxt('Mesh_2_sector_surface_water.csv', points_surface_dict[8], fmt='%d')  # surface for report
np.savetxt('Mesh_3_sector_surface_water.csv', points_surface_dict[9], fmt='%d')  # surface for report
np.savetxt('Mesh_1_sector_fringe_water.csv', points_fringe_dict[14], fmt='%d')  # surface for report
np.savetxt('Mesh_2_sector_fringe_water.csv', points_fringe_dict[15], fmt='%d')  # surface for report
np.savetxt('Mesh_3_sector_fringe_water.csv', points_fringe_dict[16], fmt='%d')  # surface for report

os.chdir('..')

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
g.volume([1, 2, 3, 4, 5, 6], ID=1)

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
g.volume([6, 5, 4, 7, 8, 9, 10, 11, 12], ID=2)
'''
# Foil
g.spline([0, 1], 1)
g.spline([0, 2], 2)
g.spline([0, 3], 3)
g.spline([1, 5], 4)
g.spline([1, 6], 5)
g.spline([2, 4], 6)
g.spline([2, 5], 7)
g.spline([3, 4], 8)
g.spline([3, 6], 9)
g.spline([4, 7], 10)
g.spline([5, 7], 11)
g.spline([6, 7], 12)

g.spline([1, 8], 12)
g.spline([2, 9], 13)
g.spline([3, 10], 14)
g.spline([9, 11], 15)
g.spline([9, 13], 16)
g.spline([8, 12], 17)
g.spline([8, 13], 18)
g.spline([10, 11], 19)
g.spline([10, 12], 20)
g.spline([11, 14], 21)
g.spline([12, 14], 22)
g.spline([13, 14], 23)

g.surface([1, 2, 4, 7], ID=1)  # me section border
g.surface([2, 3, 6, 8], ID=2)  # me section border
g.surface([1, 3, 9, 5], ID=3)  # me section border

g.surface([7, 11, 10, 6], ID=4)  # me inner border
g.surface([5, 12, 11, 4], ID=5)  # me inner border
g.surface([8, 10, 12, 9], ID=6)  # me inner border

g.surface([12, 18, 16, 13, 7, 4], ID=7)  # water section border
g.surface([13, 15, 19, 14, 8, 6], ID=8)  # water section border
g.surface([14, 20, 17, 12, 5, 9], ID=9)  # water section border
g.surface([19, 21, 22, 20], ID=10)  # water outer border
g.surface([17, 22, 23, 18], ID=11)  # water outer border
g.surface([15, 21, 23, 16], ID=12)  # water outer border

g.volume([1, 2, 3, 4, 5, 6], ID=1, marker=10)  # foil
g.volume([4, 5, 6, 7, 8, 9, 10, 11, 12], ID=1, marker=10)  # foil
# create the surface by defining what lines make up the surface
# g.surface([0, 1, 2, 3], ID=0, marker=10)  # wire
# g.surface([1, 4, 5, 6, 7, 8, 2], ID=1, marker=100)  # water'''

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
#triangles = mesh.topo
borders = mesh.nodesOnCurve
nodes = mesh.nodesOnSurface
# save_mesh(mesh, 'CurrentMesh')
# save_geometry(g, 'CurrentGeo')
pass
'''bdofs_0_set = set(bdofs[0])
bdofs_40_set = set(bdofs[40])
bdofs_0_set_new = bdofs_0_set - bdofs_40_set
bdofs[0] = np.array(list(bdofs_0_set_new))'''
pass
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

'''try:
    os.chdir('Mesh')
except:
    os.mkdir('Mesh')
    os.chdir('Mesh')
# os.chdir('Mesh')
np.savetxt('Mesh_points.csv', np.array(coords))
np.savetxt('Mesh_elements.csv', np.array(triangles) - 1, fmt='%d')
np.savetxt('Mesh_facets.csv', np.array(dofs), fmt='%d')
np.savetxt('Mesh_markers.csv', np.array(elementmarkers), fmt='%d')
np.savetxt('Mesh_vert_sector_border.csv', np.concatenate([borders[i] for i in [3, 8]]), fmt='%d')
np.savetxt('Mesh_vert_sector_border_me.csv', borders[3], fmt='%d')
np.savetxt('Mesh_hori_sector_border.csv', np.concatenate([borders[i] for i in [0, 4]]), fmt='%d')
np.savetxt('Mesh_hori_sector_border_me.csv', borders[0], fmt='%d')
# np.savetxt('Mesh_border_3.csv', np.array(bdofs[3]), fmt='%d')
np.savetxt('Mesh_wire_outer_border.csv', np.concatenate([borders[i] for i in [1, 2]]), fmt='%d')
np.savetxt('Mesh_outer_border.csv', np.concatenate([borders[i] for i in [5, 6, 7]]), fmt='%d')
np.savetxt('Mesh_water_points.csv', nodes[1], fmt='%d')
# np.savetxt('Mesh_border_OR.csv', np.array(borders[5]), fmt='%d')

os.chdir('..')'''

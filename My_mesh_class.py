import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis as cfv
from calfem.utils import save_mesh, save_geometry
import calfem.editor as editgeo
import os
import numpy as np
from tkinter import filedialog


class MyMesh:
    def __init__(self):
        self.read_folder()
        print("Folder is read")
        self.generate_geometry()
        print("Geometry is generated")
        self.generate_mesh()
        print("Mesh is generated")
        self.separate_mesh()
        print("Mesh is separated")
        self.define_triangles()
        print("Triangles are defined")
        self.save_mesh()
        print("Mesh is saved")

    def read_folder(self):
        directory = filedialog.askdirectory()
        os.chdir(directory)
        mesh_size_file = open('Mesh_sizes.txt')
        self.foil_el_size = float(mesh_size_file.readline().split('=')[-1].split(' ')[1])
        self.water_el_size = float(mesh_size_file.readline().split('=')[-1].split(' ')[1])
        mesh_size_file.close()
        physical_size_file = open('Physical_sizes.txt')
        L_foil = float(physical_size_file.readline().split('=')[-1].split(' ')[1])
        W_foil = float(physical_size_file.readline().split('=')[-1].split(' ')[1])
        H_foil = float(physical_size_file.readline().split('=')[-1].split(' ')[1])
        L_water = float(physical_size_file.readline().split('=')[-1].split(' ')[1])
        W_water = float(physical_size_file.readline().split('=')[-1].split(' ')[1])
        H_water = float(physical_size_file.readline().split('=')[-1].split(' ')[1])
        physical_size_file.close()
        self.w_foil = W_foil / 2.0
        self.h_foil = H_foil / 2.0
        self.l_foil = L_foil / 2.0
        self.w_max = W_water / 2.0
        self.h_max = H_water / 2.0
        self.l_max = L_water / 2.0

    def generate_geometry(self):
        # A Geometry-object
        # This creates a Geometry object which will be used to described our geometry
        g = cfg.Geometry()
        # g = editgeo.edit_geometry(g)

        # foil points

        g.point([0, 0, 0], 0, el_size=self.foil_el_size)
        g.point([self.w_foil, 0, 0], 1, el_size=self.foil_el_size)
        g.point([0, self.h_foil, 0], 2, el_size=self.foil_el_size)
        g.point([0, 0, self.l_foil], 3, el_size=self.foil_el_size)
        g.point([0, self.h_foil, self.l_foil], 4, el_size=self.foil_el_size)
        g.point([self.w_foil, self.h_foil, 0], 5, el_size=self.foil_el_size)
        g.point([self.w_foil, 0, self.l_foil], 6, el_size=self.foil_el_size)
        g.point([self.w_foil, self.h_foil, self.l_foil], 7, el_size=self.foil_el_size)
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
        g.point([self.w_max, 0, 0], 8)
        g.point([0, self.h_max, 0], 9)
        g.point([0, 0, self.l_max], 10)
        g.point([0, self.h_max, self.l_max], 11)
        g.point([self.w_max, 0, self.l_max], 12)
        g.point([self.w_max, self.h_max, 0], 13)
        g.point([self.w_max, self.h_max, self.l_max], 14)
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
        self.g = g

    def generate_mesh(self):
        # To create a mesh we need to create a GmshMesh object and initialize this with our geometry
        mesh = cfm.GmshMesh(self.g)

        # set some desired properties on our mesh
        mesh.elType = 4  # Degrees of freedom per node.
        mesh.dofsPerNode = 2  # Factor that changes element sizes.
        mesh.elSizeFactor = self.water_el_size  # Element size Factor

        # To generate the mesh and at the same time get the needed data structures for use with CALFEM we call the .create() method of the mesh object
        coords, edof, dofs, bdofs, elementmarkers = mesh.create()

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

        self.coords = coords
        self.mesh_dict = {
            'tetras': np.array(mesh.topo) - 1,
            'elementmarkers': elementmarkers,
        }
        self.mesh = mesh

    def separate_mesh(self):
        nodes_fringe_dict = self.mesh.nodesOnCurve
        nodes_surface_dict = self.mesh.nodesOnSurface
        nodes_volume_dict = self.mesh.nodesOnVolume
        index_dict = {
            'Mesh_XY_sector_surface_nodes': np.concatenate([nodes_surface_dict[i] for i in [1, 7]]),  # surface XY
            'Mesh_YZ_sector_surface_nodes': np.concatenate([nodes_surface_dict[i] for i in [2, 8]]),  # surface YZ
            'Mesh_XZ_sector_surface_nodes': np.concatenate([nodes_surface_dict[i] for i in [3, 9]]),  # surface XZ
            'Mesh_XY_sector_surface_nodes_water': nodes_surface_dict[7],  # surface XY
            'Mesh_YZ_sector_surface_nodes_water': nodes_surface_dict[8],  # surface YZ
            'Mesh_XZ_sector_surface_nodes_water': nodes_surface_dict[9],  # surface XZ
            'Mesh_X_sector_frange_nodes_water': nodes_fringe_dict[14],  # axis X
            'Mesh_Y_sector_frange_nodes_water': nodes_fringe_dict[15],  # axis Y
            'Mesh_Z_sector_frange_nodes_water': nodes_fringe_dict[16],  # axis Z
            'Mesh_volume_nodes_foil': nodes_volume_dict[1],
            'Mesh_volume_nodes_water': nodes_volume_dict[2],
            'Mesh_outer_surface': np.concatenate([nodes_surface_dict[i] for i in [10, 11, 12]]),
        }
        self.mesh_dict = self.mesh_dict | index_dict

    def define_triangles(self):
        tetra_list_XY = []
        triangle_list_XY = []
        Mesh_XY_sector_surface_nodes_set = set(self.mesh_dict['Mesh_XY_sector_surface_nodes_water'])
        for i, tet in enumerate(self.mesh_dict['tetras']):
            tet_set = set(tet)
            tr_list = list(tet_set & Mesh_XY_sector_surface_nodes_set)
            if len(tr_list) == 3:
                tr_list_new_num = tr_list
                for k, n in enumerate(tr_list):
                    new_index = np.where(self.mesh_dict['Mesh_XY_sector_surface_nodes_water'] == n)[0][0]
                    tr_list_new_num[k] = new_index
                triangle_list_XY.append(tr_list_new_num)
                tetra_list_XY.append(i)

        tetra_list_YZ = []
        triangle_list_YZ = []
        Mesh_YZ_sector_surface_nodes_set = set(self.mesh_dict['Mesh_YZ_sector_surface_nodes_water'])
        for i, tet in enumerate(self.mesh_dict['tetras']):
            tet_set = set(tet)
            tr_list = list(tet_set & Mesh_YZ_sector_surface_nodes_set)
            if len(tr_list) == 3:
                tr_list_new_num = tr_list
                for k, n in enumerate(tr_list):
                    new_index = np.where(self.mesh_dict['Mesh_YZ_sector_surface_nodes_water'] == n)[0][0]
                    tr_list_new_num[k] = new_index
                triangle_list_YZ.append(tr_list_new_num)
                tetra_list_YZ.append(i)

        tetra_list_XZ = []
        triangle_list_XZ = []
        Mesh_XZ_sector_surface_nodes_set = set(self.mesh_dict['Mesh_XZ_sector_surface_nodes_water'])
        for i, tet in enumerate(self.mesh_dict['tetras']):
            tet_set = set(tet)
            tr_list = list(tet_set & Mesh_XZ_sector_surface_nodes_set)
            if len(tr_list) == 3:
                tr_list_new_num = tr_list
                for k, n in enumerate(tr_list):
                    new_index = np.where(self.mesh_dict['Mesh_XZ_sector_surface_nodes_water'] == n)[0][0]
                    tr_list_new_num[k] = new_index
                triangle_list_XZ.append(tr_list_new_num)
                tetra_list_XZ.append(i)

        tetra_list_X = []
        Mesh_X_sector_frange_nodes_water_set = set(self.mesh_dict['Mesh_X_sector_frange_nodes_water'])
        for i, tet in enumerate(self.mesh_dict['tetras']):
            tet_set = set(tet)
            tr_list = list(tet_set & Mesh_X_sector_frange_nodes_water_set)
            if len(tr_list) == 2:
                tetra_list_X.append(i)

        tetra_list_Y = []
        Mesh_Y_sector_frange_nodes_water_set = set(self.mesh_dict['Mesh_Y_sector_frange_nodes_water'])
        for i, tet in enumerate(self.mesh_dict['tetras']):
            tet_set = set(tet)
            tr_list = list(tet_set & Mesh_Y_sector_frange_nodes_water_set)
            if len(tr_list) == 2:
                tetra_list_Y.append(i)
        tetra_list_Z = []
        Mesh_Z_sector_frange_nodes_water_set = set(self.mesh_dict['Mesh_Z_sector_frange_nodes_water'])
        for i, tet in enumerate(self.mesh_dict['tetras']):
            tet_set = set(tet)
            tr_list = list(tet_set & Mesh_Z_sector_frange_nodes_water_set)
            if len(tr_list) == 2:
                tetra_list_Z.append(i)
        tr_dict = {
            'tetra_XY': np.array(tetra_list_XY),
            'tetra_YZ': np.array(tetra_list_YZ),
            'tetra_XZ': np.array(tetra_list_XZ),
            'triangle_XY': np.array(triangle_list_XY),
            'triangle_YZ': np.array(triangle_list_YZ),
            'triangle_XZ': np.array(triangle_list_XZ),
            'tetra_X': np.array(tetra_list_X),
            'tetra_Y': np.array(tetra_list_Y),
            'tetra_Z': np.array(tetra_list_Z),
        }
        self.mesh_dict |= tr_dict

    def save_mesh(self):

        try:
            os.chdir('Mesh')
        except:
            os.mkdir('Mesh')
            os.chdir('Mesh')
        np.savetxt('Mesh_points.csv', np.array(self.coords))  # coordinates of the nodes points
        for my_key, my_data in self.mesh_dict.items():
            np.savetxt(f'{my_key}.csv', my_data, fmt='%d')

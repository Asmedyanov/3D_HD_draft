import multiprocessing as mp
import os
import time

import numpy as np
from numpy import tile, pi, zeros, ones, sqrt, concatenate, where, power
from EOS import EOS
from Read_Time_Power import Read_Time_Power
from Viscosity import *
from tkinter import filedialog
import traceback
from Force import *
from My_Math import *
from Volume import *
from matplotlib import pyplot as plt


class Simulator:
    def __init__(self):
        self.load_folder()
        self.init_Pool()
        print('init_Pool finished')
        self.init_EOS()
        print('init_EOS finished')
        self.init_Mesh()
        print('init_Mesh finished')
        self.init_Masses()
        print('init_Masses finished')
        self.init_Power()
        print('init_Power finished')

        self.init_tempVectors()
        print('init_tempVectors finished')

        self.init_figure()
        print('init_figure finished')
        self.init_folders()
        print('init_folders finished')

    def load_folder(self):
        self.directory = filedialog.askdirectory()
        print(self.directory)
        os.chdir(self.directory)

    def init_figure(self):
        plt.clf()
        self.fig, self.ax = plt.subplots(2, 2)
        ratio = self.points[:, 0].max() / self.points[:, 1].max()
        print(f'ratio = {ratio}')
        self.fig.set_size_inches(11.7, 8.3)
        self.ax[0, 0].set_title('Resitive power')
        self.ax[0, 0].set_xlabel('Time, ns')
        self.ax[0, 0].set_ylabel('Power, GW')
        self.ax[0, 0].plot(self.time_Power, self.Power)

        self.current_power_time_point = self.ax[0, 0].plot(0, 0, 'o')[0]
        self.current_power_time_annotation = self.ax[0, 0].annotate(f'{0} ns, {0} GW', xy=(0, 0), xytext=(3, 1.5),
                                                                    arrowprops=dict(facecolor='black', shrink=0.05))

        self.ax[0, 1].set_title('Density XY')
        self.ax[0, 1].set_xlabel('X, cm')
        self.ax[0, 1].set_ylabel('Y, cm')

        self.ax[1, 0].set_title('Density YZ')
        self.ax[1, 0].set_xlabel('X, cm')
        self.ax[1, 0].set_ylabel('Y, cm')

        self.ax[1, 1].set_title('Density XZ')
        self.ax[1, 1].set_xlabel('X, cm')
        self.ax[1, 1].set_ylabel('Y, cm')
        self.fmt = lambda x, pos: f'{x:.1e}'

        self.ax[0, 1].axis('equal')
        self.plot_ro_XY = self.ax[0, 1].tripcolor(self.points[self.sector_surface_water_XY, 0],
                                                  self.points[self.sector_surface_water_XY, 1], self.triangle_XY,
                                                  facecolors=self.Ro_Current[self.tetra_XY],
                                                  edgecolors='face')
        self.colorbar_ro_XY = self.fig.colorbar(self.plot_ro_XY, ax=self.ax[0, 1], format=self.fmt)

        self.ax[1, 0].axis('equal')

        self.plot_ro_YZ = self.ax[1, 0].tripcolor(self.points[self.sector_surface_water_YZ, 1],
                                                  self.points[self.sector_surface_water_YZ, 2], self.triangle_YZ,
                                                  facecolors=self.Ro_Current[self.tetra_YZ],
                                                  edgecolors='face')
        self.colorbar_ro_YZ = self.fig.colorbar(self.plot_ro_YZ, ax=self.ax[1, 0], format=self.fmt)
        self.ax[1, 1].axis('equal')
        self.plot_ro_XZ = self.ax[1, 1].tripcolor(self.points[self.sector_surface_water_XZ, 0],
                                                  self.points[self.sector_surface_water_XZ, 2], self.triangle_XZ,
                                                  facecolors=self.Ro_Current[self.tetra_XZ],
                                                  edgecolors='face')
        self.colorbar_ro_XZ = self.fig.colorbar(self.plot_ro_XZ, ax=self.ax[1, 1], format=self.fmt)
        self.fig.tight_layout()
        self.fig.savefig(f'Report Initial.png')
        # plt.show()
        pass

    def init_figure_velosity(self):
        plt.clf()
        self.fig, self.ax = plt.subplots(2, 2)
        ratio = self.points[:, 0].max() / self.points[:, 1].max()
        print(f'ratio = {ratio}')
        self.fig.set_size_inches(11.7, 8.3)
        self.ax[0, 0].set_title('Resitive power')
        self.ax[0, 0].set_xlabel('Time, ns')
        self.ax[0, 0].set_ylabel('Power, GW')
        self.ax[0, 0].plot(self.time_Power, self.Power)

        self.current_power_time_point = self.ax[0, 0].plot(0, 0, 'o')[0]
        self.current_power_time_annotation = self.ax[0, 0].annotate(f'{0} ns, {0} GW', xy=(0, 0), xytext=(3, 1.5),
                                                                    arrowprops=dict(facecolor='black', shrink=0.05))

        self.ax[0, 1].set_title('Velocity XY')
        self.ax[0, 1].set_xlabel('X, cm')
        self.ax[0, 1].set_ylabel('Y, cm')

        self.ax[1, 0].set_title('Velocity YZ')
        self.ax[1, 0].set_xlabel('X, cm')
        self.ax[1, 0].set_ylabel('Y, cm')

        self.ax[1, 1].set_title('Velosity XZ')
        self.ax[1, 1].set_xlabel('X, cm')
        self.ax[1, 1].set_ylabel('Y, cm')

        self.ax[0, 1].axis('equal')

        self.plot_v_1 = self.ax[0, 1].quiver(self.points[self.sector_surface_water_XY, 0],
                                             self.points[self.sector_surface_water_XY, 1],
                                             self.Velocity_Current[self.sector_surface_water_XY, 0],
                                             self.Velocity_Current[self.sector_surface_water_XY, 1])
        self.ax[1, 0].axis('equal')

        self.plot_v_2 = self.ax[1, 0].quiver(
            self.points[self.sector_surface_water_2, 1],
            self.points[self.sector_surface_water_2, 2],

            self.Velocity_Current[self.sector_surface_water_2, 1],
            self.Velocity_Current[self.sector_surface_water_2, 2])
        self.ax[1, 1].axis('equal')
        self.plot_v_3 = self.ax[1, 1].quiver(self.points[self.sector_surface_water_XZ, 0],

                                             self.points[self.sector_surface_water_XZ, 2],
                                             self.Velocity_Current[self.sector_surface_water_XZ, 0],

                                             self.Velocity_Current[self.sector_surface_water_XZ, 2])
        self.fig.tight_layout()
        self.fig.savefig(f'Report Initial.png')
        # plt.show()
        pass

    def init_folders(self):
        try:
            os.chdir('Attempts')
        except:
            os.mkdir('Attempts')
            os.chdir('Attempts')
        folder_name = f'Attempt_{len(os.listdir())}'
        os.mkdir(folder_name)
        os.chdir(folder_name)
        os.mkdir('BIN')
        os.mkdir('Mesh')

        self.vRecord = np.zeros((self.N_report, self.Np, 3), dtype=np.float32)
        self.energyRecord = np.zeros((self.N_report, self.Ntr), dtype=np.float32)
        self.roRecord = np.zeros((self.N_report, self.Ntr), dtype=np.float32)
        self.pRecord = np.zeros((self.N_report, self.Np, 3), dtype=np.float32)
        self.tempRecord = np.zeros((self.N_report, self.Ntr), dtype=np.float32)
        self.pressureRecord = np.zeros((self.N_report, self.Ntr), dtype=np.float32)
        self.file_r = open("BIN/r.bin", 'ab')
        self.file_T = open("BIN/T.bin", 'ab')
        self.file_P = open("BIN/P.bin", 'ab')
        self.file_v = open("BIN/v.bin", 'ab')
        self.file_E = open("BIN/E.bin", 'ab')
        self.file_Ro = open("BIN/Ro.bin", 'ab')
        self.n_recorded = 0
        self.main_report()

        pass

    def init_tempVectors(self):

        self.eInitial = np.ones(self.Ntr)
        self.eInitial[self.foilInd] = self.eMetalStart
        self.eInitial[self.waterInd] = self.eWaterStart
        self.Pressure_Current = ones(self.Ntr) * 1.0e-6
        self.Pressure_Prev = ones(self.Ntr) * 1.0e-6
        self.Pressure_Initial = ones(self.Ntr) * 1.0e-6
        self.presPred = ones(self.Ntr) * 1.0e-6
        self.presMed = ones(self.Ntr) * 1.0e-6
        self.presNew = ones(self.Ntr) * 1.0e-6
        self.Temperature_Current = ones(self.Ntr) * 300.0
        self.Temperature_Initial = ones(self.Ntr) * 300.0
        self.Ro_Current = np.ones(self.Ntr)
        self.Ro_Current[self.foilInd] = self.roMetalStart
        self.Ro_Current[self.waterInd] = self.roWaterStart
        self.Energy_Current = np.ones(self.Ntr)
        self.Energy_Current[self.foilInd] = self.eMetalStart
        self.Energy_Current[self.waterInd] = self.eWaterStart
        self.Pressure_Current[self.foilInd], self.Temperature_Current[self.foilInd] = self.Metal_EOS.ERoFindPT(
            self.eMetalStart,
            self.roMetalStart)
        self.Velocity_Current = zeros((self.Np, 3))

    def init_Pool(self):
        self.N_nuc = 14
        self.pool = mp.Pool(self.N_nuc + 1)

    def optimize_Pool(self):
        self.N_nuc = 2
        for mm in range(50):
            self.main_loop_mute(mm)
        start = time.perf_counter()
        for mm in range(50):
            self.main_loop_mute(mm)
        t_min = time.perf_counter() - start
        n_opt = 0
        times = []
        nuclei = []
        for n in range(3, 16):
            print(f'now check {n} cores')
            start = time.perf_counter()
            self.N_nuc = n
            for mm in range(50):
                self.main_loop_mute(mm)
            t_min_loc = time.perf_counter() - start
            if t_min > t_min_loc:
                t_min = t_min_loc
            print(t_min_loc)
            times.append(t_min_loc)
            nuclei.append(n)
        self.N_nuc = n_opt
        print(f'optimal_N_nuc = {n_opt}')
        plt.clf()
        plt.plot(nuclei, times)
        plt.show()

    def init_EOS(self):
        self.Metal_EOS = EOS('EOS/EOS_Me.KBT')
        self.Water_EOS = EOS('EOS/EOS_Water.KBT')
        # %%% Define initial density, energy and pressure
        self.roWaterStart = self.Water_EOS.rho_norm  # 0.998305  # 0.9982  # % initial density of water [g/cm^3]
        self.roMetalStart = self.Metal_EOS.rho_norm  # 8.93  # % initial density of wire [g/cm^3]

        self.eWaterStart = self.Water_EOS.e_norm  # % [J/kg]
        self.eMetalStart = self.Metal_EOS.e_norm  # % [J/kg]

        if self.eMetalStart < 0.2 * self.eWaterStart:
            self.eMetalStart = self.eWaterStart
        # %%% Define constants, all lengths are in [cm]
        self.A = 3e-3
        self.n = 7.15  # % polytropic index
        self.gamma = (self.n - 1) / 2

    def init_Mesh(self):
        # %%% Setting geometry
        self.points = np.loadtxt('GEO/Mesh/Mesh_points.csv')
        self.tetras = np.loadtxt('GEO/Mesh/tetras.csv', dtype='int')
        self.tetra_marker = np.loadtxt('GEO/Mesh/elementmarkers.csv', dtype='int')
        self.sector_surface_XY = np.loadtxt('GEO/Mesh/Mesh_XY_sector_surface_nodes.csv', dtype='int')  # surface XY
        self.sector_surface_water_XY = np.loadtxt('GEO/Mesh/Mesh_XY_sector_surface_nodes_water.csv',
                                                  dtype='int')  # surface XY
        self.sector_fringe_water_X = np.loadtxt('GEO/Mesh/Mesh_X_sector_frange_nodes_water.csv',
                                                dtype='int')  # axis X
        self.sector_surface_YZ = np.loadtxt('GEO/Mesh/Mesh_YZ_sector_surface_nodes.csv', dtype='int')  # surface YZ
        self.sector_surface_water_YZ = np.loadtxt('GEO/Mesh/Mesh_YZ_sector_surface_nodes_water.csv',
                                                  dtype='int')  # surface where v_x==0
        self.sector_fringe_water_Y = np.loadtxt('GEO/Mesh/Mesh_Y_sector_frange_nodes_water.csv',
                                                dtype='int')  # surface where v_x==0
        self.sector_surface_XZ = np.loadtxt('GEO/Mesh/Mesh_XZ_sector_surface_nodes.csv',
                                            dtype='int')  # surface where v_y==0
        self.sector_surface_water_XZ = np.loadtxt('GEO/Mesh/Mesh_XZ_sector_surface_nodes_water.csv',
                                                  dtype='int')  # surface where v_y==0
        self.sector_fringe_water_Z = np.loadtxt('GEO/Mesh/Mesh_Z_sector_frange_nodes_water.csv',
                                                dtype='int')  # surface where v_y==0
        self.border_outer = np.loadtxt('GEO/Mesh/Mesh_outer_surface.csv', dtype='int')
        self.waterInd_points = np.loadtxt('GEO/Mesh/Mesh_volume_nodes_water.csv', dtype='int')

        self.tetra_XY = np.loadtxt('GEO/Mesh/tetra_XY.csv', dtype='int')
        self.tetra_YZ = np.loadtxt('GEO/Mesh/tetra_YZ.csv', dtype='int')
        self.tetra_XZ = np.loadtxt('GEO/Mesh/tetra_XZ.csv', dtype='int')
        self.triangle_XY = np.loadtxt('GEO/Mesh/triangle_XY.csv', dtype='int')
        self.triangle_YZ = np.loadtxt('GEO/Mesh/triangle_YZ.csv', dtype='int')
        self.triangle_XZ = np.loadtxt('GEO/Mesh/triangle_XZ.csv', dtype='int')
        self.cross_metal_coef = np.loadtxt('GEO/Mesh/Cross_metal_coef.csv')
        # self.indxOR = np.loadtxt('Mesh_foil/Mesh_border_OR.csv', dtype='int')
        self.waterInd = np.argwhere(
            self.tetra_marker == 100)[:,0]  # %find all the indices of coordinates of water (subdomain 1)

        self.foilInd = np.argwhere(self.tetra_marker == 10)[:,0]

        '''tetras_water = self.tetras[self.waterInd]
        for i, ind in enumerate(self.waterInd_points):
            tetras_water = np.where(tetras_water == ind, i, tetras_water)
        self.tetras_water = tetras_water'''
        # %%% Set time
        Time_parameter_file = open('SIM/Time_parameter.txt')
        self.dT = float(Time_parameter_file.readline().split('=')[-1].split(' ')[1])
        self.Tmax = float(Time_parameter_file.readline().split('=')[-1].split(' ')[1])
        Time_parameter_file.close()
        self.Ntr = len(self.tetras)
        self.Np = len(self.points)

        physical_size_file = open('GEO/Physical_sizes.txt')
        #self.Length = float(physical_size_file.readline().split('=')[-1].split(' ')[1])  # length of cylinder,cm
        physical_size_file.close()

        Report_parameters_file = open('SIM/Report_parameters.txt')
        self.N_record = int(Report_parameters_file.readline().split('=')[-1].split(' ')[1])
        self.N_report = int(Report_parameters_file.readline().split('=')[-1].split(' ')[1])
        Report_parameters_file.close()

        self.timing = np.arange(0, self.Tmax, self.dT)
        self.Ntime = self.timing.size

        tetras_df = pd.DataFrame(self.tetras, columns=['Column0', 'Column1', 'Column2', 'Column3'])
        tetras_df['my_index'] = np.arange(self.Ntr)
        # func_pount_tetras_pandas(tetras_df,0,self.Np)

        proc_list = [self.pool.apply_async(func_pount_tetras_pandas, args=(tetras_df, i, self.Np)) for i in range(4)]

        points_tetras_0 = proc_list[0].get()
        points_tetras_1 = proc_list[1].get()
        points_tetras_2 = proc_list[2].get()
        points_tetras_3 = proc_list[3].get()
        pass
        '''for i in range(self.Np):
            tetras_to_add = np.argwhere(self.tetras[:, 0] == i)[:, 0]
            points_tetras_0.append(tetras_to_add)
            tetras_to_add = np.argwhere(self.tetras[:, 1] == i)[:, 0]
            points_tetras_1.append(tetras_to_add)
            tetras_to_add = np.argwhere(self.tetras[:, 2] == i)[:, 0]
            points_tetras_2.append(tetras_to_add)
            tetras_to_add = np.argwhere(self.tetras[:, 3] == i)[:, 0]
            points_tetras_3.append(tetras_to_add)'''
        self.list_points_tetras_0 = points_tetras_0
        self.list_points_tetras_1 = points_tetras_1
        self.list_points_tetras_2 = points_tetras_2
        self.list_points_tetras_3 = points_tetras_3
        self.points_tetras_0 = np.array(points_tetras_0, dtype=object)
        self.points_tetras_1 = np.array(points_tetras_1, dtype=object)
        self.points_tetras_2 = np.array(points_tetras_2, dtype=object)
        self.points_tetras_3 = np.array(points_tetras_3, dtype=object)

    def init_Masses(self):
        self.roInitial = np.ones(self.Ntr)
        self.roInitial[self.foilInd] = self.roMetalStart
        self.roInitial[self.waterInd] = self.roWaterStart
        vol = volume(self.points[self.tetras[:, 0]], self.points[self.tetras[:, 1]],
                     self.points[self.tetras[:, 2]], self.points[self.tetras[:, 3]])
        foil_vol = np.sum(vol[self.foilInd])
        self.tetras_mass = self.roInitial * vol
        poin_masses = np.zeros(len(self.points))
        for i, trlist0, trlist1, trlist2, trlist3 in zip(np.arange(self.Np), self.points_tetras_0,
                                                         self.points_tetras_1, self.points_tetras_2,
                                                         self.points_tetras_3):
            m = 0
            for tr in trlist0:
                m += self.tetras_mass[tr]
            for tr in trlist1:
                m += self.tetras_mass[tr]
            for tr in trlist2:
                m += self.tetras_mass[tr]
            for tr in trlist3:
                m += self.tetras_mass[tr]
            poin_masses[i] = m / 4
        self.point_mass = poin_masses
        self.ssInitial = sqrt(self.A * self.n / self.roInitial)

    def r_wire(self):
        return np.mean(norm(self.points[self.border_wire], axis=1))

    def h_foil(self):
        # return np.mean(norm(self.points[self.border_wire], axis=1))
        return np.max(self.points[self.border_vert_me, 1])

    def w_foil(self):
        # return np.mean(norm(self.points[self.border_wire], axis=1))
        return np.max(self.points[self.border_hori_me, 0])

    def init_Power(self):
        Time, Power = Read_Time_Power()
        # %%% Power input
        # simulated_part = 1.0 / 4.0
        simulated_part = 1.0 / 8.0
        effisency = 1.0
        plot_time = [np.argwhere(Time > 0)[0, 0], np.argwhere(Time > self.Tmax)[0, 0], ]
        self.Power = Power[plot_time[0]:plot_time[1]]
        self.time_Power = Time[plot_time[0]:plot_time[1]] * 1.0e3
        self.Power_interpolated = np.interp(self.timing, self.time_Power * 1.0e-3, self.Power)
        Power = Power * effisency  #
        Power = Power * (Power > 0)  #
        totMas = np.sum(self.tetras_mass[self.foilInd])
        # % total mass of one wire [g]
        powInp = Power * simulated_part / totMas * 1e6  # % input power [J/microsecond/kg]
        self.eInp = np.interp(self.timing, Time, powInp) * self.dT
        Combustion_file = open('SIM/Combustion.txt')
        q = float(Combustion_file.readline().split('=')[-1].split(' ')[1])
        t1 = float(Combustion_file.readline().split('=')[-1].split(' ')[1])
        t2 = float(Combustion_file.readline().split('=')[-1].split(' ')[1])
        e_comb = q / (t2 - t1) * 1.0e6 * self.dT
        self.eInp = np.where(((self.timing > t1) & (self.timing < t2)), self.eInp + e_comb, self.eInp, )

    def main_loop(self, mm):
        # %%% ------ Force calculation for each triangle ------ %%%
        # pressure in triangle
        tetras_force = tetra_force_pool(self.tetras, self.points,
                                        self.Pressure_Current, self.pool, self.N_nuc)
        point_force = point_force_list_pool(self.points_tetras_0, self.points_tetras_1,
                                            self.points_tetras_2, self.points_tetras_3,
                                            tetras_force, self.pool, self.N_nuc)
        point_force[self.sector_surface_XY, 2] *= 0
        point_force[self.sector_surface_YZ, 0] *= 0
        point_force[self.sector_surface_XZ, 1] *= 0
        point_force[self.border_outer] *= 0
        point_acceleration = point_force / self.point_mass[:, np.newaxis]

        # find velocity of each point
        vNew = self.Velocity_Current + point_acceleration * self.dT

        # find new position of each point
        pNew = self.points + vNew * self.dT

        # %%% New density calculation
        newVolume = Volume_r_short(self.tetras, pNew)
        roNew = self.tetras_mass / newVolume
        # ss = self.ssInitial * power((roNew / self.roInitial), self.gamma)
        # viscosity = Viscosity_short(self.tetras, pNew, self.Velocity_Current, point_acceleration, ss, roNew)
        # % % % New pressure, energy and temperature calculation
        eNew = self.Energy_Current
        presNew = self.Pressure_Current
        tempNew = self.Temperature_Current
        # press_EOS = self.Pressure_Current

        presNew[self.foilInd], tempNew[self.foilInd] = self.Metal_EOS.ERoFindPT_pool(
            eNew[self.foilInd],
            roNew[self.foilInd],
            self.pool, self.N_nuc)
        presNew[self.waterInd], tempNew[self.waterInd] = self.Water_EOS.ERoFindPT_pool(
            eNew[self.waterInd],
            roNew[self.waterInd],
            self.pool, self.N_nuc)
        presNew = (presNew + self.Pressure_Current + self.Pressure_Prev) / 3.0
        eNew -= (presNew + self.Pressure_Current) * (
                1.0 / roNew - 1.0 / self.Ro_Current) * 5.0e7
        eNew[self.foilInd] += self.eInp[mm]*self.cross_metal_coef
        # presNew += viscosity
        # % % % Assigning new values
        # self.points = (pNew + self.points) / 2.0
        self.points = pNew
        # self.Velocity_Current = (vNew + self.Velocity_Current) / 2.0
        self.Velocity_Current = vNew
        # self.Ro_Current = (roNew + self.Ro_Current) / 2.0
        self.Ro_Current = roNew
        # self.Energy_Current = (eNew + self.Energy_Current) / 2.0
        self.Energy_Current = eNew
        # self.Pressure_Current = (presNew + self.Pressure_Current) / 2.0  # * 0 + 1.0e-6
        self.Pressure_Prev = self.Pressure_Current
        self.Pressure_Current = presNew
        self.Temperature_Current = tempNew
        if (mm % self.N_record) == 0:
            self.Regular_Report()
        if (mm % self.N_report) == 0:
            self.Small_report(mm)

    def Regular_Report(self):
        self.points.tofile(self.file_r)
        self.Temperature_Current.tofile(self.file_T)
        self.Pressure_Current.tofile(self.file_P)
        self.Velocity_Current.tofile(self.file_v)
        self.Energy_Current.tofile(self.file_E)
        self.Ro_Current.tofile(self.file_Ro)

    def Small_report_velosity(self, mm=0):
        self.local_finish = time.perf_counter()
        # start = time.perf_counter()
        print(f'Small report {mm}')
        print(f'{int(100.0 * mm / float(self.Ntime))} % finished')

        '''self.colorbar_ro.remove()
        self.colorbar_press.remove()
        self.colorbar_temperature.remove()
        self.colorbar_energy.remove()

        self.plot_ro.remove()
        self.plot_press.remove()
        self.plot_temperature.remove()
        self.plot_energy.remove()'''
        self.plot_v_1.remove()
        self.plot_v_2.remove()
        self.plot_v_3.remove()
        self.current_power_time_point.remove()
        self.current_power_time_point = self.ax[0, 0].plot(self.timing[mm] * 1.0e3, self.Power_interpolated[mm], 'o')[0]
        self.current_power_time_annotation.remove()
        self.current_power_time_annotation = self.ax[0, 0].annotate(
            f'{int(self.timing[mm] * 1.0e3)} ns, {int(self.Power_interpolated[mm])} GW',
            xy=(self.timing[mm] * 1.0e3, self.Power_interpolated[mm]),
            # xytext=(self.timing[mm] * 1.0e3 + 1.0, self.Power_interpolated[mm] + 1.0),
            arrowprops=dict(facecolor='black', shrink=0.05))

        '''triangles_to_show = self.tetras_water
        points_to_show = self.points[self.waterInd_points]
        v_to_show = self.Velocity_Current[self.waterInd_points]
        ro_to_show = self.Ro_Current[self.waterInd]
        ro_to_show = np.where(ro_to_show > 0, ro_to_show, 0)
        press_to_show = self.Pressure_Current[
                            self.waterInd] * 1.0e3  # dB(self.Pressure_Current[self.waterInd], self.Pressure_Current[self.waterInd])
        press_to_show = np.where(press_to_show > 0, press_to_show, 0)
        temp_to_show = self.Temperature_Current[self.waterInd]
        ener_to_show = self.Energy_Current[self.waterInd] * 1.0e-6'''

        # self.ax[0, 1].axis('equal')
        '''self.plot_ro = self.ax[0, 1].tripcolor(points_to_show[:, 0], points_to_show[:, 1], triangles_to_show,
                                               facecolors=ro_to_show, edgecolors='face')
        self.colorbar_ro = self.fig.colorbar(self.plot_ro, ax=self.ax[0, 1], format=self.fmt)

        # self.ax[0, 2].axis('equal')
        self.plot_press = self.ax[0, 2].tripcolor(points_to_show[:, 0], points_to_show[:, 1], triangles_to_show,
                                                  facecolors=press_to_show, edgecolors='face', )
        self.colorbar_press = self.fig.colorbar(self.plot_press, ax=self.ax[0, 2])

        # self.ax[1, 1].axis('equal')
        self.plot_temperature = self.ax[1, 1].tripcolor(points_to_show[:, 0], points_to_show[:, 1], triangles_to_show,
                                                        facecolors=temp_to_show, edgecolors='face')
        self.colorbar_temperature = self.fig.colorbar(self.plot_temperature, ax=self.ax[1, 1])

        # self.ax[1, 2].axis('equal')
        self.plot_energy = self.ax[1, 2].tripcolor(points_to_show[:, 0], points_to_show[:, 1], triangles_to_show,
                                                   facecolors=ener_to_show, edgecolors='face')
        self.colorbar_energy = self.fig.colorbar(self.plot_energy, ax=self.ax[1, 2], format=self.fmt)
        '''
        # self.ax[1, 0].axis('equal')
        self.ax[0, 1].axis('equal')

        self.plot_v_1 = self.ax[0, 1].quiver(self.points[self.sector_surface_water_XY, 0],
                                             self.points[self.sector_surface_water_XY, 1],
                                             self.Velocity_Current[self.sector_surface_water_XY, 0],
                                             self.Velocity_Current[self.sector_surface_water_XY, 1])
        self.ax[1, 0].axis('equal')

        self.plot_v_2 = self.ax[1, 0].quiver(
            self.points[self.sector_surface_water_2, 1],
            self.points[self.sector_surface_water_2, 2],

            self.Velocity_Current[self.sector_surface_water_2, 1],
            self.Velocity_Current[self.sector_surface_water_2, 2])
        self.ax[1, 1].axis('equal')
        self.plot_v_3 = self.ax[1, 1].quiver(self.points[self.sector_surface_water_XZ, 0],

                                             self.points[self.sector_surface_water_XZ, 2],
                                             self.Velocity_Current[self.sector_surface_water_XZ, 0],

                                             self.Velocity_Current[self.sector_surface_water_XZ, 2])

        self.fig.tight_layout()
        self.fig.savefig(f'Report {mm}.png')
        # print(f'w_foil = {self.w_foil() * 2.0e4} mkm')
        # print(f'h_foil = {self.h_foil() * 2.0e4} mkm')
        # print(f'r_wire = {self.r_wire()} cm')
        try:
            print(f't = {int(self.local_finish - self.local_start)} sec')
        except:
            pass
        self.local_start = time.perf_counter()

        self.n_recorded = 0
        # print(f'time for report {time.perf_counter() - start}')

    def Small_report(self, mm=0):
        self.local_finish = time.perf_counter()
        # start = time.perf_counter()
        print(f'Small report {mm}')
        print(f'{int(100.0 * mm / float(self.Ntime))} % finished')

        self.colorbar_ro_XY.remove()
        self.colorbar_ro_YZ.remove()
        self.colorbar_ro_XZ.remove()

        self.plot_ro_XY.remove()
        self.plot_ro_YZ.remove()
        self.plot_ro_XZ.remove()

        self.current_power_time_point.remove()
        self.current_power_time_point = \
            self.ax[0, 0].plot(self.timing[mm] * 1.0e3, self.Power_interpolated[mm], 'o')[0]
        self.current_power_time_annotation.remove()
        self.current_power_time_annotation = self.ax[0, 0].annotate(
            f'{int(self.timing[mm] * 1.0e3)} ns, {int(self.Power_interpolated[mm])} GW',
            xy=(self.timing[mm] * 1.0e3, self.Power_interpolated[mm]),
            # xytext=(self.timing[mm] * 1.0e3 + 1.0, self.Power_interpolated[mm] + 1.0),
            arrowprops=dict(facecolor='black', shrink=0.05))
        self.ax[0, 1].axis('equal')
        self.plot_ro_XY = self.ax[0, 1].tripcolor(self.points[self.sector_surface_water_XY, 0],
                                                  self.points[self.sector_surface_water_XY, 1], self.triangle_XY,
                                                  facecolors=self.Ro_Current[self.tetra_XY],
                                                  edgecolors='face')
        self.colorbar_ro_XY = self.fig.colorbar(self.plot_ro_XY, ax=self.ax[0, 1], format=self.fmt)

        self.ax[1, 0].axis('equal')

        self.plot_ro_YZ = self.ax[1, 0].tripcolor(self.points[self.sector_surface_water_YZ, 1],
                                                  self.points[self.sector_surface_water_YZ, 2], self.triangle_YZ,
                                                  facecolors=self.Ro_Current[self.tetra_YZ],
                                                  edgecolors='face')
        self.colorbar_ro_YZ = self.fig.colorbar(self.plot_ro_YZ, ax=self.ax[1, 0], format=self.fmt)
        self.ax[1, 1].axis('equal')
        self.plot_ro_XZ = self.ax[1, 1].tripcolor(self.points[self.sector_surface_water_XZ, 0],
                                                  self.points[self.sector_surface_water_XZ, 2], self.triangle_XZ,
                                                  facecolors=self.Ro_Current[self.tetra_XZ],
                                                  edgecolors='face')
        self.colorbar_ro_XZ = self.fig.colorbar(self.plot_ro_XZ, ax=self.ax[1, 1], format=self.fmt)

        self.fig.tight_layout()
        self.fig.savefig(f'Report {mm}.png')

        try:
            print(f't = {int(self.local_finish - self.local_start)} sec')
        except:
            pass
        self.local_start = time.perf_counter()

        self.n_recorded = 0
        # print(f'time for report {time.perf_counter() - start}')

    def main_loop_processing(self):
        # self.Cupper_EOS.Stop_processes()
        # self.Water_EOS.Stop_processes()
        print('simulation started')
        start = time.perf_counter()
        # self.prereport()
        # self.Small_report(0)
        self.local_start = time.perf_counter()
        for mm in range(self.Ntime):
            try:
                self.main_loop(mm)

            except Exception as e:
                traceback.print_exc()
                break
        print(f'Simulation took {time.perf_counter() - start} sec')

    def main_report(self):
        self.timing[::self.N_record].tofile("BIN/my_time.bin")
        self.Power_interpolated[::self.N_record].tofile("BIN/my_power.bin")
        '''np.savetxt('Mesh/Mesh_points.csv', self.points)
        np.savetxt('Mesh/Mesh_elements.csv', self.tetras, )
        np.savetxt('Mesh/Mesh_markers.csv', self.tetra_marker, )
        np.savetxt('Mesh/Mesh_1_sector_surface.csv', self.sector_surface_XY)
        np.savetxt('Mesh/Mesh_1_sector_surface_water.csv', self.sector_surface_water_XY)
        np.savetxt('Mesh/Mesh_1_sector_fringe_water.csv', self.sector_fringe_water_X)
        np.savetxt('Mesh/Mesh_2_sector_surface.csv', self.sector_surface_YZ)
        np.savetxt('Mesh/Mesh_2_sector_surface_water.csv', self.sector_surface_water_2)
        np.savetxt('Mesh/Mesh_2_sector_fringe_water.csv', self.sector_fringe_water_2)
        np.savetxt('Mesh/Mesh_3_sector_surface.csv', self.sector_surface_XZ)
        np.savetxt('Mesh/Mesh_3_sector_surface_water.csv', self.sector_surface_water_XZ)
        np.savetxt('Mesh/Mesh_3_sector_fringe_water.csv', self.sector_fringe_water_Z)
        np.savetxt('Mesh/Mesh_outer_border.csv', self.border_outer, )
        np.savetxt('Mesh/Mesh_water_points.csv', self.waterInd_points, )'''

    def plot_array_on_mesh(self, arr, text, numer, time):
        try:
            os.chdir(f'{text}')
        except:
            os.mkdir(f'{text}')
            os.chdir(f'{text}')
        # os.chdir('Mesh')
        plt.clf()

        tpc = plt.tripcolor(self.points[:, 0], self.points[:, 1], self.tetras, facecolors=arr, edgecolors='face')
        plt.colorbar(tpc)
        plt.title(f'{text} {time} ns')
        plt.axis('equal')
        plt.savefig(f'i{text}_{time}_ns.png')

        # plt.show()
        plt.clf()

        os.chdir('..')

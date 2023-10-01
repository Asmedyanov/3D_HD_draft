from readKBT import readKB
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import multiprocessing as mp
import weakref
import pickle
import time
from multiprocessing import shared_memory
from scipy.signal import find_peaks

global q
q = mp.Queue()
global q_out_array
q_out_array = []


def find_left_right_pic(value, array):
    test_array = np.abs(array - value)
    # plt.clf()
    # plt.plot(np.log(test_array))
    pics = find_peaks(-test_array)[0]
    # plt.plot(np.arange(array.size)[pics], np.log(test_array)[pics], 'o')
    # plt.show()
    # n = np.argmin(test_array)
    n = np.arange(array.size)[pics[-1]]
    # if (value == array[n]) | (n == 0):
    # return n, n+1
    if (n >= (len(array) - 1)):
        return n - 1, n
    # n_right = n
    # n_left = n - 1
    # if (value > array[n]) & (value > array[n - 1]):
    # n_right = n + 1
    # n_left = n
    # return n_left, n_right
    return n, n + 1


def find_left_right(value, array):
    test_array = np.abs(array - value)
    # plt.clf()
    # plt.plot(test_array)
    # peak_ind = find_peaks(-test_array)[0]
    # peak_values = test_array[peak_ind]
    # plt.plot(peak_ind,peak_values,'or')
    # plt.show()
    n = np.argmin(test_array)
    if (n >= (len(array) - 1)):
        return n - 1, n
    return n, n + 1


def line_from_2_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b


def my_distance(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2


def my_distance2(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


class EOS:
    """Equation of state"""

    def __init__(self, filename):
        """
        Reading Equation of state file
        :param filename: Equation of state file name
        """
        # Rho_table, self.Temp_table, self.Press_table, self.Ener_table = readKB(filename)
        Rho_table, Temp_table, Press_table, Ener_table = readKB(filename)
        N_mult = 3
        func_press = interpolate.interp2d(Temp_table, Rho_table, Press_table, kind='cubic')
        func_energy = interpolate.interp2d(Temp_table, Rho_table, Ener_table, kind='cubic')
        func_rho = interpolate.interp1d(np.arange(len(Rho_table)), Rho_table, kind='cubic')
        func_temp = interpolate.interp1d(np.arange(len(Temp_table)), Temp_table, kind='cubic')
        self.Rho_table = func_rho(np.arange(0, len(Rho_table) - 1, 1.0 / N_mult))
        self.Temp_table = func_temp(np.arange(0, len(Temp_table) - 1, 1.0 / N_mult))
        self.Press_table = func_press(self.Temp_table, self.Rho_table, )
        self.Ener_table = func_energy(self.Temp_table, self.Rho_table, )

        '''plt.clf()
        plt.imshow(self.Ener_table)
        plt.colorbar()
        plt.savefig(f"Internal_energy_{filename.split('.')[0]}.png")
        plt.clf()
        plt.imshow(self.Press_table)
        plt.colorbar()
        plt.savefig(f"Pressure_{filename.split('.')[0]}.png")
        plt.clf()'''

        self.ERoFindPT_vector = np.vectorize(self.ERoFindPT)
        self.T_min = 280.0
        self.full_array_len = 0
        self.n_proc = 0
        self.my_n_proc = 8
        print('Normal (300 K, 1 bar) conditions:\n')
        p_norm = 1.0e-6
        t_norm = 300.0
        self.e_norm, self.rho_norm = self.FromPTtoERo(p_norm, t_norm)
        # p_norm, t_norm = self.ERoFindPT(e_norm, rho_norm)
        print(f'E = {self.e_norm}, Ro = {self.rho_norm}\n')
        # print(f'P = {p_norm}, T = {t_norm}\n')

    def start_processes(self, Nelem):
        self.full_array_len = Nelem
        self.start = [i * self.full_array_len // self.my_n_proc for i in range(self.my_n_proc)]
        self.finish = [(i + 1) * self.full_array_len // self.my_n_proc for i in range(self.my_n_proc)]
        self.finish[-1] = self.full_array_len
        self.my_E = []
        self.my_Ro = []
        self.my_P = [np.zeros(2) for i in range(self.my_n_proc)]
        self.my_T = [np.zeros(2) for i in range(self.my_n_proc)]
        self.flag_ready = np.zeros(self.my_n_proc)
        self.flag_off = np.zeros(self.my_n_proc)
        global q
        global q_out_array

        q = mp.Queue()
        # self.q = q
        for i in range(self.my_n_proc):
            q.put(self)
            q_out_array.append(mp.Queue())
        # self.manager = mp.Manager()
        # self.queues = [self.manager.Queue() for i in range(self.my_n_proc)]
        # for q in self.queues:
        #    q.put(self)
        # self.my_pool = mp.Pool(processes=self.my_n_proc)
        # shm = shared_memory.SharedMemory(create=True, size=self.flag_ready.nbytes)
        # self.flag_ready = np.ndarray(self.flag_ready.shape, buffer=shm.buf)
        # self.flag_ready[:] = np.zeros(self.my_n_proc)
        # self.my_Processes = [self.my_pool.apply_async(self.Process_func, args=(self.queues[i], i,)) for i in
        #                     range(self.my_n_proc)]
        self.my_Processes = [mp.Process(target=self.Process_func, args=(q, q_out_array[i], i,)) for i in
                             range(self.my_n_proc)]
        for proc in self.my_Processes:
            # pickle.dumps(proc._config['authkey'])
            proc.start()
        # for proc in self.my_Processes:
        # proc.join()

    def Process_func(self, q, q_out, i: int):
        # my_EOS = q.get()
        while True:
            # print(f'queue is empty {q.empty()}')
            if not q.empty():
                my_EOS = q.get()
                # print(f'process {i} works')
                if len(my_EOS.my_E) != 0:
                    P, T = my_EOS.ERoFindPT_vector(my_EOS.my_E[i], my_EOS.my_Ro[i])
                    q_out.put([P, T])
                    # my_EOS.my_P[i] = P
                    # my_EOS.my_T[i] = T
                if my_EOS.flag_off[i] == 1:
                    print(f'switch off process {i}')
                    break

    def ERoFindPT_mp_perm(self, E: np.array, Ro: np.array):
        # print(f'I started {self.my_n_proc} processes')
        global q
        self.set_my_E(E)
        self.set_my_Ro(Ro)
        # self.flag_ready[:] = np.ones(self.my_n_proc, dtype='int')
        for i in range(self.my_n_proc):
            q.put(self)
        # print(f'I started {self.my_n_proc} processes')
        while not q.empty():
            pass
            # print(f'queue is empty {q.empty()}')
            # print(f'I wait {self.my_n_proc} processes')
            # time.sleep(1)
        for i in range(self.my_n_proc):
            self.my_P[i], self.my_T[i], = q_out_array[i].get()
        # print(f'{self.my_n_proc} processes finished')
        P = self.get_my_P()
        T = self.get_my_T()
        return P, T

    def Stop_processes(self):
        self.flag_off = np.ones(self.my_n_proc)

    def set_my_E(self, E: np.array):
        self.my_E = [E[self.start[i]:self.finish[i]] for i in range(self.my_n_proc)]
        pass

    def set_my_Ro(self, Ro):
        self.my_Ro = [Ro[self.start[i]:self.finish[i]] for i in range(self.my_n_proc)]

    def get_my_P(self):
        ret_P = self.my_P[0]
        for P in self.my_P[1:]:
            ret_P = np.concatenate([ret_P, P])
        return ret_P

    def get_my_T(self):
        ret_T = self.my_T[0]
        for T in self.my_T[1:]:
            ret_T = np.concatenate([ret_T, T])
        return ret_T

    def PT_Bingo_Bingo(self, E, Ro, i_bingo, j_bingo):
        """
        Scalar linear solution of EOS
        Pressure [Mbar] and temperature [K] from energy [J/kg] and density [g/cm^3]
        if point absolutly in initial file
        :param E: energy [J/kg]
        :param Ro: density [g/cm^3]
        :param i_bingo: index of density in the EOS file
        :param j_bingo: index of temperature in the EOS file
        :return: Pressure [Mbar] and temperature [K]
        """
        return self.Press_table[i_bingo, j_bingo], self.Temp_table[j_bingo]

    def PT_Bingo(self, E, Ro, i_bingo):
        """
        Scalar linear solution of EOS
        Pressure [Mbar] and temperature [K] from energy [J/kg] and density [g/cm^3]
        if point density in initial file
        :param E: energy [J/kg]
        :param Ro: density [g/cm^3]
        :param i_bingo: index of density in the EOS file
        :return: Pressure [Mbar] and temperature [K]
        """
        E_bingo = self.Ener_table[i_bingo]
        P_bingo = self.Press_table[i_bingo]
        j_e_bingo_left, j_e_bingo_right = find_left_right(E, E_bingo)
        if j_e_bingo_left == j_e_bingo_right:
            return self.PT_Bingo_Bingo(E, Ro, i_bingo, j_e_bingo_right)
        E_bingo_left = E_bingo[j_e_bingo_left]
        E_bingo_right = E_bingo[j_e_bingo_right]
        P_bingo_left = P_bingo[j_e_bingo_left]
        P_bingo_right = P_bingo[j_e_bingo_right]
        T_bingo_left = self.Temp_table[j_e_bingo_left]
        T_bingo_right = self.Temp_table[j_e_bingo_right]
        point1 = [T_bingo_left, E_bingo_left]
        point2 = [T_bingo_right, E_bingo_right]
        a, b = line_from_2_points(point1, point2)
        T_output = (E - b) / a
        point1 = [T_bingo_left, P_bingo_left]
        point2 = [T_bingo_right, P_bingo_right]
        a, b = line_from_2_points(point1, point2)
        P_output = a * T_output + b
        if P_output == 0:
            pass
        return P_output, T_output

    def ERoFindPT(self, E, Ro):
        """
        Scalar linear solution of EOS
        Pressure [Mbar] and temperature [K] from energy [J/kg] and density [g/cm^3]
        :param E: energy [J/kg]
        :param Ro: density [g/cm^3]
        :return: Pressure [Mbar] and temperature [K]
        """
        i_ro_left, i_ro_right = find_left_right(Ro, self.Rho_table)

        if i_ro_left == i_ro_right:
            return self.PT_Bingo(E, Ro, i_ro_right)
        Ro_left = self.Rho_table[i_ro_left]
        Ro_right = self.Rho_table[i_ro_right]
        E_up = self.Ener_table[i_ro_left]
        E_down = self.Ener_table[i_ro_right]
        P_up = self.Press_table[i_ro_left]
        P_down = self.Press_table[i_ro_right]
        j_e_left_down, j_e_right_down = find_left_right(E, E_down)
        j_e_left_up, j_e_right_up = find_left_right(E, E_up)
        E_left_up = E_up[j_e_left_up]
        E_left_down = E_down[j_e_left_down]
        E_right_up = E_up[j_e_right_up]
        E_right_down = E_down[j_e_right_down]

        P_left_up = P_up[j_e_left_up]
        P_left_down = P_down[j_e_left_down]
        P_right_up = P_up[j_e_right_up]
        P_right_down = P_down[j_e_right_down]

        T_left_up = self.Temp_table[j_e_left_up]
        T_left_down = self.Temp_table[j_e_left_down]
        T_right_up = self.Temp_table[j_e_right_up]
        T_right_down = self.Temp_table[j_e_right_down]
        set_of_points = [
            [Ro_left, T_left_up, E_left_up],
            [Ro_right, T_left_down, E_left_down],
            [Ro_left, T_right_up, E_right_up],
            [Ro_right, T_right_down, E_right_down]
        ]
        # print(len(set_of_points))
        exlude_index = 0
        max_dist = 0
        for i, point in enumerate(set_of_points):
            point1 = [Ro, 300.0, E]
            dist = my_distance(point, point1)
            if dist > max_dist:
                max_dist = dist
                exlude_index = i
        set_of_points = [set_of_points[i] for i in range(len(set_of_points)) if i != exlude_index]
        set_of_points = np.array(set_of_points)
        # the cross product is a vector normal to the plane
        p1, p2, p3 = set_of_points[[0, 1, 2]]
        # These two vectors are in the plane
        v1 = p3 - p1
        v2 = p2 - p1

        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        a, b, c = cp
        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(cp, p3)
        T = (d - a * Ro - c * E) / b
        '''if (T < self.T_min) | (T > 1.0e5):
            T = self.T_min'''
        T_output = T
        set_of_points = [
            [Ro_left, T_left_up, P_left_up],
            [Ro_right, T_left_down, P_left_down],
            [Ro_left, T_right_up, P_right_up],
            [Ro_right, T_right_down, P_right_down]
        ]
        '''for mylist in set_of_points:
            if ((np.abs((Ro - mylist[0]) / Ro) <= 0.1) & (np.abs((T_output - mylist[1]) / T_output) <= 0.1)):
                return mylist[2], T_output'''
        exlude_index = 0
        max_dist = 0
        for i, point in enumerate(set_of_points):
            point1 = [Ro, T_output, 1.0e-6]
            dist = my_distance2(point, point1)
            if dist > max_dist:
                max_dist = dist
                exlude_index = i
        set_of_points = [set_of_points[i] for i in range(len(set_of_points)) if i != exlude_index]
        set_of_points = np.array(set_of_points)
        p1, p2, p3 = set_of_points[[0, 1, 2]]
        # These two vectors are in the plane
        v1 = p3 - p1
        v2 = p2 - p1
        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        a, b, c = cp
        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(cp, p3)
        P = (d - a * Ro - b * T_output) / c
        '''if (P < min([P_left_up, P_left_down, P_right_up, P_left_down])):
            P = min([P_left_up, P_left_down, P_right_up, P_left_down])'''
        P_output = P
        return P_output, T_output

    def FromPTtoERo(self, P, T):
        """
        Scalar linear solution of EOS
        energy [J/kg] and density [g/cm^3] from Pressure [Mbar] and temperature [K]
        :param P: energy [J/kg]
        :param T: temperature [K]
        :return: energy [J/kg] and density [g/cm^3]
        """
        i_T_left, i_T_right = find_left_right_pic(T, self.Temp_table)
        # i_T_left += 1
        # i_T_right += 1
        if i_T_left == i_T_right:
            return self.ERo_Bingo(P, T, i_T_right)
        T_left = self.Temp_table[i_T_left]
        T_right = self.Temp_table[i_T_right]
        E_up = self.Ener_table[:, i_T_left]
        E_down = self.Ener_table[:, i_T_right]
        P_up = self.Press_table[:, i_T_left]
        P_down = self.Press_table[:, i_T_right]
        j_P_left_down, j_P_right_down = find_left_right_pic(P, P_down)
        j_P_left_up, j_P_right_up = find_left_right_pic(P, P_up)
        E_left_up = E_up[j_P_left_up]
        E_left_down = E_down[j_P_left_down]
        E_right_up = E_up[j_P_right_up]
        E_right_down = E_down[j_P_right_down]

        P_left_up = P_up[j_P_left_up]
        P_left_down = P_down[j_P_left_down]
        P_right_up = P_up[j_P_right_up]
        P_right_down = P_down[j_P_right_down]

        Ro_left_up = self.Rho_table[j_P_left_up]
        Ro_left_down = self.Rho_table[j_P_left_down]
        Ro_right_up = self.Rho_table[j_P_right_up]
        Ro_right_down = self.Rho_table[j_P_right_down]

        set_of_points = [
            [T_left, Ro_left_up, P_left_up],
            [T_right, Ro_left_down, P_left_down],
            [T_left, Ro_right_up, P_right_up],
            [T_right, Ro_right_down, P_right_down]
        ]
        exlude_index = 0
        max_dist = 0
        for i, point in enumerate(set_of_points):
            point1 = [T, 1, P]
            dist = my_distance(point, point1)
            if dist > max_dist:
                max_dist = dist
                exlude_index = i
        set_of_points = [set_of_points[i] for i in range(len(set_of_points)) if i != exlude_index]
        set_of_points = np.array(set_of_points)
        p1, p2, p3 = set_of_points[[0, 1, 2]]
        # These two vectors are in the plane
        v1 = p3 - p1
        v2 = p2 - p1
        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        a, b, c = cp
        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(cp, p3)
        Ro = (d - a * T - c * P) / b

        set_of_points = [
            [T_left, Ro_left_up, E_left_up],
            [T_right, Ro_left_down, E_left_down],
            [T_left, Ro_right_up, E_right_up],
            [T_right, Ro_right_down, E_right_down]
        ]
        exlude_index = 0
        max_dist = 0
        for i, point in enumerate(set_of_points):
            point1 = [T, Ro, 1]
            dist = my_distance2(point, point1)
            if dist > max_dist:
                max_dist = dist
                exlude_index = i
        set_of_points = [set_of_points[i] for i in range(len(set_of_points)) if i != exlude_index]
        set_of_points = np.array(set_of_points)
        p1, p2, p3 = set_of_points[[0, 1, 2]]
        # These two vectors are in the plane
        v1 = p3 - p1
        v2 = p2 - p1
        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        a, b, c = cp
        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(cp, p3)
        E = (d - a * T - b * Ro) / c

        return E, Ro

    def ERoFindPT_pool(self, E: np.array, Ro: np.array, pool, n_proc: int):
        if ((self.full_array_len == 0) | (n_proc != self.n_proc)):
            self.n_proc = n_proc
            self.full_array_len = E.size
            self.start = [i * self.full_array_len // n_proc for i in range(n_proc)]
            self.finish = [(i + 1) * self.full_array_len // n_proc for i in range(n_proc)]
            self.finish[-1] = self.full_array_len
        E_splited = np.array_split(E, n_proc)
        Ro_splited = np.array_split(Ro, n_proc)
        proc_array = [pool.apply_async(self.ERoFindPT_vector, args=(E_, Ro_)) for E_, Ro_ in zip(E_splited, Ro_splited)]
        P, T = proc_array[0].get()
        for proc in proc_array[1:]:
            tempP, tempT = proc.get()
            T = np.concatenate([T, tempT])
            P = np.concatenate([P, tempP])
        return P, T

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        try:
            # del self_dict['my_pool']
            del self_dict['my_Processes']
            # del self_dict['manager']
            # del self_dict['queues']
        except:
            pass
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

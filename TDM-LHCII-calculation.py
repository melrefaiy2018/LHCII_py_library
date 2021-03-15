import numpy as np
import pandas as pd
import xlsxwriter
import MesoBio.PDB as PDB
from MesoBio.cg_particles import CGParticle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def prepare_lhcii(PDB_id, PDB_path):
    """

    :param PDB_id:
    :type PDB_id:
    :param PDB_path:
    :type PDB_path:
    :return:
    :rtype:
    """
    lhcii = PDB.PDBParser(QUIET=True).get_structure(PDB_id, PDB_path)
    #https://lists.open-bio.org/pipermail/biopython/2014-July/015371.html
    lhcii._name = 'LHCII'
    list_chain_keep = ['J', 'D', 'I']
    list_chain_all = [chain.id for chain in lhcii[0].child_list]

    for id in list_chain_all:
        # print(id)
        if id not in list_chain_keep:
            lhcii[0].detach_child(id)

    chl_list = []
    for res in lhcii.get_residues():
        if res.get_resname() == 'CLA' or res.get_resname() == 'CHL':
            chl_list.append(res)
            res.prepare_chlorophyll()

    lhcii[0].list_pigments = chl_list
    lhcii[0].dict_pigments = {chl._name : chl for chl in chl_list}
    lhcii[0].list_names = [chl._name for chl in chl_list]
    lhcii[0].transform(np.eye(3), -lhcii.center_of_mass)
    lhcii[0].flatten()
    return lhcii

def calc_QyQx_tdm_vectors(dict_tdm):
    """
    This function gets TDM position and magnitude
    :param dict_tdm:
    :return:
    """
    TDM_chl_qx_vectors = [ ]  # TDM vectors
    TDM_chl_qy_vectors = [ ]  # TDM vectors
    calc_qx_tdm_vectors = [ ]  # TDM vectors
    calc_qy_tdm_vectors = [ ]  # TDM vectors
    location_vectors = [ ]    # location vectors
    # to extract certain keys from dictionary:
    # ----------------------------------------
    for key, value in dict_tdm.items():
        search_key = "CHL"
        if search_key in key[4]:           # this condition does not satisfy the current condition, need to be fixed
            tdm_magnitude_vec = np.sqrt(
                (value.vector[ 0 ]) ** 2 + (value.vector[ 1 ]) ** 2 + (
                    value.vector[ 2 ]) ** 2)
            tdm_unit_vec = np.array(
                [ value.vector[ 0 ] / tdm_magnitude_vec,
                  value.vector[ 1 ] / tdm_magnitude_vec,
                  value.vector[ 2 ] / tdm_magnitude_vec ])

            qx = tdm_unit_vec * 2.61
            qy = tdm_unit_vec * 3.19
            TDM_chl_qx_vectors.append(qx)
            TDM_chl_qy_vectors.append(qy)
        else:
            tdm_magnitude_vec = np.sqrt(
                (value.vector[ 0 ]) ** 2 + (value.vector[ 1 ]) ** 2 + (
                    value.vector[ 2 ]) ** 2)
            tdm_unit_vec = np.array(
                [ value.vector[ 0 ] / tdm_magnitude_vec,
                  value.vector[ 1 ] / tdm_magnitude_vec,
                  value.vector[ 2 ] / tdm_magnitude_vec])
            qx = tdm_unit_vec * 3.07
            qy = tdm_unit_vec * 3.74
            calc_qx_tdm_vectors.append(qx)
            calc_qy_tdm_vectors.append(qy)
        qx_tdm_vectors = TDM_chl_qx_vectors + calc_qx_tdm_vectors  # nx1 dimensions
        qy_tdm_vectors = TDM_chl_qy_vectors + calc_qy_tdm_vectors  # nx1 dimensions
        location_vectors.append(value.location)
    return qx_tdm_vectors, qy_tdm_vectors, location_vectors


def gen_distanceMatrix(loc_vec):
    """
    This function extracts the location vectors from the TDM dictionary. Then, it calculates the distance and magnitude
    between each pair of pigments.
    """
    location_vec = np.array(loc_vec)  # n x 1 dim.
    distance_matrix = location_vec.reshape(len(location_vec), 1, 3) - location_vec  # Vector matrix describing the
    # center-to-center difference in position of the two pigments
    distance_matrix_mag = np.sqrt((distance_matrix ** 2).sum(2))
    distance_matrix_magnitude = distance_matrix_mag.reshape(len(loc_vec), len(loc_vec),
                                                            1)  # reshape the matrix to be (n x n x 1) dimension
    # replace the diagonal by inf

    distance_UnitMatrix = distance_matrix / distance_matrix_magnitude  # The Unit Vector matrix describing the
    # center-to-center difference in position of the two pigments
    Distance_UnitMatrix = np.where(np.isnan(distance_UnitMatrix), 0,
                                   distance_UnitMatrix)  # to remove "nan" errors from the matrix
    return Distance_UnitMatrix, distance_matrix_magnitude


def gen_tdm_matrix(TDM_vectors):
    """
    This function is used to build the matrix for all the TDM and calculate the magnitude of each TDM vector
    Param:
    ------
    1. v1: is the collection of all TDM vectors

    """
    v1 = np.array(TDM_vectors)
    TDM_matrix = v1.reshape(len(v1), 1, 3)  # build (n x 1 x 3) TDM matrix
    TDM_matrix_symetric = np.tile(TDM_matrix, (len(TDM_matrix), 1))  # build (n x n x 3) TDM matrix
    TDM_matrix_symetric2 = np.array([ list(i) for i in zip(*TDM_matrix_symetric) ])  # transpose the matrix
    return TDM_matrix_symetric, TDM_matrix_symetric2


def calc_dipole_interaction(P1, P2, r_unit_vec, r):
    # Define constants
    # ----------------
    epsilon_0 = 1.5812E-5  # (Units: D^2/[cm^-1*Angstrom^3])
    epsilon_eff = 1  # (Units: unitless)
    # Calculate the dipole-dipole coupling between pairs
    # --------------------------------------------------
    pre_eps = (epsilon_eff * epsilon_0 * 4 * np.pi)
    numerator = (np.array(P1 * P2).sum(2)) - (np.array(3 * P1 * r_unit_vec).sum(2)) * (np.array(P2 * r_unit_vec).sum(2))
    denominator = pre_eps * np.array(r ** 3).sum(2)
    return numerator / denominator


def run_calc(input):
    """
    This function is used to get the TDM vectors and then calculates the distance matrix that represent the distance
    between each pair of CHL in the model and its magnitude values
    :param input:
    :type input:
    :return:
    :rtype:
    """
    TDM_qx_vectors, TDM_qy_vectors, position_vectors = calc_QyQx_tdm_vectors(input)
    distance_matrix, distance_matrix_magnitude = gen_distanceMatrix(position_vectors)
    # TDM_qx_matrix, TDM_qx_matrix2 = gen_tdm_matrix(np.array(TDM_qx_vectors))
    TDM_qy_matrix, TDM_qy_matrix2 = gen_tdm_matrix(np.array(TDM_qy_vectors))
    U = calc_dipole_interaction(TDM_qy_matrix, TDM_qy_matrix2, distance_matrix, distance_matrix_magnitude)
    return np.where(np.isinf(U), 0, U)  # to remove all the inf values and replaced by zero

def export_data2excel(path,data):
    """

    Parameter:
    ----------
    :param path:
    :type path:
    :param data:
    :type data:
    :return:
    :rtype:
    """
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet()
    row = 0
    for col, data in enumerate(data):
        worksheet.write_column(row, col, data)
    workbook.close()

def gen_chl_states(Dict_sorted_tdm):
    """
    This function generates all the different combinations between different CHL pigments. ex: CHL601-CHL602

    Parameter:
    ----------
    1. Dict_sorted_tdm: Dict that contains all the TDM information
                        Dict

    return:
    -------
    1. chl_states : list that contains all the coupling states between all the CHL pigments
                    list
    """
    list_keys = []
    for key, value in Dict_sorted_tdm.items():
        list_keys.append(key)
    # print(list_keys)
    chl_index = []
    for i in enumerate(list_keys):
        chl_index.append(i)
    dict_chl_index = dict(chl_index)
    # print(dict_chl_index)
    chl_states = []
    for i in (dict_chl_index.items()):
        for j in (dict_chl_index.items()):
            chl_states.append((i[1], j[1]))
    return chl_states

def gen_chl_indexing(dict_tdm, chl_matrix, chl_states):
    """
    This function will match item in the CHL states list to its value in the coupling matrix

    Parameter:
    ----------

    1. dict_tdm :   Dict that contains all TDM values
                    Dict

    2. chl_matrix : Coupling matrix between all the CHL pigments
                   np.matrix

    3. chl_states :
                   list

    :return:

    1. dict_chl_indexing : Dict that will have all the values of the coupling matrix indexed to the CHL states list
                           Dict

    """
    coupling = []
    for i in range(len(dict_tdm)):
        for j in range(len(dict_tdm)):
            coupling.append(chl_matrix[i, j])
    dict_chl_indexing = dict(zip(chl_states, coupling))
    return dict_chl_indexing

def update_tdm_dict(dict_tdm_list):
    """
    This function extract the TDM values from each LHCII model and update the main dict_tdm
    Param:
    ------
    1 . dict_tdm_list : this list contains all the
                       list
    return:
    -------
    1. dict_tdm : Dict contains all TDM values for all the CGP objects that are in the model you calculating coupling for
                  Dict
    """
    dict_tdm = {}
    for index in range(len(dict_tdm_list)):
        dict_tdm.update(dict_tdm_list[index])
    return dict_tdm
########################################################################################################################
# Running calculation:
# -------------------
lhcii = prepare_lhcii('1rwt','/Users/48107674/Box/Reseach/2020/research/LHCII/rw/1rwt.pdb')
lhcii.flatten()
# Building Coarse-grained particles:
# ------------------------------
lhcii_0 = CGParticle(lhcii, name="main PDB")
lhcii_1 = CGParticle(lhcii, [100,100,0], name="CGP1")
# lhcii_2 = CGParticle(lhcii, [100,0,0], name="CGP2")
# lhcii_3 = CGParticle(lhcii, [-100,0,0], name="CGP3")
# lhcii_4 = CGParticle(lhcii, [0,100,0], name="CGP4")
# lhcii_5 = CGParticle(lhcii, [0,-100,0], name="CGP5")
# lhcii_6 = CGParticle(lhcii, [-100,-100,0], name="CGP6")
# lhcii_7 = CGParticle(lhcii, [100,-100,0], name="CGP7")
# lhcii_8 = CGParticle(lhcii, [-100,100,0], name="CGP8")
# Visualize CGP:
# --------------
# fig = plt.figure()
# ax = Axes3D(fig)
# lhcii.visualize_data(ax, 'mg_pos')
# lhcii_1.visualize_data(ax, 'mg_pos')
# lhcii_2.visualize_data(ax, 'mg_pos')
# lhcii_3.visualize_data(ax, 'mg_pos')
# lhcii_4.visualize_data(ax, 'mg_pos')
# lhcii_5.visualize_data(ax, 'mg_pos')
# lhcii_6.visualize_data(ax, 'mg_pos')
# lhcii_7.visualize_data(ax, 'mg_pos')
# lhcii_8.visualize_data(ax, 'mg_pos')
#
# fig2 = plt.figure()
# ax = Axes3D(fig2)
# lhcii.visualize_data(ax, 'qy_tdm')
# lhcii_1.visualize_data(ax, 'qy_tdm')
# lhcii_2.visualize_data(ax, 'qy_tdm')
# lhcii_3.visualize_data(ax, 'qy_tdm')
# lhcii_4.visualize_data(ax, 'qy_tdm')
# lhcii_5.visualize_data(ax, 'qy_tdm')
# lhcii_6.visualize_data(ax, 'qy_tdm')
# lhcii_7.visualize_data(ax, 'qy_tdm')
# lhcii_8.visualize_data(ax, 'qy_tdm')

# Extract TDM :
# -------------
dict0_tdm = lhcii_0.get_data('qy_tdm')
dict1_tdm = lhcii_1.get_data('qy_tdm')
# dict2_tdm = lhcii_2.get_data('qy_tdm')
# dict3_tdm = lhcii_3.get_data('qy_tdm')
# dict4_tdm = lhcii_4.get_data('qy_tdm')
# dict5_tdm = lhcii_5.get_data('qy_tdm')
# dict6_tdm = lhcii_6.get_data('qy_tdm')
# dict7_tdm = lhcii_7.get_data('qy_tdm')
# dict8_tdm = lhcii_8.get_data('qy_tdm')

# dict_tdm_list = [dict0_tdm, dict1_tdm, dict2_tdm, dict3_tdm, dict4_tdm, dict5_tdm, dict6_tdm, dict7_tdm, dict8_tdm]
dict_tdm_list = [dict0_tdm, dict1_tdm]
total_dict_tdm = update_tdm_dict(dict_tdm_list) # generate the full dict_tdm

U = run_calc(total_dict_tdm)
chl_states = gen_chl_states(total_dict_tdm)
dict_matrix = gen_chl_indexing(total_dict_tdm, U, chl_states)
print("Dict_matrix = {}".format(dict_matrix))

# CLA611_612_coupling = dict_matrix[(('LHCII', 0, 'D', 'CLA612'), ('CGP1', 'LHCII',0, 'D', 'CLA611'))]
# CLA611_612_coupling = dict_matrix[(('CGP1', 'LHCII',0, 'D', 'CLA612'), ('CGP1', 'LHCII',0, 'D', 'CLA611'))]
# print(CLA611_612_coupling)

# exporting the output to excel format:
# -------------------------------------
# results = export_data2excel('/Users/48107674/Documents/research/LHCII/results/DipoleInteraction-1rwt.xlsx',U)

# now the following two function doesnot have any usage now after we change the chl_list to be dict:
# --------------------------------------------------------------------------------------------------
# def gen_chl_states_doran(chl_list):
#     chl_index = []
#     for i in enumerate(chl_list):
#         chl_index.append(i)
#     dict_chl_index = dict(chl_index)
#     # print(dict_chl_index)
#     chl_states = []
#     for i in (dict_chl_index.items()):
#         for j in (dict_chl_index.items()):
#             chl_states.append((i[1], j[1]))
#     return chl_states
#
# def gen_chl_indexing_doran(chl_list, chl_matrix, chl_states):
#     coupling = []
#     for i in range(len(chl_list)):
#         for j in range(len(chl_list)):
#             coupling.append(chl_matrix[i, j])
#     dict_chl_indexing = dict(zip(chl_states, coupling))
#     return dict_chl_indexing


# dict_chl_doran = {('main PDB', 'LHCII', 0, 'D', 'CHL601'): 0, ('main PDB', 'LHCII', 0, 'D', 'CHL602'): 0,
#                   ('main PDB', 'LHCII', 0, 'D', 'CHL603'): 0, ('main PDB', 'LHCII', 0, 'D', 'CHL604'): 0,
#                   ('main PDB', 'LHCII', 0, 'D', 'CHL605'): 0, ('main PDB', 'LHCII', 0, 'D', 'CHL606'): 0,
#                   ('main PDB', 'LHCII', 0, 'D', 'CHL607'): 0, ('main PDB', 'LHCII', 0, 'D', 'CHL608'): 0,
#                   ('main PDB', 'LHCII', 0, 'D', 'CHL609'): 0, ('main PDB', 'LHCII', 0, 'D', 'CHL610'): 0,
#                   ('main PDB', 'LHCII', 0, 'D', 'CHL611'): 0, ('main PDB', 'LHCII', 0, 'D', 'CHL612'): 0,
#                   ('main PDB', 'LHCII', 0, 'D', 'CHL613'): 0, ('main PDB', 'LHCII', 0, 'D', 'CHL614'): 0}
#
# H = np.array(pd.read_csv('/Users/48107674/Documents/research/LHCII/Doran/Hamiltonian.csv'))
# chl_Hamiltonian = H[ 128:170, 128:170 ]  # Hamiltonian values
#
# chl_states_doran = gen_chl_states(dict_chl_doran)
# dict_matrix_doran = gen_chl_indexing(dict_chl_doran, chl_Hamiltonian, chl_states_doran)
# print("Dict_matrix Doran = {}".format(dict_matrix_doran))
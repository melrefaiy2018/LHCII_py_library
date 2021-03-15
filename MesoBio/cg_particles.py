import numpy as np
import copy

class CGParticle:

    def __init__(self, atm_wrp, location=(0,0,0), R2_orient=None, name=None):
        self.atm = atm_wrp
        self.location = np.array(location, dtype=np.float64)
        if R2_orient is None:
            self.R2_orient = np.eye(3)
        else:
            self.R2_orient = R2_orient

        if name is None:
            self._name = ''
        else:
            self._name = name

    def get_data(self, data_name):
        data_dict = copy.deepcopy(self.atm.get_data(data_name, log_path=tuple([self._name] + [self.atm._name])))
        for data_obj in data_dict.values():
            data_obj.transform(self.R2_orient, self.location)
        return data_dict

    def translate(self, vector):
        self.location += np.array(vector, dtype=np.float64)

    def rotate(self, angle, axis):
        center_of_mass = self.atm.center_of_mass
        try:
            axis = [1 if i == "xyz".index(axis) else 0 for i in range(3)]
        except ValueError:
            raise ValueError("'{}' is not a valid axis".format(axis))
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(angle / 2)
        b, c, d = -axis * np.sin(angle / 2)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        self.translate(-center_of_mass)
        self.transform(np.transpose(np.array([
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
        ])), center_of_mass)

    def transform(self, R2_rotate, V1_trans):
        self.R2_orient = np.dot(self.R2_orient, R2_rotate)
        self.translate(V1_trans)

    def visualize_data(self, ax, data_name):
        dict_data = self.get_data(data_name)
        for data_obj in dict_data.values():
            data_obj.visualize(ax)

    @property
    def name(self):
        if self._name is not None:
            return self._name
        else:
            return self.__repr__()

    def __repr__(self):
        return self.atm.get_id() + ' loc: {}'.format(self.location)


# class PigmentProteinComplex(CGParticle):
#     def __init__(self, meso_pdb_obj, dict_pig, enumerate_pig,
#                  location =(0,0,0), R2_orient=None, name=None):
#         super().__init__(meso_pdb_obj, location, R2_orient, name)
#         self.dict_pig = dict_pig
#         self.list_pig = [dict_pig[name] for name in enumerate_pig]
#         self.list_names = enumerate_pig
#
#
# class CGSet:
#
#     def __init__(self, iterable_cgparticles):

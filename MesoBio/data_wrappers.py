import numpy as np


class DataWrapper:
    def __init__(self):
        pass


class LocationWrapper(DataWrapper):
    def __init__(self, location, color='k'):
        self.location = np.array(location, dtype=np.float64)
        self.color = color

    def __eq__(self, other):
        if isinstance(other, LocationWrapper):
            if np.all(self.location == other.location):
                return True
        return False

    def __repr__(self):
        return 'location: {}'.format(self.location)

    def translate(self, vector):
        self.location += np.array(vector, dtype=np.float64)

    def rotate(self, angle, axis):
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
        self.transform(np.transpose(np.array([
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
        ])), [0,0,0])

    def transform(self, R2_rot, V1_trans):
        """
        To maintain consistency with BioPython this needs to use a right-handed
        definition
        """
        self.location = np.dot(self.location, R2_rot) + V1_trans


    def visualize(self, ax):
        xpos = [self.location[0]]
        ypos = [self.location[1]]
        zpos = [self.location[2]]
        ax.scatter(xpos, ypos, zpos, marker='d', s=30, color=self.color)


class VectorWrapper(LocationWrapper):

    def __init__(self, vector, location = (0,0,0), color='k'):
        super().__init__(location, color)
        self.vector = np.array(vector, dtype=np.float64)

    def __eq__(self, other):
        if isinstance(other, VectorWrapper):
            if (np.all(self.location == other.location)
                    and (np.all(self.vector == other.vector))):
                return True
        return False

    def __repr__(self):
        return 'vector: {}, ' \
               'location: {}, '.format(self.vector,
                                  self.location)

    def translate(self, vector):
        self.location += np.array(vector, dtype=np.float64)

    def transform(self, R2_rot, V1_trans):
        super().transform(R2_rot, V1_trans)
        self.vector = np.dot(self.vector, R2_rot)

    def visualize(self, ax):
        x0 = [-0.5*self.vector[0] + self.location[0]]
        y0 = [-0.5*self.vector[1] + self.location[1]]
        z0 = [-0.5*self.vector[2] + self.location[2]]
        dx = [self.vector[0]]
        dy = [self.vector[1]]
        dz = [self.vector[2]]

        ax.quiver(x0, y0, z0, dx, dy, dz, color=self.color)
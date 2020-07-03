import numpy as np
from sklearn import linear_model


class Plane3D:
    def __init__(self, data):
        XY = data[:,:2]
        Z  = data[:,2]
        self.ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),residual_threshold=0.1)
        self.ransac.fit(XY, Z)

    def get_z(self, x, y):
        return self.ransac.predict(np.array([[x, y]]))


if __name__ == "__main__":
    data = np.array([[15.401, -0.82035, -1.5316], 
                     [15.226, 1.8142, -1.5227],
                     [5.2479, 1.674, -1.6736],
                     [15.294, 1.5374, -1.5264],
                     [10.793, -1.553, -1.5937],
                     [5.5749, -3.3064, -1.7005],
                     [8.8118, -0.44693, -1.6224],
                     [6.1645, -3.977, -1.698]])

    plane = Plane3D(data)
    print('<< RANSAC Result >>')
    print('x: %s, y: %s --> z: %s'%(4.5, -4.0, plane.get_z(4.5, -4.0)))
    print('x: %s, y: %s --> z: %s'%(4.5, 4.0, plane.get_z(4.5, 4.0)))
    print('x: %s, y: %s --> z: %s'%(15.0, -4.0, plane.get_z(15.0, -4.0)))
    print('x: %s, y: %s --> z: %s'%(15.0, 4.0, plane.get_z(15.0, 4.0)))
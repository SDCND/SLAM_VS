import numpy as np


measurements = [1., 2., 3.]

x = np.matrix([[0.], [0.]])                 # initial state (location and velocity)
P = np.matrix([[1000., 0.], [0., 1000.]])   # initial uncertainty
u = np.matrix([[0.], [0.]])                 # external motion
F = np.matrix([[1., 1.], [0., 1.]])         # next state function
H = np.matrix([[1., 0.]])                   # measurement function
R = np.matrix([[1.]])                       # Measurement uncertainty
I = np.matrix([[1., 0.], [0., 1.]])         # identity matrix


def filter(x, P):

    for n in range(len(measurements)):

        # messurement update
        Z = np.matrix(measurements[n])
        y = Z - (H * x)
        S = H * P * H.transpose() + R
        #K = P * H.transpose() * S.inverse()
        K = P * H.transpose() * np.linalg.inv(S)
        x = x + (K * y)

        P = (I - (K * H)) * P

        # prediction
        x = (F * x) + u
        P = F * P * F.transpose()

        print("x = ", x)
        print("P = ", P)


filter(x, P)

# OUTPUT:

# x =  [[ 0.999001]
#  [ 0.      ]]
# P =  [[ 1000.999001  1000.      ]
#  [ 1000.        1000.      ]]
# x =  [[ 2.99800299]
#  [ 0.999002  ]]
# P =  [[ 4.99002494  2.99301795]
#  [ 2.99301795  1.99501297]]
# x =  [[ 3.99966644]
#  [ 0.99999983]]
# P =  [[ 2.33189042  0.99916761]
#  [ 0.99916761  0.49950058]]
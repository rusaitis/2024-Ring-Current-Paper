import numpy as np

# Linear Algebra
def r_unit_vector(r):
    """Return a unit vector."""
    return r / np.linalg.norm(r)

def phi_unit_vector(r):
    """Return a unit vector in the phi direction."""
    # phi = r[0] * np.array([0, 1]) - r[1] * np.array([1, 0])
    phi = r[0] * np.array([0, 1]) - r[1] * np.array([1, 0])
    return phi / np.linalg.norm(phi)

def phi_unit_vector_3d(r):
    """Return a unit vector in the phi direction."""
    phi = r[0] * np.array([0, 1, 0]) - r[1] * np.array([1, 0, 0])
    return phi / np.linalg.norm(phi)

def perpendicular_vector(v):
    # Create a random vector to serve as the initial perpendicular vector
    random_vector = np.random.rand(3)
    # Compute the cross product between the radial vector and the random vector
    perpendicular_vector = np.cross(v, random_vector)
    return perpendicular_vector

# Particles
def particle_par_perp_velocities(U, V):
    print('particle_par_perp_velocities')
        # for i in range(x_bins.size-1):
        #     for j in range(y_bins.size-1):
        #         if U[i, j] is None or V[i,j] is None:
        #             continue
        #         v = np.array([U[i, j], V[i, j]])
        #         phi_comp = np.dot(v, phi_unit_vector([X[i, j], Y[i, j]]))
        #         r_comp = np.dot(v, r_unit_vector([X[i, j], Y[i, j]]))
        #         VR[i, j] = r_comp

        #         VPHI[i, j] = phi_comp
        #         VPHI_ratio[i, j] = phi_comp / r_comp

        #         radius = np.sqrt(X[i, j]**2 + Y[i, j]**2)
        #         # make L be normally distributed about radius
        #         if radius > 5 and radius < 8:
        #             L = 4
        #         else:
        #             L = 4./(radius-6.5)

                # U[i, j] = phi_unit_vector([X[i, j], Y[i, j]])[0] * L
                # V[i, j] = phi_unit_vector([X[i, j], Y[i, j]])[1] * L


def Rx(phi):
    return np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rz(psi):
    return np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])

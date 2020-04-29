import math

# from numba import njit
import numpy as np
import scipy as sp

def two_norm(a):
    return math.sqrt(a[0]*a[0] + a[1]*a[1])

def find_normals(P):
    # find pointing unit vectors assuming clock-wise points
    r10 = (P[1] - P[0])/two_norm(P[1] - P[0])
    r21 = (P[2] - P[1])/two_norm(P[2] - P[1])
    r32 = (P[3] - P[2])/two_norm(P[3] - P[2])
    r03 = (P[0] - P[3])/two_norm(P[0] - P[3])
    # rotate by 90deg to get normal vectors inward
    N0 = [-r03[1],r03[0]]
    N1 = [-r10[1],r10[0]]
    N2 = [-r21[1],r21[0]]
    N3 = [-r32[1],r32[0]]
    return (N0, N1, N2, N3)


# @njit
def map_uv_to_xy(u, v, P, N):
    '''
    A[0, :] = u*N[2]-(1-u)*N[0]
    A[1, :] = v*N[3]-(1-v)*N[1]
    b[0] = u*P[2]@N[2]-(1-u)*P[0]@N[0]
    b[1] = v*P[3]@N[3]-(1-v)*P[0]@N[1]
    return np.linalg.solve(A, b)
    '''
    nu = 1 - u
    nv = 1 - v
    '''
    A =
    u*N[2][0]-nu*N[0][0]  u*N[2][1]-nu*N[0][1]
    v*N[3][0]-nv*N[1][0]  v*N[3][1]-nv*N[1][1]

    A_inv =
     v*N[3][1]-nv*N[1][1]  -u*N[2][1]+nu*N[0][1]
    -v*N[3][0]+nv*N[1][0]   u*N[2][0]-nu*N[0][0]
    
    '''
    b_0 = u*(P[2][0]*N[2][0] + P[2][1]*N[2][1])-nu*(P[0][0]*N[0][0] + P[0][1]*N[0][1])
    b_1 = v*(P[3][0]*N[3][0] + P[3][1]*N[3][1])-nv*(P[0][0]*N[1][0] + P[0][1]*N[1][1])
    x = b_0 * ( v*N[3][1]-nv*N[1][1]) + b_1*(-u*N[2][1]+nu*N[0][1])
    y = b_0 * (-v*N[3][0]+nv*N[1][0]) + b_1*( u*N[2][0]-nu*N[0][0])
    # print(u,v,'->',int(x),int(y))
    return x, y

# @njit
def get_sample_xy_points(points, normals, npoints, height_pixels, width_pixels):
    out_x = np.zeros((npoints, npoints))
    out_y = np.zeros((npoints, npoints))
    u = np.linspace(0, 1, npoints)
    v = np.linspace(0, 1, npoints)
    for i in range(npoints):
        for j in range(npoints):
            xy = map_uv_to_xy(u[j], v[i], points, normals)
            r = int(xy[1])
            c = int(xy[0])
            if r < height_pixels:
                out_y[i,j] = r  
            else:
                out_y[i,j] = height_pixels - 1

            if c < width_pixels:
                out_x[i,j] = c
            else:
                out_x[i,j] = width_pixels - 1

    return u, v, out_x, out_y

# @njit
def get_square_impl(xx, yy, gray):
    out = np.empty(gray.shape, gray.dtype)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            out[i, j] = gray[yy[i, j], xx[i, j]]
    return out

def get_square_image(gray, width_pixels, height_pixels, points):
    normals = find_normals(points)
    out = np.zeros((height_pixels, width_pixels))
    u, v, x_map, y_map = get_sample_xy_points(points, normals, 30, height_pixels, width_pixels)

    x_func = sp.interpolate.interp2d(u, v, x_map)
    y_func = sp.interpolate.interp2d(u, v, y_map)
    u = np.linspace(0,1,width_pixels)
    v = np.linspace(0,1,height_pixels)

    # uu, vv = np.meshgrid(u, v)

    xx = x_func(u, v)
    np.rint(xx, out=xx)
    np.clip(xx, a_min=0, a_max=None, out=xx)
    xx = xx.astype(np.int)

    yy = y_func(u, v)
    np.rint(yy, out=yy)
    np.clip(yy, a_min=0, a_max=None, out=yy)
    yy = yy.astype(np.int)
    return gray[yy, xx]

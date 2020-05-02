import math

# import matplotlib.pyplot as plt
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
    N0 = np.array([r03[1],-r03[0]])
    N1 = np.array([r10[1],-r10[0]])
    N2 = np.array([r21[1],-r21[0]])
    N3 = np.array([r32[1],-r32[0]])
    # N0 = [-r03[1],r03[0]]
    # N1 = [-r10[1],r10[0]]
    # N2 = [-r21[1],r21[0]]
    # N3 = [-r32[1],r32[0]]
    # N0 = [-r03[0], -r03[1]]
    # N1 = [-r10[0], -r10[1]]
    # N2 = [-r21[0], -r21[1]]
    # N3 = [-r32[0], -r32[1]]
    return (N0, N1, N2, N3)


# @njit
def map_uv_to_xy(u, v, P, N):
    A = np.zeros((2,2))
    b = np.zeros(2)
    # print(N[2])
    # print(u)
    # print(u*N[2])
    # print((1-u))
    # print(N[0])
    A[0, :] = u*N[2]-(1-u)*N[0]
    A[1, :] = v*N[3]-(1-v)*N[1]
    b[0] = u*P[2]@N[2]-(1-u)*P[0]@N[0]
    b[1] = v*P[3]@N[3]-(1-v)*P[0]@N[1]
    return np.linalg.solve(A, b)

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
    # print(points)
    # print(u)
    # print(v)
    for i in range(npoints):
        for j in range(npoints):
            xy = map_uv_to_xy(u[j], v[i], points, normals)
            r = int(xy[1])
            c = int(xy[0])
            if r < height_pixels:
                out_y[i,j] = r  
            else:
                # assert False
                out_y[i,j] = height_pixels - 1

            if c < width_pixels:
                out_x[i,j] = c
            else:
                # assert False
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
    u, v, x_map, y_map = get_sample_xy_points(points, normals, 30, gray.shape[0], gray.shape[1])
    # u, v, x_map, y_map = get_sample_xy_points(points, normals, 30, height_pixels, width_pixels)
    # print(u)
    # print(v)
    # print(x_map)
    # print(y_map)
    # plt.imshow(out, )
    # plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    # plt.plot(x_map.flatten(), y_map.flatten(), marker='x', color='k')
    # plt.show()
    x_func = sp.interpolate.interp2d(u, v, x_map)
    y_func = sp.interpolate.interp2d(u, v, y_map)
    u = np.linspace(0,1,width_pixels)
    v = np.linspace(0,1,height_pixels)
    # plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    # plt.plot(x_map.flatten(), y_map.flatten(), marker='x', color='k')
    # plt.show()
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


def test_map_uv_to_xy():
    import matplotlib.pyplot as plt
    points = [
        np.array([1,-2]),
        np.array([10,-3]),
        np.array([12,-9]),
        np.array([2, -7]),
        np.array([1,-2]),
    ]
    normals = find_normals(points)
    # print(points)
    print(normals)
    # print(np.linalg.norm(normals, axis=1))
    np.testing.assert_array_almost_equal(np.linalg.norm(normals, axis=1), np.ones(4))
    # slope = -1/9
    print(np.array([9, 1])/math.sqrt(82))
    # assert normals[0] == np.array([9, 1])/math.sqrt(82)
    plt.plot(*zip(*points), color='k')
    for i in range(4):
        p = (points[i] + points[(i+1)%4])/2
        normal = normals[(i+1)%4]
        # normal = normals[i]
        plt.plot(*zip(*[p, p+normal]),label=f'n{(i+1)%4}')
        # plt.plot(*zip(*[p, p+normal]),label=f'n{i}')

    for u in np.linspace(0,1,10):
        for v in np.linspace(0,1,10):
            (x, y) = p = map_uv_to_xy(u, v, points, normals)
            a = (p - points[0])@normals[0]
            b = (p - points[2])@normals[2]
            _u = a / (a+b)
            c = (p - points[0])@normals[1]
            d = (p - points[3])@normals[3]
            _v = c / (c+d)
            # print(u, _u)
            # print(v, _v)
            # print('#######')

            plt.plot([x], [y], marker='x', color='r')
    plt.xlim([0,15])
    plt.ylim([-15, 0])
    plt.legend()
    plt.show()
    # print(map)
    print(map_uv_to_xy(0, 0, points, normals))
    print(map_uv_to_xy(0, 1, points, normals))
    print(map_uv_to_xy(1, 0, points, normals))
    print(map_uv_to_xy(1, 1, points, normals))

if __name__ == '__main__':
    test_map_uv_to_xy()

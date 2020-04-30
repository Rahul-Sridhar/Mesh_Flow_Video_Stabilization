import cv2
import numpy as np

def mesh_warp_frame(frame, x_motion_mesh, y_motion_mesh, PIXELS):
    """
    :param frame: current frame
    :param x_motion_mesh: motion mesh to be warped on frame along x-direction
    :param y_motion_mesh: motion mesh to be warped on frame along y-direction
    :param PIXELS: number of pixels
    :return:
            returns a mesh warped frame according to given motion meshes according to given motion meshes x_motion_mesh, y_motion_mesh
    """
    map_x = np.zeros((frame.shape[0], frame.shape[1]), np.float32)

    # define handles on mesh in y-direction
    map_y = np.zeros((frame.shape[0], frame.shape[1]), np.float32)

    for i in range(x_motion_mesh.shape[0] - 1):
        for j in range(x_motion_mesh.shape[1] - 1):

            src = [[j * PIXELS, i * PIXELS],
                   [j * PIXELS, (i + 1) * PIXELS],
                   [(j + 1) * PIXELS, i * PIXELS],
                   [(j + 1) * PIXELS, (i + 1) * PIXELS]]
            src = np.asarray(src)

            dst = [[j * PIXELS + x_motion_mesh[i, j], i * PIXELS + y_motion_mesh[i, j]],
                   [j * PIXELS + x_motion_mesh[i + 1, j], (i + 1) * PIXELS + y_motion_mesh[i + 1, j]],
                   [(j + 1) * PIXELS + x_motion_mesh[i, j + 1], i * PIXELS + y_motion_mesh[i, j + 1]],
                   [(j + 1) * PIXELS + x_motion_mesh[i + 1, j + 1], (i + 1) * PIXELS + y_motion_mesh[i + 1, j + 1]]]
            dst = np.asarray(dst)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC)

            for k in range(PIXELS * i, PIXELS * (i + 1)):
                for l in range(PIXELS * j, PIXELS * (j + 1)):
                    x = H[0, 0] * l + H[0, 1] * k + H[0, 2]
                    y = H[1, 0] * l + H[1, 1] * k + H[1, 2]
                    w = H[2, 0] * l + H[2, 1] * k + H[2, 2]
                    if not w == 0:
                        x = x / (w * 1.0);
                        y = y / (w * 1.0)
                    else:
                        x = l;
                        y = k
                    map_x[k, l] = x
                    map_y[k, l] = y

    for i in range(PIXELS * x_motion_mesh.shape[0], map_x.shape[0]):
        map_x[i, :] = map_x[PIXELS * x_motion_mesh.shape[0] - 1, :]
        map_y[i, :] = map_y[PIXELS * x_motion_mesh.shape[0] - 1, :]

        # repeat motion vectors for remaining frame in x-direction
    for j in range(PIXELS * x_motion_mesh.shape[1], map_x.shape[1]):
        map_x[:, j] = map_x[:, PIXELS * x_motion_mesh.shape[0] - 1]
        map_y[:, j] = map_y[:, PIXELS * x_motion_mesh.shape[0] - 1]

        # deforms mesh
    new_frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return new_frame
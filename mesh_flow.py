import cv2
import numpy as np
from scipy.signal import medfilt
from homography import point_transform

def motion_propagate(old_points, new_points, old_frame, PIXELS, RADIUS):
    """
    :param old_points: previous frame matched features
    :param new_points: current frame matched features
    :param old_frame: frames for which motion mesh needs to be obtained
    :param PIXELS: pixel dimension for a mesh grid
    :param RADIUS: radius for allocating a point to the mesh vertices
    :return:
            returns motion mesh in x-direction and y-direction for the frame
    """

    x_motion = {}
    y_motion = {}
    cols, rows = int(old_frame.shape[1]/PIXELS), int(old_frame.shape[0]/PIXELS)

    H, _ = cv2.findHomography(old_points, new_points, cv2.RANSAC)
    for i in range(rows):
        for j in range(cols):
            pt = [PIXELS*j, PIXELS*i]
            ptrans = point_transform(H, pt)
            x_motion[i, j] = pt[0] - ptrans[0]
            y_motion[i, j] = pt[1] - ptrans[1]

    temp_x_motion = {}
    temp_y_motion = {}
    for i in range(rows):
        for j in range(cols):
            vertex = [PIXELS*j, PIXELS*i]
            for pt, st in zip(old_points, new_points):
                dst = np.sqrt((vertex[0] - pt[0])**2 + (vertex[1] - pt[1])**2)
                if dst < RADIUS:
                    ptrans = point_transform(H, pt)
                    try:
                        temp_x_motion[i, j].append(st[0] - ptrans[0])
                    except:
                        temp_x_motion[i, j] = [st[0] - ptrans[0]]
                    try:
                        temp_y_motion[i, j].append(st[1] - ptrans[1])
                    except:
                        temp_y_motion[i, j] = [st[1] - ptrans[1]]


    x_motion_mesh = np.zeros((rows, cols), dtype = float)
    y_motion_mesh = np.zeros((rows, cols), dtype = float)
    for key in x_motion.keys():
        try:
            temp_x_motion[key].sort()
            x_motion_mesh[key] = x_motion[key] + temp_x_motion[key][int(len(temp_x_motion[key])/2)]
        except KeyError:
            x_motion_mesh[key] = x_motion[key]
        try:
            temp_y_motion[key].sort()
            y_motion_mesh[key] = y_motion[key] + temp_y_motion[key][int(len(temp_y_motion[key])/2)]
        except KeyError:
            y_motion_mesh[key] = y_motion[key]

    x_motion_mesh = medfilt(x_motion_mesh, kernel_size = [3, 3])
    y_motion_mesh = medfilt(y_motion_mesh, kernel_size = [3, 3])

    return x_motion_mesh, y_motion_mesh

def generate_vertex_profiles(x_paths, y_paths, x_motion_mesh, y_motion_mesh):
    """
    :param x_paths: vertex profiles along x_direction
    :param y_paths: vertex profiles along y_direction
    :param x_motion_mesh: motion along x-direction from motion_propagate()
    :param y_motion_mesh: motion along y-direction from motion_propagate()
    :return:
    """
    new_x_path = x_paths[:, :, -1] + x_motion_mesh
    new_y_path = y_paths[:, :, -1] + y_motion_mesh
    x_paths = np.concatenate((x_paths, np.expand_dims(new_x_path, axis = 2)), axis = 2)
    y_paths = np.concatenate((y_paths, np.expand_dims(new_y_path, axis=2)), axis=2)
    return x_paths, y_paths
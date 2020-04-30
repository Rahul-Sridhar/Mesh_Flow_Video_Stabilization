import cv2
import sys
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
from mesh_flow import motion_propagate,generate_vertex_profiles
from optimization import optimize_path
from smooth_recontructed_frames import mesh_warp_frame

PIXELS = 16
RADIUS = 300
HORIZONTAL_BORDER = 30
frame_rate = 0
frame_width = 0
frame_height = 0
frame_count = 0

def main():
    filename = sys.argv[1]
    cap = cv2.VideoCapture(filename)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('results/stabilized_video/stable.avi', fourcc, frame_rate, (2 * frame_width, frame_height))

    # Parameters for ShiTomasi corner detector
    feature_params = dict(maxCorners = 1000, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    VERTICAL_BORDER = int((HORIZONTAL_BORDER * old_gray.shape[1]) / old_gray.shape[0])

    x_motion_meshes = []
    y_motion_meshes = []

    x_paths = np.zeros((int(old_frame.shape[0]/PIXELS), int(old_frame.shape[1]/PIXELS), 1))
    y_paths = np.zeros((int(old_frame.shape[0]/PIXELS), int(old_frame.shape[1]/PIXELS), 1))

    frame_num = 1
    bar = tqdm(total=frame_count)
    while(frame_num < frame_count):
        # processing frames
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find corners
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # estimate motion mesh for old_frame
        x_motion_mesh, y_motion_mesh = motion_propagate(good_old, good_new, frame, PIXELS, RADIUS)
        try:
            x_motion_meshes = np.concatenate((x_motion_meshes, np.expand_dims(x_motion_mesh, axis=2)), axis=2)
            y_motion_meshes = np.concatenate((y_motion_meshes, np.expand_dims(y_motion_mesh, axis=2)), axis=2)
        except:
            x_motion_meshes = np.expand_dims(x_motion_mesh, axis=2)
            y_motion_meshes = np.expand_dims(y_motion_mesh, axis=2)

        # generate vertex profiles
        x_paths, y_paths = generate_vertex_profiles(x_paths, y_paths, x_motion_mesh, y_motion_mesh)

        # update frames
        bar.update(1)
        frame_num += 1
        old_frame = frame.copy()
        old_gray = frame_gray.copy()

    bar.close()

    sx_paths = optimize_path(x_paths)
    sy_paths = optimize_path(y_paths)

    for i in range(0, x_paths.shape[0]):
        for j in range(0, x_paths.shape[1], 10):
            plt.plot(x_paths[i, j, :])
            plt.plot(sx_paths[i, j, :])
            plt.savefig("results/paths/"+str(i)+"_"+str(j)+".png")
            plt.clf()

    #x_motion_meshes = np.concatenate((x_motion_meshes, np.expand_dims(x_motion_meshes[:, :, -1], axis=2)), axis=2)
    #y_motion_meshes = np.concatenate((y_motion_meshes, np.expand_dims(y_motion_meshes[:, :, -1], axis=2)), axis=2)
    new_x_motion_meshes = sx_paths - x_paths
    new_y_motion_meshes = sy_paths - y_paths

    r = 3
    frame_num = 0
    bar = tqdm(total=frame_count)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while(frame_num < frame_count):
        try:
            ret, frame = cap.read()
            x_motion_mesh = x_motion_meshes[:, :, frame_num]
            y_motion_mesh = y_motion_meshes[:, :, frame_num]
            new_x_motion_mesh = new_x_motion_meshes[:, :, frame_num]
            new_y_motion_mesh = new_y_motion_meshes[:, :, frame_num]
            new_frame = mesh_warp_frame(frame, new_x_motion_mesh, new_y_motion_mesh, PIXELS)
            new_frame = new_frame[HORIZONTAL_BORDER:-HORIZONTAL_BORDER, VERTICAL_BORDER:-VERTICAL_BORDER, :]
            new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_CUBIC)
            output = np.concatenate((frame, new_frame), axis=1)
            out.write(output)

            for i in range(x_motion_mesh.shape[0]):
                for j in range(x_motion_mesh.shape[1]):
                    thetha = np.arctan2(y_motion_mesh[i, j], x_motion_mesh[i, j])
                    cv2.line(frame, (j*PIXELS, i*PIXELS), (int(j*PIXELS + r*np.cos(thetha)), int(i*PIXELS + r*np.sin(thetha))), 1)
            cv2.imwrite("results/old_motion_vectors/" + str(frame_num) + ".jpg", frame)

            for i in range(new_x_motion_mesh.shape[0]):
                for j in range(new_x_motion_mesh.shape[1]):
                    thetha = np.arctan2(new_y_motion_mesh[i, j], new_x_motion_mesh[i, j])
                    cv2.line(new_frame, (j*PIXELS, i*PIXELS), (int(j*PIXELS + r*np.cos(thetha)), int(i*PIXELS + r*np.sin(thetha))), 1)
            cv2.imwrite("results/new_motion_vectors/" + str(frame_num) + ".jpg", new_frame)

            frame_num += 1
            bar.update(1)

        except:
            break
    bar.close()

    cap.release()
    out.release()




if __name__=="__main__":
    main()
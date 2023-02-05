import matplotlib.pyplot as plt
import imageio
import numpy as np
import os

def visualize(vessels, gif_path='./dynamicSceneExample.gif', figsize=(6, 6), y_x_lim=400, fps=3):
    '''
    Saves a gif of the track of a vessel
    Input:
    - gif_path (string): Path to save visualization
    - figsize (tuple, int): Size of figure
    - y_x_lim (int): limitation of x and y axis
    - fps (int): frames per second in the gif
    '''
    for i in range(len(vessels)-1):
        if vessels[i].get_track().get_time_stamps() != vessels[i+1].get_track().get_time_stamps():
            raise Exception("Points are not collected at the same time stamps")

    for t in vessels[0].get_track().get_time_stamps():
        create_frame(t, vessels, figsize, y_x_lim)
    
    frames = []
    for t in vessels[0].get_track().get_time_stamps():
        image = imageio.v2.imread(f'./img/img_{t}.png')
        frames.append(image)
    
    imageio.mimsave(gif_path, frames, fps = fps, loop = 1)


def create_frame(t, vessels, figsize, y_x_lim):
    '''
    Creates the plot image for the given time step
    Input:
    - t (int): current time step
    - vessels (array): List of vessels in the scene
    - figsize (int): Size of figure
    - y_x_lim (int): limitation of x and y axis
    '''
    _ = plt.figure(figsize=figsize)
    for vessel in vessels:
        track = vessel.get_track()
        x = track.get_x_values()
        y = track.get_y_values()
        plt.plot(x[:(t+1)], y[:(t+1)])
        #plt.plot(x[t], y[t], marker = 'o' )
        # Create square plot for shape of vessel
        #position = np.array([x[t], y[t]])
        #direction_vector = track.get_direction_vector(t)
        cornerpoints = vessel.calculate_2D_cornerpoints(t)
        xs = list(cornerpoints[:,0])+[cornerpoints[:,0][0]]
        ys = list(cornerpoints[:,1])+[cornerpoints[:,1][0]]
        plt.plot(xs, ys, 'b-')
    plt.xlim([0,y_x_lim])
    plt.xlabel('x', fontsize = 14)
    plt.ylim([0,y_x_lim])
    plt.ylabel('y', fontsize = 14)
    plt.title(f'Relationship between x and y at step {t}', fontsize=14)
    plt.savefig(f'img/img_{t}.png', transparent = False,  facecolor = 'white')
    plt.close()

def plot_image(projected_points, image_bounds, t, figsize=(6,6)):
    '''
    Input:
    projected_points (List): List of lists of points for each vessel
    figsize (int): Size of figure
    image_bounds (Tuple): x and y pixel boundaries

    '''
    _, ax = plt.subplots(figsize=figsize)
    for vessel in projected_points:
        vessel_x = vessel[:,0]
        vessel_y = vessel[:,1]
        ax.plot(vessel_x, vessel_y, 'o')
    plt.xlim([0,image_bounds[0]])
    plt.ylim([image_bounds[1],0])
    plt.ylabel('y', fontsize = 14)
    ax.xaxis.tick_top()
    ax.set_xlabel('x', fontsize = 14)    
    ax.xaxis.set_label_position('top') 
    plt.title(f'Projected points', fontsize=14)
    plt.savefig(f'./projectedPoints/projectedPoints_{t}.png', transparent = False,  facecolor = 'white')
    plt.close()

def visualize_camera(t, camera, vessels, y_x_lim=400, figsize=(6,6)):

    _ = plt.figure(figsize=figsize)
    for vessel in vessels:
        track = vessel.get_track()
        x = track.get_x_values()
        y = track.get_y_values()
        plt.plot(x[:(t+1)], y[:(t+1)])
        #plt.plot(x[t], y[t], marker = 'o' )
        # Create square plot for shape of vessel
        #position = np.array([x[t], y[t]])
        #direction_vector = track.get_direction_vector(t)
        cornerpoints = vessel.calculate_2D_cornerpoints(t)
        xs = list(cornerpoints[:,0])+[cornerpoints[:,0][0]]
        ys = list(cornerpoints[:,1])+[cornerpoints[:,1][0]]
        plt.plot(xs, ys, 'b-')
    
    camera_position = camera.get_position()
    camera_orientation = camera.get_orientation_vector()
    plt.plot(camera_position[0], camera_position[1], 'ro')
    plt.plot([camera_position[0],  camera_position[0]+camera_orientation[0]*50], [camera_position[1],  camera_position[1]+camera_orientation[1]*50], 'r-')
    plt.xlabel('x', fontsize = 14)
    plt.ylabel('y', fontsize = 14)
    x_lim = max(camera_position[0]+10, y_x_lim)
    y_lim = max(camera_position[1]+10, y_x_lim)
    plt.xlim([0,x_lim])
    plt.ylim([0,y_lim])
    plt.title(f'Camera position in scene {t}', fontsize=14)
    plt.savefig(f'./cameraPosition/cameraPosition_{t}.png', transparent = False,  facecolor = 'white')
    plt.close()

import matplotlib.pyplot as plt
import imageio
import numpy as np

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
        position = np.array([x[t], y[t]])
        direction_vector = track.get_direction_vector(t)
        cornerpoints = vessel.calculate_cornerpoints(direction_vector, position)
        xs = list(cornerpoints[:,0])+[cornerpoints[:,0][0]]
        ys = list(cornerpoints[:,1])+[cornerpoints[:,1][0]]
        plt.plot(xs, ys, 'b-')
    plt.xlim([0,y_x_lim])
    plt.xlabel('x', fontsize = 14)
    plt.ylim([0,y_x_lim])
    plt.ylabel('y', fontsize = 14)
    plt.title(f'Relationship between x and y at step {t}', fontsize=14)
    plt.savefig(f'./img/img_{t}.png', transparent = False,  facecolor = 'white')
    plt.close()
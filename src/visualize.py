import matplotlib.pyplot as plt
import imageio
import numpy as np
import os

###############################################################################################
#
#               Help functions for visualization
#
###############################################################################################
plot_colors = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
vesselID2Color = {}
vessel_count = 0

def get_color(vesselID):
    global vesselID2Color
    global vessel_count
    global plot_colors
    if vesselID in vesselID2Color.keys():
        return vesselID2Color[vesselID]
    color_index = vessel_count % len(plot_colors)
    vessel_count += 1
    vesselID2Color[vesselID] = plot_colors[color_index]
    return vesselID2Color[vesselID]

###############################################################################################
#
#               Dynamic scene visualization
#
###############################################################################################

def visualize_dynamic_scene(vessels, folder_path='./gifs/', figsize=(6, 6), y_x_lim=400, fps=3):
    '''
    Saves a gif of the track of a vessel
    Input:
    - folder_path (string): Path to the folder to save visualization
    - figsize (tuple, int): Size of figure
    - y_x_lim (int): limitation of x and y axis
    - fps (int): frames per second in the gif
    '''
    for i in range(len(vessels)-1):
        if vessels[i].get_track().get_time_stamps() != vessels[i+1].get_track().get_time_stamps():
            raise Exception("Points are not collected at the same time stamps")

    for t in vessels[0].get_track().get_time_stamps():
        plot_dynamic_scene_t(t, vessels, figsize, y_x_lim)
    
    frames = []
    for t in vessels[0].get_track().get_time_stamps():
        image = imageio.v2.imread(f'./img/img_{t}.png')
        frames.append(image)
    gif_path = os.path.join(folder_path, 'dynamicScene.gif')
    imageio.mimsave(gif_path, frames, fps = fps, loop = 1)


def plot_dynamic_scene_t(t, vessels, figsize, y_x_lim):
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
        track = vessel.get_track().get_track_dict()
        points = np.array([[track[time]['center_position_m'][0], track[time]['center_position_m'][1]] for time in sorted(track.keys()) if time <= t])
        x = points[:, 0]
        y = points[:, 1]
        plt.plot(x, y)
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

###############################################################################################
#
#               Camera position visualization
#
###############################################################################################
def visualize_camera_pose_t(t, camera, vessels, y_x_lim=None, figsize=(6,6)):

    if not y_x_lim:
        y_x_lim = camera.position_WRF[0] + 50
    _ = plt.figure(figsize=figsize)
    for vessel in vessels:
        track = vessel.get_track().get_track_dict()
        points = np.array([[track[time]['center_position_m'][0], track[time]['center_position_m'][1]] for time in sorted(track.keys()) if time <= t])
        x = points[:, 0]
        y = points[:, 1]
        plt.plot(x, y)
        '''
        OLD
        track = vessel.get_track()
        x = track.get_x_values()
        y = track.get_y_values()
        plt.plot(x[:(t+1)], y[:(t+1)])'''
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

def visualize_camera_pose_in_dsg(camera, vessels, folder_path='./gifs', y_x_lim=None, figsize=(6,6)):
    time_steps = vessels[0].get_track().get_time_stamps()
    for t in time_steps:
        visualize_camera_pose_t(t, camera, vessels, y_x_lim, figsize)
    
    frames=[]
    for t in time_steps:
        image = imageio.v2.imread(f'./cameraPosition/cameraPosition_{t}.png')
        frames.append(image)
    
    gif_path = os.path.join(folder_path,'camera_position.gif')
    imageio.mimsave(gif_path, frames, fps = 3, loop = 1)


###############################################################################################
#
#               Projection visualization
#
###############################################################################################
def plot_projections(projected_points, image_bounds, t, show_box, fastplot=False):
    '''
    Input:
    projected_points (List): List of lists of points for each vessel
    figsize (int): Size of figure
    image_bounds (Tuple): x and y pixel boundaries

    '''
    if fastplot:
        _, ax = plt.subplots()
        fontsize = 10
        ticks_fontsize = 8
    else:
        figsize = (image_bounds[0]/100, image_bounds[1]/100)
        _, ax = plt.subplots(figsize=figsize)
        fontsize = 28
        ticks_fontsize = 24

    for vessel in projected_points.values():
        vessel_x = np.array([point.image_coordinate[0] for point in vessel])
        vessel_y = np.array([point.image_coordinate[1] for point in vessel])
        ax.plot(vessel_x, vessel_y, 'o')
        # Order of cornerpoints (length, beam, height): 
        # Front back lower, back back lower, 
        # back front lower, front front lower, 
        # Front back upper, back back upper, 
        # back front upper, front front upper,
        if show_box:
            xs = list(vessel_x[0:4])+[vessel_x[0]]+list(vessel_x[4:])+[vessel_x[4]]
            ys = list(vessel_y[0:4])+[vessel_y[0]]+list(vessel_y[4:])+[vessel_y[4]]
            ax.plot(xs, ys, 'b-')
            ax.plot([vessel_x[1], vessel_x[5]], [vessel_y[1], vessel_y[5]], 'b-')
            ax.plot([vessel_x[2], vessel_x[6]], [vessel_y[2], vessel_y[6]], 'b-')
            ax.plot([vessel_x[3], vessel_x[7]], [vessel_y[3], vessel_y[7]], 'b-')

    plt.xlim([0,image_bounds[0]])
    plt.ylim([image_bounds[1],0])
    plt.ylabel('y', fontsize = fontsize)
    ax.xaxis.tick_top()
    ax.set_xlabel('x', fontsize = fontsize)    
    ax.xaxis.set_label_position('top') 
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.title(f'Projected points at time {t}', fontsize=fontsize)
    plt.savefig(f'./projectedPoints/projectedPoints_{t}.png', transparent = False,  facecolor = 'white')
    plt.close()

 # OLD Now the calculations are done here. Maybe we want to just send in a list with projected points with timestamps?
def visualize_projections_calculate(vessels, camera, folder_path = './gifs', show_box=True, fastplot=False):
    for i in range(len(vessels)-1):
        if vessels[i].get_track().get_time_stamps() != vessels[i+1].get_track().get_time_stamps():
            raise Exception("Points are not collected at the same time stamps")

    for t in vessels[0].get_track().get_time_stamps():
        points = [vessel.calculate_3D_cornerpoints(t) for vessel in vessels]
        projected_points = [camera.project_points(vessel_points) for vessel_points in points]
        plot_projections(projected_points, camera.image_bounds, t, show_box, fastplot)

    frames = []
    for t in vessels[0].get_track().get_time_stamps():
        image = imageio.v2.imread(f'./projectedPoints/projectedPoints_{t}.png')
        frames.append(image)

    gif_path = os.path.join(folder_path,'projected_point.gif')
    imageio.mimsave(gif_path, frames, fps = 3, loop = 1)


def visualize_projections(projected_points, image_bounds, show_box=True, fastplot=False, folder_path='./gifs/', fps=3):
    for t in projected_points.keys():
        plot_projections(projected_points[t], image_bounds, t, show_box, fastplot)
    
    frames = []
    for t in projected_points.keys():
        image = imageio.v2.imread(f'./projectedPoints/projectedPoints_{t}.png')
        frames.append(image)

    gif_path = os.path.join(folder_path,'projected_point.gif')
    imageio.mimsave(gif_path, frames, fps=fps, loop = 1)


###############################################################################################
#
#               Bounding box visualization
#
###############################################################################################

def plot_bbs(bounding_boxes, image_bounds, t, projected_points, show_projected_points, fastplot=False):
    if fastplot:
        _, ax = plt.subplots()
        fontsize = 10
        ticks_fontsize = 8
    else:
        figsize = (image_bounds[0]/100, image_bounds[1]/100)
        _, ax = plt.subplots(figsize=figsize)
        fontsize = 28
        ticks_fontsize = 24
    for i in range(len(bounding_boxes)):
        bb = bounding_boxes[i]
        xs, ys = bb.get_points_for_visualizing()
        ax.plot(xs, ys, '-', color=get_color(bb.vesselID))
        if show_projected_points:
            if not projected_points:
                print("Provide projected points when show projected points is true")
            else:
                vessel_proj = projected_points[t][i]
                x_vals = np.array([point.image_coordinate[0] for point in vessel_proj])
                y_vals = np.array([point.image_coordinate[1] for point in vessel_proj])
                ax.plot(x_vals, y_vals, 'o', color=get_color(bb.vesselID))

    plt.xlim([0,image_bounds[0]])
    plt.ylim([image_bounds[1],0])
    plt.ylabel('y', fontsize = fontsize)
    ax.xaxis.tick_top()
    ax.set_xlabel('x', fontsize = fontsize)    
    ax.xaxis.set_label_position('top') 
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.title(f'Bounding boxes at time {t}', fontsize=fontsize)
    plt.savefig(f'./boundingBoxes/boundingBoxes_{t}.png', transparent = False,  facecolor = 'white')
    plt.close()


def visualize_bounding_boxes(bounding_boxes, image_bounds, projected_points=None, show_projected_points=False, fastplot=False, folder_path='./gifs/', fps=3):
    for t in bounding_boxes.keys():
        plot_bbs(bounding_boxes[t], image_bounds, t, projected_points, show_projected_points, fastplot)
    
    frames = []
    for t in bounding_boxes.keys():
        image = imageio.v2.imread(f'./boundingBoxes/boundingBoxes_{t}.png')
        frames.append(image)

    gif_path = os.path.join(folder_path, 'boundingBoxes.gif')
    imageio.mimsave(gif_path, frames, fps=fps, loop = 1)

###############################################################################################
#
#               Distorted Bounding box visualization
#
###############################################################################################

def plot_distorted_bbs(distorted_bbs, image_bounds, t, original_BBs, show_original_BBS, fastplot=False):
    if fastplot:
        _, ax = plt.subplots()
        fontsize = 10
        ticks_fontsize = 8
    else:
        figsize = (image_bounds[0]/100, image_bounds[1]/100)
        _, ax = plt.subplots(figsize=figsize)
        fontsize = 28
        ticks_fontsize = 24
    if show_original_BBS:
        if not original_BBs:
            print("Provide original BBs when show original BBs is true")
        else:
            for bb in original_BBs[t]:
                xs, ys = bb.get_points_for_visualizing()
                ax.plot(xs, ys, '-', color='lightgrey')
    for bb in distorted_bbs:
        xs, ys = bb.get_points_for_visualizing()
        ax.plot(xs, ys, '-', color=get_color(bb.vesselID))

    plt.xlim([0,image_bounds[0]])
    plt.ylim([image_bounds[1],0])
    plt.ylabel('y', fontsize = fontsize)
    ax.xaxis.tick_top()
    ax.set_xlabel('x', fontsize = fontsize) 
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)   
    ax.xaxis.set_label_position('top') 
    plt.title(f'Distorted bounding boxes at time {t}', fontsize=fontsize)
    plt.savefig(f'./distortedBoundingBoxes/boundingBoxes_{t}.png', transparent = False,  facecolor = 'white')
    plt.close()


def visualize_distorted_bounding_boxes(distorted_bbs, image_bounds, original_BBs=None, show_original_BBS=False, folder_path='./gifs/', fps=3, fastplot=False):
    for t in distorted_bbs.keys():
        plot_distorted_bbs(distorted_bbs[t], image_bounds, t, original_BBs, show_original_BBS, fastplot=fastplot)
    
    frames = []
    for t in distorted_bbs.keys():
        image = imageio.v2.imread(f'./distortedBoundingBoxes/boundingBoxes_{t}.png')
        frames.append(image)

    gif_path = os.path.join(folder_path, 'distortedBoundingBoxes.gif')
    imageio.mimsave(gif_path, frames, fps=fps, loop = 1)
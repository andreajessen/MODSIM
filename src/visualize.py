import matplotlib.pyplot as plt
import numpy as np
import os
# importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from utils import json_to_projectedPoints, json_to_annot, json_to_detection

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

def get_dict_item(dict, key):
    try:
        return dict[key]
    except KeyError:
        try:
            return dict[str(key)]
        except KeyError:
            try:
                return dict[float(key)]
            except KeyError:
                return dict[int(key)]
 
###############################################################################################
#
#               Dynamic scene visualization
#
###############################################################################################
def visualize_dynamic_scene_mov(vessels, folder_path='./gifs/', figsize=(6, 6), y_x_lim=400, fps=3, skip=0, max_duration=None):
    '''
    Creates the plot image for the given time step
    Input:
    - t (int): current time step
    - vessels (array): List of vessels in the scene
    - figsize (int): Size of figure
    - y_x_lim (int): limitation of x and y axis
    '''
    fig, ax = plt.subplots(figsize=figsize)
    
    time_stamps = vessels[0].get_track().get_time_stamps()

    def make_frame(idiot_time):
        # Some hack to fix the time stamp index because of
        # the idiot FPS in videoClip that you can't change!!!!!
        temp = (int(idiot_time%1*10)-6)/3-1
        time_index = int((int(idiot_time)+1)*3+temp)
        time_index = time_index if skip == 0 else time_index*skip

        # Get time stamp
        t = time_stamps[time_index]

        # Clear
        ax.clear()

        # Plot
        ax.set_xlim([0,y_x_lim])
        ax.set_xlabel('x', fontsize = 14)
        ax.set_ylim([0,y_x_lim])
        ax.set_ylabel('y', fontsize = 14)
        ax.set_title(f'Relationship between x and y at step {t}', fontsize=14)
        for vessel in vessels:
            track = vessel.get_track().get_track_dict()
            points = np.array([[track[time]['center_position_m'][0], track[time]['center_position_m'][1]] for time in sorted(track.keys()) if time <= t])
            x = points[:, 0]
            y = points[:, 1]
            ax.plot(x, y)
            #plt.plot(x[t], y[t], marker = 'o' )
            # Create square plot for shape of vessel
            #position = np.array([x[t], y[t]])
            #direction_vector = track.get_direction_vector(t)
            cornerpoints = vessel.calculate_2D_cornerpoints(t)
            xs = list(cornerpoints[:,0])+[cornerpoints[:,0][0]]
            ys = list(cornerpoints[:,1])+[cornerpoints[:,1][0]]
            ax.plot(xs, ys, 'b-')
        # returning numpy image
        return mplfig_to_npimage(fig)
    # creating animation
    duration = int(len(time_stamps)/3) # Because of the idiot FPS that i can't change!!!!
    if max_duration and max_duration/3<duration:
        duration = max_duration/3
    if skip !=0:
        duration = duration/skip
    animation = VideoClip(make_frame, duration = duration)
    gif_path = os.path.join(folder_path, 'dynamicScene.mp4')
    animation.write_videofile(gif_path,fps=fps)

###############################################################################################
#
#               Camera position visualization
#
###############################################################################################
def visualize_camera_pose_in_dsg_mov(camera_rig, vessels, folder_path='./gifs', y_x_lim=None, figsize=(6,6), fps=3, skip=0, max_duration=None):
    '''
    Creates the plot image for the given time step
    Input:
    - t (int): current time step
    - vessels (array): List of vessels in the scene
    - figsize (int): Size of figure
    - y_x_lim (int): limitation of x and y axis
    '''
    if not y_x_lim:
        y_x_lim = camera_rig.get_camera_position(0)[0] + 50
    fig, ax = plt.subplots(figsize=figsize)
    time_stamps = vessels[0].get_track().get_time_stamps()

    def make_frame(idiot_time):
        # Some hack to fix the time stamp index because of
        # the idiot FPS in videoClip that you can't change!!!!!
        temp = (int(idiot_time%1*10)-6)/3-1
        time_index = int((int(idiot_time)+1)*3+temp)
        time_index = time_index if skip == 0 else time_index*skip


        # Get time stamp
        t = time_stamps[time_index]

        # Clear
        ax.clear()

        # Plot
        for vessel in vessels:
            track = vessel.get_track().get_track_dict()
            points = np.array([[track[time]['center_position_m'][0], track[time]['center_position_m'][1]] for time in sorted(track.keys()) if time <= t])
            x = points[:, 0]
            y = points[:, 1]
            ax.plot(x, y)

            cornerpoints = vessel.calculate_2D_cornerpoints(t)
            xs = list(cornerpoints[:,0])+[cornerpoints[:,0][0]]
            ys = list(cornerpoints[:,1])+[cornerpoints[:,1][0]]
            ax.plot(xs, ys, 'b-')
        
        camera_position = camera_rig.get_camera_position(t)
        camera_orientation = camera_rig.get_camera_orientation(t)
        ax.plot(camera_position[0], camera_position[1], 'ro')
        ax.plot([camera_position[0],  camera_position[0]+camera_orientation[0]*50], [camera_position[1],  camera_position[1]+camera_orientation[1]*50], 'r-')

        x_lim = max(camera_position[0]+10, y_x_lim)
        y_lim = max(camera_position[1]+10, y_x_lim)
        ax.set_xlim([0,x_lim])
        ax.set_xlabel('x', fontsize = 14)
        ax.set_ylim([0,y_lim])
        ax.set_ylabel('y', fontsize = 14)
        ax.set_title(f'Camera position in scene {t}', fontsize=14)

        # returning numpy image
        return mplfig_to_npimage(fig)
    # creating animation
    duration = int(len(time_stamps)/3) # Because of the idiot FPS that i can't change!!!!
    if max_duration and max_duration/3<duration:
        duration = max_duration/3
    if skip != 0:
        duration = duration/skip
    animation = VideoClip(make_frame, duration = duration)
    gif_path = os.path.join(folder_path, 'camera_position.mp4')
    animation.write_videofile(gif_path,fps=fps)


###############################################################################################
#
#               Projection visualization
#
###############################################################################################
def visualize_projections_mov(all_projected_points, image_bounds, show_box=True, fastplot=False, folder_path='./gifs/', fps=3, skip=0, max_duration=None):
    '''
    Input:
    all_projected_points (List): List of lists of points for each vessel
    figsize (int): Size of figure
    image_bounds (Tuple): x and y pixel boundaries

    '''

    if fastplot:
        fig, ax = plt.subplots()
        fontsize = 10
        ticks_fontsize = 8
    else:
        figsize = (image_bounds[0]/200, image_bounds[1]/200)
        fig, ax = plt.subplots(figsize=figsize)
        fontsize = 28
        ticks_fontsize = 24

    time_stamps = list(all_projected_points.keys())
    # time_stamps.sort()

    def make_frame(idiot_time):
        # Some hack to fix the time stamp index because of
        # the idiot FPS in videoClip that you can't change!!!!!
        temp = (int(idiot_time%1*10)-6)/3-1
        time_index = int((int(idiot_time)+1)*3+temp)
        time_index = time_index if skip == 0 else time_index*skip

        # Get time stamp
        t = time_stamps[time_index]
        projected_points = all_projected_points[t]
        # Clear
        ax.clear()

        for vessel in projected_points.values():
            vessel_x = np.array([point.image_coordinate[0] for point in vessel if point.depth>=0])
            vessel_y = np.array([point.image_coordinate[1] for point in vessel if point.depth>=0])
            ax.plot(vessel_x, vessel_y, 'o')
            # Order of cornerpoints (length, beam, height): 
            # Front back lower, back back lower, 
            # back front lower, front front lower, 
            # Front back upper, back back upper, 
            # back front upper, front front upper,
            if show_box and vessel_x.size == 8:
                xs = list(vessel_x[0:4])+[vessel_x[0]]+list(vessel_x[4:])+[vessel_x[4]]
                ys = list(vessel_y[0:4])+[vessel_y[0]]+list(vessel_y[4:])+[vessel_y[4]]
                ax.plot(xs, ys, 'b-')
                ax.plot([vessel_x[1], vessel_x[5]], [vessel_y[1], vessel_y[5]], 'b-')
                ax.plot([vessel_x[2], vessel_x[6]], [vessel_y[2], vessel_y[6]], 'b-')
                ax.plot([vessel_x[3], vessel_x[7]], [vessel_y[3], vessel_y[7]], 'b-')
        
        ax.set_xlim([0,image_bounds[0]])
        ax.set_ylim([image_bounds[1],0])
        ax.set_ylabel('y', fontsize = fontsize)
        ax.xaxis.tick_top()
        ax.set_xlabel('x', fontsize = fontsize)    
        ax.xaxis.set_label_position('top') 
        ax.tick_params(labelsize=ticks_fontsize)
        ax.set_title(f'Projected points at time {t}', fontsize=fontsize)

        # returning numpy image
        return mplfig_to_npimage(fig)
    # creating animation
    duration = int(len(time_stamps)/3) # Because of the idiot FPS that i can't change!!!!
    if max_duration and max_duration/3<duration:
        duration = max_duration/3
    if skip !=0:
        duration = duration/skip
    animation = VideoClip(make_frame, duration = duration)
    gif_path = os.path.join(folder_path, 'projected_points.mp4')
    animation.write_videofile(gif_path,fps=fps)

def visualize_projections_json_mov(projected_points_path, image_bounds, show_box=True, fastplot=False, folder_path='./gifs/', fps=3, skip=0, max_duration=None):
    all_projected_points = json_to_projectedPoints(projected_points_path)
    visualize_projections_mov(all_projected_points, image_bounds, show_box=show_box, fastplot=fastplot, folder_path=folder_path, fps=fps, skip=skip, max_duration=max_duration)

###############################################################################################
#
#               Bounding box visualization
#
###############################################################################################

def visualize_annotations(annotations, image_bounds, classification=True, projected_points=None, show_projected_points=False, fastplot=False, folder_path='./gifs/', fps=3, skip=0, max_duration=None):
    '''
    Input:
    projected_points (List): List of lists of points for each vessel
    figsize (int): Size of figure
    image_bounds (Tuple): x and y pixel boundaries

    '''
    if fastplot:
        fig, ax = plt.subplots()
        fontsize = 10
        ticks_fontsize = 8
    else:
        figsize = (image_bounds[0]/200, image_bounds[1]/200)
        fig, ax = plt.subplots(figsize=figsize)
        fontsize = 28
        ticks_fontsize = 24

    time_stamps = list(annotations.keys())
    # time_stamps.sort()

    def make_frame(idiot_time):
        # Some hack to fix the time stamp index because of
        # the idiot FPS in videoClip that you can't change!!!!!
        temp = (int(idiot_time%1*10)-6)/3-1
        time_index = int((int(idiot_time)+1)*3+temp)
        time_index = time_index if skip == 0 else time_index*skip


        # Get time stamp
        t = time_stamps[time_index]
        annotations_t = get_dict_item(annotations, t)
        # Clear
        ax.clear()

        for vesselID, annot in annotations_t.items():
            bb = annot['bbox']
            xs, ys = bb.get_points_for_visualizing()
            ax.plot(xs, ys, '-', color=get_color(vesselID))
            if classification:
                ax.text(xs[1], ys[0]-5, annot['label'], color=get_color(vesselID))
            if show_projected_points:
                if not projected_points:
                    print("Provide projected points when show projected points is true")
                else:
                    vessel_proj = projected_points[t][vesselID]
                    x_vals = np.array([point.image_coordinate[0] for point in vessel_proj])
                    y_vals = np.array([point.image_coordinate[1] for point in vessel_proj])
                    ax.plot(x_vals, y_vals, 'o', color=get_color(vesselID))
        
        ax.set_xlim([0,image_bounds[0]])
        ax.set_ylim([image_bounds[1],0])
        ax.set_ylabel('y', fontsize = fontsize)
        ax.xaxis.tick_top()
        ax.set_xlabel('x', fontsize = fontsize)    
        ax.xaxis.set_label_position('top') 
        ax.tick_params(labelsize=ticks_fontsize)
        ax.set_title(f'Bounding boxes at time {t}', fontsize=fontsize)

        # returning numpy image
        return mplfig_to_npimage(fig)
    # creating animation
    duration = int(len(time_stamps)/3) # Because of the idiot FPS that i can't change!!!!
    if max_duration and max_duration/3<duration:
        duration = max_duration/3
    if skip !=0:
        duration = duration/skip
    animation = VideoClip(make_frame, duration = duration)
    gif_path = os.path.join(folder_path, 'boundingBoxes.mp4')
    animation.write_videofile(gif_path,fps=fps)

def visualize_annotations_json(annots_path, image_bounds, classification=True, pps_path = None, show_projected_points=False, fastplot=False, folder_path='./gifs/', fps=3, skip=0, max_duration=None):
    all_annots = json_to_annot(annots_path)
    all_pps = json_to_projectedPoints(pps_path) if (pps_path and show_projected_points) else None
    
    visualize_annotations(all_annots, image_bounds, classification=classification, projected_points=all_pps, show_projected_points=show_projected_points, fastplot=fastplot, folder_path=folder_path, fps=fps, skip=skip, max_duration=max_duration)

###############################################################################################
#
#               Distorted Bounding box visualization
#
###############################################################################################

def visualize_detections(detections, image_bounds, classification=True, annotations=None, show_annotations=False, folder_path='./gifs/', fps=3, fastplot=False, skip=0, max_duration=None):
    '''
    Input:
    projected_points (List): List of lists of points for each vessel
    figsize (int): Size of figure
    image_bounds (Tuple): x and y pixel boundaries

    '''
    if fastplot:
        fig, ax = plt.subplots()
        fontsize = 10
        ticks_fontsize = 8
    else:
        figsize = (image_bounds[0]/200, image_bounds[1]/200)
        fig, ax = plt.subplots(figsize=figsize)
        fontsize = 28
        ticks_fontsize = 24

    time_stamps = list(detections.keys())
    #time_stamps.sort()

    def make_frame(idiot_time):
        # Some hack to fix the time stamp index because of
        # the idiot FPS in videoClip that you can't change!!!!!
        temp = (int(idiot_time%1*10)-6)/3-1
        time_index = int((int(idiot_time)+1)*3+temp)
        time_index = time_index if skip == 0 else time_index*skip

        # Get time stamp
        t = time_stamps[time_index]
        detections_t = get_dict_item(detections, t)
        # Clear
        ax.clear()
        if show_annotations:
            if not annotations:
                print("Provide original BBs when show original BBs is true")
            else:
                annotations_t = get_dict_item(annotations,t)
                for vesselID, annot in annotations_t.items():
                    xs, ys = annot['bbox'].get_points_for_visualizing()
                    ax.plot(xs, ys, '-', color='lightgrey')
                    if classification:
                        ax.text(xs[1], ys[0]-5, annot['label'], color='lightgrey')
        for vesselID, detection in detections_t.items():
            xs, ys = detection['bbox'].get_points_for_visualizing()
            ax.plot(xs, ys, '-', color=get_color(vesselID))
            if classification:
                text = f"{detection['label']} {detection['confidenceScore']}" if detection['confidenceScore'] else f"{detection['label']}"
                ax.text(xs[1], ys[0]-5, text, color=get_color(vesselID))
    
        ax.set_xlim([0,image_bounds[0]])
        ax.set_ylim([image_bounds[1],0])
        ax.set_ylabel('y', fontsize = fontsize)
        ax.xaxis.tick_top()
        ax.set_xlabel('x', fontsize = fontsize)    
        ax.xaxis.set_label_position('top') 
        ax.tick_params(labelsize=ticks_fontsize)
        ax.set_title(f'Distorted bounding boxes at time {t}', fontsize=fontsize)

        # returning numpy image
        return mplfig_to_npimage(fig)
    # creating animation
    duration = int(len(time_stamps)/3) # Because of the idiot FPS that i can't change!!!!
    if max_duration and max_duration/3<duration:
        duration = max_duration/3
    if skip !=0:
        duration = duration/skip
    animation = VideoClip(make_frame, duration = duration)
    gif_path = os.path.join(folder_path, 'distortedBoundingBoxes.mp4')
    animation.write_videofile(gif_path,fps=fps)

def visualize_detections_json(detections_path, image_bounds, annotations_path=None, show_annotations=False, folder_path='./gifs/', fps=3, fastplot=False, skip=0, max_duration=None):
    detections = json_to_detection(detections_path)
    annotations = json_to_annot(annotations_path) if (show_annotations and annotations_path) else None
    visualize_detections(detections, image_bounds,  annotations=annotations, show_annotations=show_annotations, folder_path=folder_path, fps=fps, fastplot=fastplot, skip=skip, max_duration=max_duration)

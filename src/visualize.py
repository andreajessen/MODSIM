import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import os
# importing movie py libraries
from moviepy.editor import VideoClip, clips_array
from moviepy.video.io.bindings import mplfig_to_npimage
from utils import json_to_projectedPoints, json_to_annot, json_to_detection
import seaborn as sns
from matplotlib.collections import PolyCollection

###############################################################################################
#
#               Help functions for visualization
#
###############################################################################################
#plot_colors = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
plot_colors = sns.color_palette()[:7]+sns.color_palette()[8:]

vesselID2Color = {}
vessel_count = 0
drop_frames3 = []

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

def map_timestamp_to_video_time(time_stamps, fps, duration):
    '''
    Input:
    - all_projected_points (dict): {time_step: {vesselID: [ProjectedPoint1,..., ProjectedPoint8]}}
    - image_bound: (xmax, ymax)
    '''
    frames = {}
    idiot_time = 0.0
    for t in time_stamps:
        frames[round(idiot_time,3)] = t
        idiot_time += 1/fps
        if idiot_time > duration:
            return frames
    return frames
 
###############################################################################################
#
#               Dynamic scene visualization
#
###############################################################################################


def visualize_dynamic_scene_mov(vessels, folder_path='./gifs/', figsize=(6, 6), y_x_lim=400, fps=3, max_time_steps=None):
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
    duration = int(len(time_stamps)/fps) # Because of the idiot FPS that i can't change!!!!
    if max_time_steps and int(max_time_steps/fps)<duration:
        duration = int(max_time_steps/fps)
    
    frames = map_timestamp_to_video_time(time_stamps, fps, duration)

    def make_frame(idiot_time):
        # Get time stamp
        t = frames[round(idiot_time, 3)]

        # Clear
        ax.clear()

        # Plot
        ax.set_xlim([0,y_x_lim])
        ax.set_xlabel('x', fontsize = 14)
        ax.set_ylim([0,y_x_lim])
        ax.set_ylabel('y', fontsize = 14, rotation = 0)
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
    animation = VideoClip(make_frame, duration=duration)
    gif_path = os.path.join(folder_path, 'dynamicScene.mp4')
    animation.write_videofile(gif_path,fps=fps)

###############################################################################################
#
#               Camera position visualization
#
###############################################################################################
def visualize_camera_pose_in_dsg_mov(camera_rig, vessels, folder_path='./gifs', y_x_lim=None, figsize=(6,6), fps=3, max_time_steps=None, ownship_id=None, display_camera_pose=False, frequency=1):
    '''
    Creates the plot image for the given time step
    Input:
    - t (int): current time step
    - vessels (array): List of vessels in the scene
    - figsize (int): Size of figure
    - y_x_lim (int): limitation of x and y axis
    '''
    sns.set()

    if not y_x_lim:
        y_x_lim = camera_rig.get_camera_position(0,0)[0] + 50
    fig, ax = plt.subplots(figsize=figsize)

    time_stamps = vessels[0].get_track().get_time_stamps()
    duration = int(len(time_stamps)/fps) # Because of the idiot FPS that i can't change!!!!
    if max_time_steps and max_time_steps/fps<duration:
        duration = max_time_steps/fps
    frames = map_timestamp_to_video_time(time_stamps, fps, duration)

    def make_frame(idiot_time):
        # Get time stamp
        t = frames[round(idiot_time,3)]

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
            if ownship_id and ownship_id==vessel.id:
                ax.plot(xs, ys, 'b-', linewidth=2)
            else:
                ax.plot(xs, ys, 'k-', linewidth=1)
        for cameraID in camera_rig.cameras.keys():
            camera_position = camera_rig.get_camera_position(cameraID, t)
            if display_camera_pose:
                camera_orientation = camera_rig.get_camera_orientation(cameraID, t)
                ax.plot(camera_position[0], camera_position[1], 'ro')
                ax.plot([camera_position[0],  camera_position[0]+camera_orientation[0]*50], [camera_position[1],  camera_position[1]+camera_orientation[1]*50], 'r-')

        x_lim = max(camera_position[0]+10, y_x_lim)
        y_lim = max(camera_position[1]+10, y_x_lim)
        ax.set_xlim([0,x_lim])
        ax.set_xlabel('x', fontsize = 14)
        ax.set_ylim([0,y_lim])
        ax.set_ylabel('y', fontsize = 14, rotation = 0, labelpad=10)
        ax.set_title(f'Dynamic scene position at time {round(float(t)*frequency,1)}', fontsize=14)

        # returning numpy image
        return mplfig_to_npimage(fig)
    # creating animation
    animation = VideoClip(make_frame, duration = duration)
    gif_path = os.path.join(folder_path, 'camera_position.mp4')
    animation.write_videofile(gif_path,fps=fps)


###############################################################################################
#
#               Projection visualization
#
###############################################################################################
def vessels_in_view_pps(projected_points, image_bounds):
    '''
    Input:
    - projected_points (dict): {vesselID: [ProjectedPoint1,..., ProjectedPoint8]}
    - image_bound: (xmax, ymax)
    '''
    vessels = []
    def point_in_image(point):
        return point.depth>0 and 0<= point.image_coordinate[0] <= image_bounds[0] and 0<= point.image_coordinate[1] <= image_bounds[1]
    for vesselID, points in projected_points.items():
        for point in points:
            if point_in_image(point): 
                vessels.append(vesselID)
                break
    return vessels

def find_frames_pps_multiple_cameras(cameraIDs, all_projected_points, image_bounds, display_when_min_vessels, fps, max_time_steps):
    '''
    Input:
    - all_projected_points (dict): {time_step: {vesselID: [ProjectedPoint1,..., ProjectedPoint8]}}
    - image_bound: (xmax, ymax)
    '''
    frames_cam = {cameraID: {} for cameraID in cameraIDs}
    idiot_time = 0.0
    for t in all_projected_points[cameraIDs[0]].keys():
        vessels_in_image = 0 
        for cameraID in cameraIDs:
            vessels_in_image_cam = vessels_in_view_pps(all_projected_points[cameraID][t], image_bounds[cameraID])
            if len(vessels_in_image_cam) > vessels_in_image:
                vessels_in_image = len(vessels_in_image_cam)
        if vessels_in_image >= display_when_min_vessels:
            frame_time = round(idiot_time,3)
            for cameraID in cameraIDs:
                frames_cam[cameraID][frame_time] = {'time': t, 'pps': all_projected_points[cameraID][t]}
            idiot_time += 1/fps
        if max_time_steps and max_time_steps/fps<float(t):
            return frames_cam
    return frames_cam


def find_frames_pps(all_projected_points, image_bounds, display_when_min_vessels, fps, max_time_steps):
    '''
    Input:
    - all_projected_points (dict): {time_step: {vesselID: [ProjectedPoint1,..., ProjectedPoint8]}}
    - image_bound: (xmax, ymax)
    '''
    frames = {}
    idiot_time = 0.0
    for t, pps in all_projected_points.items():
        vessels_in_image = vessels_in_view_pps(pps, image_bounds)
        if len(vessels_in_image) >= display_when_min_vessels:
            frames[round(idiot_time,3)] = {'time': t, 'pps': pps}
            idiot_time += 1/fps
        if max_time_steps and max_time_steps/fps<float(t):
            return frames
    return frames

def visualize_projections_mov(all_projected_points, image_bounds, display_frames=None, 
                              horizon=None, show_box=True, fastplot=False, 
                              filename='./pps.mp4', fps=3, max_time_steps=None, 
                              display_when_min_vessels=0, frequency=1):
    '''
    Input:
    all_projected_points (List): List of lists of points for each vessel
    figsize (int): Size of figure
    image_bounds (Tuple): x and y pixel boundaries

    '''
    global drop_frames3
    sns.set()

    if fastplot:
        fig, ax = plt.subplots()
        fontsize = 10
        ticks_fontsize = 10
    else:
        figsize = (image_bounds[0]/200, image_bounds[1]/200)
        fig, ax = plt.subplots(figsize=figsize)
        fontsize = 20
        ticks_fontsize = 10

    frames = display_frames if display_frames else find_frames_pps(all_projected_points, image_bounds, display_when_min_vessels, fps, max_time_steps)


    if len(frames) == 0:
        print('No frames satisfy the minimum number of vessel requirement')
        return

    def make_frame(idiot_time):
        frame = frames[round(idiot_time,3)]
        # Get time stamp
        t = frame['time']
        projected_points = frame['pps']
        # Clear
        ax.clear()
        if horizon:
            horizon_points = get_dict_item(horizon,t)
            ax.axline(horizon_points[0].image_coordinate, horizon_points[1].image_coordinate, color='#85C1E9', alpha=0.4)
            
            # Define the x values
            x = np.linspace(0, image_bounds[0], 1000)

            # Define the y values of the horizon line
            m = (horizon_points[1].image_coordinate[1] - horizon_points[0].image_coordinate[1]) / (horizon_points[1].image_coordinate[0] - horizon_points[0].image_coordinate[0])
            b = horizon_points[0].image_coordinate[1] - m * horizon_points[0].image_coordinate[0]
            horizon_line = m * x + b

            # Fill the area below the horizon with green
            ax.fill_between(x, 0, horizon_line, color='#85C1E9', alpha=0.2)


            # Fill the area above the horizon with blue
            ax.fill_between(x, horizon_line, image_bounds[1], color='#3498DB', alpha=0.3)

                    
        for pps in projected_points.values():
            vessel_x = np.array([point.image_coordinate[0] for point in pps if point.depth>=0])
            vessel_y = np.array([point.image_coordinate[1] for point in pps if point.depth>=0])
            ax.plot(vessel_x, vessel_y, 'o')
            #ax.fill_between(vessel_x, vessel_y, color='grey', alpha=0.5)
            # Order of cornerpoints (length, beam, height): 
            # Front back lower, back back lower, 
            # back front lower, front front lower, 
            # Front back upper, back back upper, 
            # back front upper, front front upper,
            if show_box and vessel_x.size == 8:
                xs = list(vessel_x[0:4])+[vessel_x[0]]+list(vessel_x[4:])+[vessel_x[4]]
                ys = list(vessel_y[0:4])+[vessel_y[0]]+list(vessel_y[4:])+[vessel_y[4]]
                ax.plot(xs, ys, 'k-')
                ax.plot([vessel_x[1], vessel_x[5]], [vessel_y[1], vessel_y[5]], 'k-')
                ax.plot([vessel_x[2], vessel_x[6]], [vessel_y[2], vessel_y[6]], 'k-')
                ax.plot([vessel_x[3], vessel_x[7]], [vessel_y[3], vessel_y[7]], 'k-')

                                # Define the colors for each face
                grey_color = (0.7019607843137254, 0.7019607843137254, 0.7019607843137254)

                # Define the indices of the cube vertices that form each face
                face_indices = [[0, 1, 2, 3, 0],  # Front face
                                [1, 2, 6, 5, 1],  # Right face
                                [2, 3, 7, 6, 2],  # Back face
                                [3, 0, 4, 7, 3],  # Left face
                                [0, 1, 5, 4, 0],  # Bottom face
                                [4, 5, 6, 7, 4]]  # Top face

                # Fill each face of the cube with a different color
                for indices in face_indices:
                    face_x = [vessel_x[i] for i in indices]
                    face_y = [vessel_y[i] for i in indices]
                    ax.fill(face_x, face_y, color=grey_color, alpha=0.5)
                #ax.fill_between([xs[6], xs[5], xs[4], xs[7], xs[6]], [ys[6], ys[5], ys[4], ys[7], ys[6]], color='grey', alpha=0.5)
                #ax.fill_between([xs[6], xs[5]], [ys[6], ys[5]], color='red', alpha=0.5)
                #ax.fill_between([xs[0],xs[2]], [ys[0], ys[2]], color='green', alpha=0.5)
                #ax.fill_between([xs[1],xs[3]], [ys[1], ys[3]], color='orange', alpha=0.5)
                #ax.fill_between(xs[2:], ys[2:], color='yellow', alpha=0.5)
                #ax.fill_between(xs, ys, color='grey', alpha=0.5)
                #l = [[[xs[0], ys[0]], [xs[1], ys[1]]], [[xs[0], ys[0]], [xs[3], ys[3]]], [[xs[0], ys[0]], [xs[4], ys[4]]], [[xs[1], ys[1]], [xs[2], ys[2]]], [[xs[1], ys[1]], [xs[5], ys[5]]],[[xs[2], ys[2]], [xs[3], ys[3]]],[[xs[2], ys[2]], [xs[6], ys[6]]], [[xs[3], ys[3]], [xs[7], ys[7]]],[[xs[4], ys[4]], [xs[5], ys[5]]]]
                #pc2 = PolyCollection(l,  facecolors='red', edgecolor="k", alpha=0.9)
                #ax.add_collection(pc2)
                #ax.fill(pc2, facecolor='red')
        
        ax.set_xlim([0,image_bounds[0]])
        ax.set_ylim([image_bounds[1],0])
        ax.set_ylabel('y', fontsize = 14, rotation = 0, labelpad=10)
        ax.xaxis.tick_top()
        ax.set_xlabel('x', fontsize = 14)    
        ax.xaxis.set_label_position('top') 
        ax.tick_params(labelsize=ticks_fontsize)
        ax.set_title(f'Projected points at time {round(float(t)*frequency,1)}', fontsize=fontsize)

        # returning numpy image
        return mplfig_to_npimage(fig)
    
    # creating animation
    duration = int(len(frames)/fps) # Because of the idiot FPS that i can't change!!!!
    if max_time_steps and max_time_steps/fps<duration:
        duration = max_time_steps/fps
    animation = VideoClip(make_frame, duration = duration)
    animation.write_videofile(filename,fps=fps)
    return animation


def visualize_projections_json_mov(projected_points_path, image_bounds, display_frames = None, 
                                   horizon=None, show_box=True, fastplot=False, filename='./pps.mp4', 
                                   fps=3, max_time_steps=None, display_when_min_vessels=0, frequency=1):
    print('Loading projections from json')
    all_projected_points = json_to_projectedPoints(projected_points_path)
    print('Visualizing projections')
    return visualize_projections_mov(all_projected_points, image_bounds, horizon=horizon, 
                                     display_frames=display_frames, show_box=show_box, 
                                     fastplot=fastplot, filename=filename, fps=fps,
                                     max_time_steps=max_time_steps, 
                                     display_when_min_vessels=display_when_min_vessels, frequency=frequency)


def visualize_projections_multiple_cameras(camera_ids, projected_points_path, image_bounds, 
                                           horizons=None, show_box=True, fastplot=False, 
                                           folder_path='./gifs/', fps=3, skip=0, 
                                           max_time_steps=None, display_when_min_vessels=0, frequency=1):
    all_projected_points = {cameraID: json_to_projectedPoints(projected_points_path[cameraID]) for cameraID in camera_ids}
    frames = find_frames_pps_multiple_cameras(camera_ids, all_projected_points, image_bounds, display_when_min_vessels, fps, max_time_steps)
    clips = []
    for cameraID in camera_ids:
        horizon = horizons[cameraID] if horizons else None
        filename = os.path.join(folder_path, f'projectedPoints_C{cameraID}.mp4')
        animation = visualize_projections_json_mov(projected_points_path[cameraID], image_bounds[cameraID], 
                                                   display_frames=frames[cameraID], horizon=horizon, 
                                                   show_box=show_box, fastplot=fastplot, filename=filename, 
                                                   fps=fps, max_time_steps=max_time_steps, 
                                                   display_when_min_vessels=display_when_min_vessels, frequency=frequency)
        clips.append(animation)
    if len(clips)>1:
        final = clips_array([clips])
        filepath = os.path.join(folder_path, f'projectedPoints.mp4')
        final.write_videofile(filepath,fps=fps)
###############################################################################################
#
#               Bounding box visualization
#
###############################################################################################
def vessels_in_view_anns(annotations, image_bounds):
    '''
    Input:
    - annotations (dict): {vesselID: {label: string, 'bbox': BoundingBox}}
    - image_bound: (xmax, ymax)
    '''
    vessels = []
    def point_in_image(x, y):
        return 0<= x <= image_bounds[0] and 0<= y <= image_bounds[1]
    for vesselID, ann in annotations.items():
        xpoints, ypoints = ann['bbox'].get_points_for_visualizing()
        for i in range(len(xpoints)):
            if point_in_image(xpoints[i],ypoints[i]) and ann['bbox'].depth >= 0: 
                vessels.append(vesselID)
                break
    return vessels

def find_frames_anns(annotations, image_bounds, display_when_min_vessels, fps, max_time_steps):
    '''
    Input:
    - annotations (dict): {vesselID: {label: string, 'bbox': BoundingBox}}
    - image_bound: (xmax, ymax)
    - display_when_min_vessels (int)
    '''
    frames = {}
    idiot_time = 0.0
    for t, anns in annotations.items():
        vessels_in_image = vessels_in_view_anns(anns, image_bounds)
        if len(vessels_in_image) >= display_when_min_vessels:
            frames[round(idiot_time,3)] = {'time': t, 'anns': anns}
            idiot_time += 1/fps
        if max_time_steps and max_time_steps/fps<float(t):
            return frames
    return frames

def find_frames_anns_multiple_cameras(cameraIDs, annotations, image_bounds, display_when_min_vessels, fps, max_time_steps):
    '''
    Input:
    - all_projected_points (dict): {time_step: {vesselID: [ProjectedPoint1,..., ProjectedPoint8]}}
    - image_bound: (xmax, ymax)
    '''
    frames_cam = {cameraID: {} for cameraID in cameraIDs}
    idiot_time = 0.0
    for t, anns in annotations[cameraIDs[0]].items():
        vessels_in_image = 0 
        for cameraID in cameraIDs:
            vessels_in_image_cam = vessels_in_view_anns(annotations[cameraID][t], image_bounds[cameraID])
            if len(vessels_in_image_cam) > vessels_in_image:
                vessels_in_image = len(vessels_in_image_cam)
        if vessels_in_image >= display_when_min_vessels:
            frame_time = round(idiot_time,3)
            for cameraID in cameraIDs:
                frames_cam[cameraID][frame_time] = {'time': t, 'anns': annotations[cameraID][t]}
            idiot_time += 1/fps
        if max_time_steps and max_time_steps/fps<float(t):
            return frames_cam
    return frames_cam

def visualize_annotations(annotations, image_bounds, display_frames = None, horizon=None, classification=True, projected_points=None, show_projected_points=False, fastplot=False, filename='./gifs.mp4', fps=3, max_time_steps=None, display_when_min_vessels=0, step=1):
    '''
    Input:
    projected_points (List): List of lists of points for each vessel
    figsize (int): Size of figure
    image_bounds (Tuple): x and y pixel boundaries

    '''
    if fastplot:
        fig, ax = plt.subplots()
        fontsize = 10
        ticks_fontsize = 10
    else:
        figsize = (image_bounds[0]/200, image_bounds[1]/200)
        fig, ax = plt.subplots(figsize=figsize)
        fontsize = 20
        ticks_fontsize = 10
    


    frames = display_frames if display_frames else find_frames_anns(annotations, image_bounds, display_when_min_vessels, fps, max_time_steps)
    if len(frames) == 0:
        print('No frames satisfy the minimum number of vessel requirement')
        return

    def make_frame(idiot_time):
        frame = frames[round(idiot_time,3)]
        annotations_t = frame['anns']
        t = frame['time']
        # Clear
        ax.clear()
        if horizon:
            horizon_points = get_dict_item(horizon,t)
            ax.axline(horizon_points[0].image_coordinate, horizon_points[1].image_coordinate)
        for vesselID, annot in annotations_t.items():
            bb = annot['bbox']
            xs, ys = bb.get_points_for_visualizing()
            ax.plot(xs, ys, '-', color=get_color(vesselID))
            ax.fill(xs, ys, color=get_color(vesselID))
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
        ax.set_ylabel('y', fontsize = 14, rotation = 0)
        ax.xaxis.tick_top()
        ax.set_xlabel('x', fontsize = 14)    
        ax.xaxis.set_label_position('top') 
        ax.tick_params(labelsize=ticks_fontsize)
        ax.set_title(f'Annotations at time {t}', fontsize=fontsize)

        # returning numpy image
        return mplfig_to_npimage(fig)
    # creating animation
    duration = int(len(frames)/fps) # Because of the idiot FPS that i can't change!!!!
    if max_time_steps and max_time_steps/fps<duration:
        duration = max_time_steps/fps
    animation = VideoClip(make_frame, duration = duration)
    animation.write_videofile(filename,fps=fps)
    return animation

def visualize_annotations_json(annots_path, image_bounds, display_frames=None, horizon=None, classification=True, pps_path = None, show_projected_points=False, fastplot=False, filename='annot.mp4', fps=3, max_time_steps=None, display_when_min_vessels=0, step=1):
    all_annots = json_to_annot(annots_path)
    all_pps = json_to_projectedPoints(pps_path) if (pps_path and show_projected_points) else None
    return visualize_annotations(all_annots, image_bounds, display_frames=display_frames, horizon=horizon, classification=classification, projected_points=all_pps, show_projected_points=show_projected_points, fastplot=fastplot, filename=filename, fps=fps, max_time_steps=max_time_steps, display_when_min_vessels=display_when_min_vessels, step=step)

def visualize_annotations_multiple_cameras(camera_ids, annots_path, image_bounds, horizons=None, classification=True, pps_path = None, show_projected_points=False, fastplot=False, folder_path='./gifs/', fps=3, max_time_steps=None, display_when_min_vessels=0, step=1):
    all_annots = {cameraID: json_to_annot(annots_path[cameraID]) for cameraID in camera_ids}
    
    frames = find_frames_anns_multiple_cameras(camera_ids, all_annots, image_bounds, display_when_min_vessels, fps, max_time_steps)
    if len(frames) == 0:
        print('No frames satisfy the minimum number of vessel requirement')
        return
    clips = []
    for cameraID in camera_ids:
        filename = os.path.join(folder_path, f'annotations_C{cameraID}.mp4')
        horizon = horizons[cameraID] if horizons else None
        pps_path = pps_path[cameraID] if pps_path else None
        animation = visualize_annotations_json(annots_path[cameraID], image_bounds[cameraID], display_frames = frames[cameraID], horizon=horizon, classification=classification, pps_path = pps_path, show_projected_points=show_projected_points, fastplot=fastplot, filename=filename, fps=fps, max_time_steps=max_time_steps, display_when_min_vessels=display_when_min_vessels, step=step)
        clips.append(animation)
    if len(clips)>1:
        final = clips_array([clips])
        filepath = os.path.join(folder_path, f'annotations.mp4')
        final.write_videofile(filepath,fps=fps)



###############################################################################################
#
#               Distorted Bounding box visualization
#
###############################################################################################

def find_frames_detections(detections, annotations, image_bounds, display_when_min_vessels, fps, max_time_steps):
    '''
    Input:
    - detections (dict): {vesselID: {label: string, 'bbox': BoundingBox, confidenceScore: float}}
    - image_bound: (xmax, ymax)
    - display_when_min_vessels (int)
    '''
    frames = {}
    idiot_time = 0.0
    if not annotations:
        # Include all frames
        for t, detections in detections.items():
            frames[round(idiot_time,3)] = {'time': t, 'detections': detections}
            idiot_time += 1/fps
        return frames
    for t, detections in detections.items():
        vessels_in_image = vessels_in_view_anns(annotations[t], image_bounds)
        if len(vessels_in_image) >= display_when_min_vessels:
            frames[round(idiot_time,3)] = {'time': t, 'detections': detections}
            idiot_time += 1/fps
        if max_time_steps and max_time_steps/fps<float(t):
            return frames
    return frames

def find_frames_detections_multiple_cameras(cameraIDs, detections, annotations, image_bounds, display_when_min_vessels, fps, max_time_steps, skip_ten=False):
    '''
    Input:
    - all_projected_points (dict): {time_step: {vesselID: [ProjectedPoint1,..., ProjectedPoint8]}}
    - image_bound: (xmax, ymax)
    '''
    frames_cam = {cameraID: {} for cameraID in cameraIDs}
    idiot_time = 0.0
    if not annotations:
        # Include all frames
        for t in detections[cameraIDs[0]].keys():
            for cameraID in cameraIDs:
                frames_cam[cameraID][round(idiot_time,3)] = {'time': t, 'detections': detections[cameraID][t]}
            idiot_time += 1/fps
        return frames_cam

    for t in detections[cameraIDs[0]].keys():
        if skip_ten and float(t)%1!=0: continue
        vessels_in_image = 0 
        for cameraID in cameraIDs:
            vessels_in_image_cam = vessels_in_view_anns(annotations[cameraID][t], image_bounds[cameraID])
            if len(vessels_in_image_cam) > vessels_in_image:
                vessels_in_image = len(vessels_in_image_cam)
        if vessels_in_image >= display_when_min_vessels:
            frame_time = round(idiot_time,3)
            for cameraID in cameraIDs:
                frames_cam[cameraID][frame_time] = {'time': t, 'detections': detections[cameraID][t]}
            idiot_time += 1/fps
        if max_time_steps and max_time_steps/fps<float(t):
            return frames_cam
    return frames_cam

def visualize_detections(detections, image_bounds, display_frames=None, frequency=1, temporal_state_history=None, temporal_state_names=None, horizon=None, classification=True, annotations=None, show_annotations=False, display_when_min_vessels=0, filepath='detections.mp4', fps=3, fastplot=False, max_time_steps=None):
    '''
    Input:
    projected_points (List): List of lists of points for each vessel
    figsize (int): Size of figure
    image_bounds (Tuple): x and y pixel boundaries

    '''
    sns.set()
    if fastplot:
        fig, ax = plt.subplots()
        fontsize = 10
        ticks_fontsize = 10
    else:
        figsize = (image_bounds[0]/200, image_bounds[1]/200)
        fig, ax = plt.subplots(figsize=figsize)
        fontsize = 20
        ticks_fontsize = 10

    frames = display_frames if display_frames else find_frames_detections(detections, annotations, image_bounds, display_when_min_vessels, fps, max_time_steps)
    if len(frames) == 0:
        print('No frames satisfy the minimum number of vessel requirement')
        return
    
    def make_frame(idiot_time):
        # Some hack to fix the time stamp index because of
        # the idiot FPS in videoClip that you can't change!!!!!
        frame = frames[round(idiot_time,3)]

        # Get time stamp
        t = frame['time']
        detections_t = frame['detections']
        # Clear
        ax.clear()
        if horizon:
            horizon_points = get_dict_item(horizon,t)
            ax.axline(horizon_points[0].image_coordinate, horizon_points[1].image_coordinate, color='#85C1E9', alpha=0.4)
            
            # Define the x values
            x = np.linspace(0, image_bounds[0], 1000)

            # Define the y values of the horizon line
            m = (horizon_points[1].image_coordinate[1] - horizon_points[0].image_coordinate[1]) / (horizon_points[1].image_coordinate[0] - horizon_points[0].image_coordinate[0])
            b = horizon_points[0].image_coordinate[1] - m * horizon_points[0].image_coordinate[0]
            horizon_line = m * x + b

            # Fill the area below the horizon with green
            ax.fill_between(x, 0, horizon_line, color='#85C1E9', alpha=0.2)


            # Fill the area above the horizon with blue
            ax.fill_between(x, horizon_line, image_bounds[1], color='#3498DB', alpha=0.3)
        if show_annotations:
            if not annotations:
                print("Provide original BBs when show original BBs is true")
            else:
                annotations_t = get_dict_item(annotations,t)
                for vesselID, annot in annotations_t.items():
                    xs, ys = annot['bbox'].get_points_for_visualizing()
                    grey_color = (0.7019607843137254, 0.7019607843137254, 0.7019607843137254)
                    ax.plot(xs, ys, '-', color=grey_color)
                    ax.fill(xs, ys, color=grey_color)
                    if classification:
                        ax.text(xs[1], ys[0]-5, annot['label'], color='grey')
        for vesselID, detection in detections_t.items():
            xs, ys = detection['bbox'].get_points_for_visualizing()
            ax.plot(xs, ys, '-', color=get_color(vesselID))
            ax.fill(xs, ys, color=get_color(vesselID))
            if classification:
                text = f"{detection['label']} {detection['confidenceScore']}" if detection['confidenceScore'] else f"{detection['label']}"
                ax.text(xs[1], ys[0]-5, text, color=get_color(vesselID))
    
        ax.set_xlim([0,image_bounds[0]])
        ax.set_ylim([image_bounds[1],0])
        ax.set_ylabel('y', fontsize = 14, rotation = 0, labelpad=10)
        ax.xaxis.tick_top()
        ax.set_xlabel('x', fontsize = 14)    
        ax.xaxis.set_label_position('top') 
        ax.tick_params(labelsize=ticks_fontsize)
        if temporal_state_history and temporal_state_names:
            current_state = get_dict_item(temporal_state_history, t)
            state_name = get_dict_item(temporal_state_names, current_state)
            plt.suptitle(f'Detections at time {round(float(t)*frequency,1)}', fontsize=24)
            title = f'In state {current_state}: {state_name}'
            ax.set_title(title, fontsize=15)
        else:
            title = f'Detections at time {round(float(t)*frequency,1)}'
            ax.set_title(title, fontsize=fontsize)

        # returning numpy image
        return mplfig_to_npimage(fig)
    # creating animation
    duration = int(len(frames)/fps) # Because of the idiot FPS that i can't change!!!!
    if max_time_steps and max_time_steps/fps<duration:
        duration = max_time_steps/fps
    animation = VideoClip(make_frame, duration = duration)
    animation.write_videofile(filepath,fps=fps)
    return animation


def visualize_detections_json(detections_path, image_bounds, frequency=1, display_frames=None, temporal_state_history=None, temporal_state_names=None, horizon=None, annotations_path=None, show_annotations=False, filepath='.detection.mp4', fps=3, fastplot=False, max_time_steps=None, display_when_min_vessels=0):
    detections = json_to_detection(detections_path)
    annotations = json_to_annot(annotations_path) if (show_annotations and annotations_path) else None
    return visualize_detections(detections, image_bounds, frequency=frequency, display_frames=display_frames, temporal_state_history=temporal_state_history, temporal_state_names=temporal_state_names, horizon=horizon, annotations=annotations, show_annotations=show_annotations, filepath=filepath, fps=fps, fastplot=fastplot, display_when_min_vessels=display_when_min_vessels, max_time_steps=max_time_steps)

def visualize_detections_multiple_cameras(camera_ids, detections_paths, image_bounds, temporal_state_history=None, 
                                          temporal_state_names=None, horizons=None, annotations_paths=None, 
                                          show_annotations=False, folder_path='./', fps=3, fastplot=False, 
                                          max_time_steps=None, display_when_min_vessels=0, skip_ten=False, frequency=1):
    detections = {cameraID: json_to_detection(detections_paths[cameraID]) for cameraID in camera_ids}
    annotations = {cameraID: json_to_annot(annotations_paths[cameraID]) for cameraID in camera_ids} if (show_annotations and annotations_paths) else None
    frames = find_frames_detections_multiple_cameras(camera_ids, detections, annotations, image_bounds, display_when_min_vessels, fps, max_time_steps, skip_ten=skip_ten)
    clips = []
    for cameraID in camera_ids:
        filename = os.path.join(folder_path, f'detections_C{cameraID}.mp4')
        image_bound = image_bounds[cameraID]
        horizon = horizons[cameraID]
        animation = visualize_detections_json(detections_paths[cameraID], image_bound, frequency=frequency, display_frames=frames[cameraID], temporal_state_history=temporal_state_history, temporal_state_names=temporal_state_names, horizon=horizon, annotations_path=annotations_paths[cameraID], show_annotations=show_annotations, filepath=filename, fps=fps, fastplot=fastplot, display_when_min_vessels=display_when_min_vessels, max_time_steps=max_time_steps)
        clips.append(animation)
    if len(clips)>1:
        final = clips_array([clips])
        filepath = os.path.join(folder_path, f'detections.mp4')
        final.write_videofile(filepath,fps=fps)

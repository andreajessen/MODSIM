import matplotlib.pyplot as plt
import imageio

def visualize(vessels, gif_path='./dynamicSceneExample.gif', figsize=(6, 6), y_x_lim=15, fps=3):
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
    fig = plt.figure(figsize=figsize)
    for vessel in vessels:
        x = vessel.get_track().get_x_values()
        y = vessel.get_track().get_y_values()
        cornerpoints = vessel.get_track().get_cornerpoints_values()        
        plt.plot(x[:(t+1)], y[:(t+1)])
        plt.plot(x[t], y[t], marker = 'o' )
        # Create square plot for shape of vessel
        xs = list(cornerpoints[t][:,0])+[cornerpoints[t][:,0][0]]
        ys = list(cornerpoints[t][:,1])+[cornerpoints[t][:,1][0]]
        plt.plot(xs, ys, 'b-')
    plt.xlim([-y_x_lim,y_x_lim])
    plt.xlabel('x', fontsize = 14)
    plt.ylim([-y_x_lim,y_x_lim])
    plt.ylabel('y', fontsize = 14)
    plt.title(f'Relationship between x and y at step {t}', fontsize=14)
    plt.savefig(f'./img/img_{t}.png', transparent = False,  facecolor = 'white')
    plt.close()
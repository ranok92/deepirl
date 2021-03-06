import pygame
from gridworld_drone import GridWorldDrone

from drone_data_utils import preprocess_data_from_control_points
import numpy as np 

import argparse
import os

import matplotlib
import datetime, time
#from debugtools import compile_results

parser = argparse.ArgumentParser()

parser.add_argument('--splines', type=int, default=3)
parser.add_argument('--world-size', type=int, nargs="*", default=[100, 100], help="The world size is of the\
                    format [cols, rows]")
parser.add_argument('--interpoint-frames', type=int, default=30)
parser.add_argument('--file-name', type=str, default='Another_spline_file.txt')


args = parser.parse_args()

game_display = pygame.display.set_mode((args.world_size[0], args.world_size[1]))
game_display.fill(((255, 255, 255)), [0,0, args.world_size[0], args.world_size[1]])
pygame.display.update()
record_flag = False
done =  False
traj_counter = 0

file_io = open(args.file_name, 'w')

per_frame_file = args.file_name+'_per_frame.txt'
file_io.write('{} - the number of splines\n'.format(args.splines))
color_tuple = [0, 0, 0]
update_color = False
while traj_counter <  args.splines:

    if update_color:
        color_tuple[traj_counter%3] = (color_tuple[traj_counter%3] + 120)%255
        update_color=False

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            key = pygame.key.get_pressed()
            if key[pygame.K_s]:
                prev_point = None
                spline_point_counter = 0
                traj_point_container = []
                record_flag = True
            
            if key[pygame.K_q]:
                record_flag = False
                traj_counter += 1
                update_color=True
                file_io.write("{} - Num of control points\n".format(spline_point_counter))
                for i in range(len(traj_point_container)):
                    point = traj_point_container[i]
                    file_io.write("{} {} {} {} - (2D pont, m_id)\n".format(float(point[0]), 
                                                                        float(point[1]), 
                                                                        args.interpoint_frames*i,
                                                                        0))

        if record_flag:
            if event.type == pygame.MOUSEBUTTONDOWN: 
                (x,y) = pygame.mouse.get_pos()
                print(x,y)
                print(color_tuple)
                #for visualizing the points of the trajectory
                pygame.draw.rect(game_display, tuple(color_tuple), [x-5, y-5, 10, 10])
                if prev_point is not None:
                    pygame.draw.line(game_display, tuple(color_tuple), prev_point, (x,y))
                prev_point = (x,y)
                pygame.display.update()
                #transform the points
                x = x - (args.world_size[0]/2)
                y = -(y - (args.world_size[1]/2))

                spline_point_counter += 1
                traj_point_container.append((x,y))

file_io.close()
preprocess_data_from_control_points(args.file_name, 1, world_size=args.world_size)
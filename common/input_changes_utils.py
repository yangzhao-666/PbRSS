import numpy as np
import random
import sys
sys.path.append('../gym_sokoban/envs/')

from render_utils import room_to_rgb
from utilities.downScale import downScale

def room_states2feature_planes(room_states):
    #convert HxWx3 to HxWx6, 6 channels are feature planes, includes:
    #1. walls; 2. empty goal squares; 3. boxes on empty squares; 4. boxes on goal squares; 5. player-reachable cells; 6. player-reachable cells on goal squares;
    #input:HWC state, c=3
    #output:HWC state, c=6
    splited_room_states = []
    for room_state in room_states:
        walls, goal_squares, boxes_empty, boxes_on_goal = get_features(room_state)
        splited_room_states.append([walls, goal_squares, boxes_empty, boxes_on_goal])
    return splited_room_states

def get_features(room_state):
    walls = []
    goal_squares = []
    boxes_empty = []
    boxes_on_goal = []
    
    for row in room_state:
        walls_r = []
        goal_squares_r = []
        boxes_empty_r = []
        boxes_on_goal_r = []
        for col in row:
            if col == 0:
                walls_r.append(0)
                goal_squares_r.append(1)
                boxes_empty_r.append(1)
                boxes_on_goal_r.append(1)
            elif col == 1:
                walls_r.append(1)
                goal_squares_r.append(1)
                boxes_empty_r.append(1)
                boxes_on_goal_r.append(1)
            elif col == 2:
                goal_squares_r.append(2)
                boxes_on_goal_r.append(2)
                walls_r.append(1)
                boxes_empty_r.append(1)
            elif col == 3:
                boxes_empty_r.append(4)
                boxes_on_goal_r.append(3)
                walls_r.append(1)
                goal_squares_r.append(1)
            elif col == 4:
                boxes_empty_r.append(4)
                boxes_on_goal_r.append(4)
                walls_r.append(1)
                goal_squares_r.append(1)
            elif col == 5:
                walls_r.append(5)
                goal_squares_r.append(5)
                boxes_empty_r.append(5)
                boxes_on_goal_r.append(5)
            elif col == 6:
                goal_squares_r.append(6)
                walls_r.append(1)
                boxes_empty_r.append(1)
                boxes_on_goal_r.append(1)
        walls.append(walls_r)
        goal_squares.append(goal_squares_r)
        boxes_empty.append(boxes_empty_r)
        boxes_on_goal.append(boxes_on_goal_r)
        
        rgb_walls = room_to_rgb(walls)
        rgb_goal_squares = room_to_rgb(goal_squares)
        rgb_boxes_empty = room_to_rgb(boxes_empty)
        rgb_boxes_on_goal = room_to_rgb(boxes_on_goal)

        grey_walls = rgb2grey(rgb_walls)
        grey_goal_squares = rgb2grey(rgb_goal_squares)
        grey_boxes_empty = rgb2grey(rgb_boxes_empty)
        grey_boxes_on_goal = rgb2grey(rgb_boxes_on_goal)

    return [grey_walls, grey_goal_squares, grey_boxes_empty, grey_boxes_on_goal]

def get_walls(room_state):
    room = []
    for row in room_state:
        r = []
        for col in row:
            if col == 0:
                r.append(0)
            elif col == 5:
                r.append(5)
            else:
                r.append(1)
        room.append(r)
        rgb_room = room_to_rgb(room)
    return rgb_room

def get_goal_squares(room_state):
    room = []
    for row in room_state:
        r = []
        for col in row:
            if col == 2:
                r.append(2)
            elif col == 5:
                r.append(5)
            elif col == 6:
                r.append(6)
            else:
                r.append(1)
        room.append(r)
        rgb_room = room_to_rgb(room)
    return rgb_room

def get_boxes_empty(room_state):
    room = []
    for row in room_state:
        r = []
        for col in row:
            if col == 3 or col == 4:
                r.append(4)
            elif col == 5:
                r.append(5)
            else:
                r.append(1)
        room.append(r)
        rgb_room = room_to_rgb(room)
    return rgb_room

def get_boxes_on_goal(room_state):
    room = []
    for row in room_state:
        r = []
        for col in row:
            if col == 3:
                r.append(3)
            elif col == 4:
                r.append(4)
            elif col == 5:
                r.append(5)
            elif col == 2:
                r.append(2)
            else:
                r.append(1)
        room.append(r)
        rgb_room = room_to_rgb(room)
    return rgb_room

def get_player_reachable(room_state):
    room = []
    for row in room_state:
        r = []
        for col in row:
            #need to be considered;
            if col == 1:
                r.append(1)
            elif col == 5:
                r.append(5)
            else:
                r.append(1)
        room.append(r)
        rgb_room = room_to_rgb(room)
    return rgb_room

def get_reachable_goal(room_state):
    room = []
    for row in room_state:
        r = []
        for col in row:
            if col == 2:
                r.append(2)
            elif col == 1:
                r.append(1)
            elif col == 5:
                r.append(5)
            else:
                r.append(1)
        room.append(r)
        rgb_room = room_to_rgb(room)
    return rgb_room

def meta_converter(action, state):
    #convert the state into different sub-cases locally;
    #input: HWC state
    #output: HWC sub-state
    raise NotImplementedError

def rgb2grey(hwc_img):

    down_img = downScale(hwc_img)
    r = down_img[:, :, 0]
    g = down_img[:, :, 1]
    b = down_img[:, :, 2]

    r = (r *.299)
    g = (g *.587)
    b = (b *.114)

    avg = (r + g + b)

    return avg

def room_states2feature_planes_list(room_states): #room_states: a list of room_state
    #import ipdb; ipdb.set_trace()
    input_rooms_features = room_states2feature_planes(room_states)
    rooms_features = []
    for room in input_rooms_features:
        room_features = []
        for feature in room[0:4]: #we only use the first 4 features;
            #tensor_room_feature = torch.FloatTensor(np.float32(rgb2grey(feature)))
            room_feature = rgb2grey(feature)
            #room_features.append(tensor_room_feature)
            room_features.append(room_feature)
        #tensor_room_features = torch.stack(room_features)
        #rooms_features.append(tensor_room_features)
        rooms_features.append(room_features)
    #tensor_rooms_features = torch.stack(rooms_features)
    #return tensor_rooms_features
    return rooms_features

'''
@Author: Michiel and Zhao
@Date: 2021.4.5
@Descriptions: This file will include functions like checking if there's a possible irreversible action, in order to decide when to take the expert action, ...
'''
from .sokoban import legalActions
from .sokoban import PosOfPlayer
from .sokoban import PosOfBoxes
from .sokoban import PosOfWalls
from .sokoban import PosOfGoals
from .sokoban import isFailed
from .sokoban import updateState
from .sokoban import aStarSearch_detector
from .sokoban import get_solution

def check_next_irreversible(gameState):
    '''
    check if there's a irreversible action at next step. If the given state is already unsolvable, return True.
    '''

    posWalls = PosOfWalls(gameState)
    posBoxes = PosOfBoxes(gameState)
    posPlayer = PosOfPlayer(gameState)
    posGoals = PosOfGoals(gameState)

    if isFailed(posBoxes, posWalls, posGoals):
        #print('Voice from the expert: the game is already a dud, I couldnt do anything...')
        return True
    
    ############# To use isFailed detector to detect deadlock ##################
    #get all legal actions
    actions = legalActions(posPlayer, posBoxes, posWalls)
    #take each action, and check if the new state is fail, if yes, return True.
    for action in actions:
        newPosPlayer, newPosBox = updateState(posPlayer, posBoxes, action)
        if isFailed(newPosBox, posWalls, posGoals):
            return True

    ############# If isFailed didn't work, use aStarSearch to detect deadlock ##################
    for action in actions:
        newPosPlayer, newPosBox = updateState(posPlayer, posBoxes, action)
        solution = aStarSearch_detector(newPosPlayer,newPosBox, posWalls, posGoals)
        if solution == [0]:
            return True

    return False

def get_distance(gameState):
    '''
    return the distance from the current state to the goal state;
    '''
    solution = get_solution(gameState, 'astar')
    dis = len(solution)
    if solution == [0]:
        return -1
    return dis

'''
@Author: KnightofLuna
@Project Link: https://github.com/KnightofLuna/sokoban-solver
@Modified by: Zhao
@Date: 2021.3.31
'''

import sys
import collections
import numpy as np
import heapq
import time

class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 1   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 0 # wall
            elif layout[irow][icol] == '@': layout[irow][icol] = 5 # player
            elif layout[irow][icol] == '$': layout[irow][icol] = 4 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 2 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 3 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 
    return np.array(layout)

def PosOfPlayer(gameState):
    """Return the position of agent"""
    #return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)
    return tuple(np.argwhere((gameState == 5))[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    #return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 3))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 0)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    no_agent_on_target = tuple(tuple(x) for x in np.argwhere((gameState == 2) | (gameState == 3)))
    if len(no_agent_on_target) != len(PosOfBoxes(gameState)):
        return tuple(tuple(x) for x in np.argwhere((gameState == 2) | (gameState == 3) | (gameState == 5))) # e.g. like those above
    return no_agent_on_target # e.g. like those above

def isEndState(posBox, posGoals):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox, posWalls):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox, posWalls):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox, posWalls):
            legalActions.append(action)
        else: 
            continue     
    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox, posWalls, posGoals):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def breadthFirstSearch(gameState, posWalls, posGoals):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox) # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    frontier = collections.deque([[startState]]) # store states
    actions = collections.deque([[0]]) # store actions
    exploredSet = set()
    while frontier:
        node = frontier.popleft()
        node_action = actions.popleft() 
        if isEndState(node[-1][-1], posGoals):
            #print(','.join(node_action[1:]).replace(',',''))
            solution = ','.join(node_action[1:]).replace(',','')
            return solution
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1], posWalls):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox, posWalls, posGoals):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])

def depthFirstSearch(gameState, posWalls, posGoals):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]] 
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1], posGoals):
            #print(','.join(node_action[1:]).replace(',',''))
            solution = ','.join(node_action[1:]).replace(',','')
            return solution
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1], posWalls):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox, posWalls, posGoals):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])

def heuristic(posPlayer, posBox, posGoals):
    """A heuristic function to calculate the overall distance between the else boxes and the else goals"""
    distance = 0
    completes = set(posGoals) & set(posBox)
    sortposBox = list(set(posBox).difference(completes))
    sortposGoals = list(set(posGoals).difference(completes))
    for i in range(len(sortposBox)):
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1]))
    return distance

def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])

def uniformCostSearch(gameState, posWalls, posGoals):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = PriorityQueue()
    frontier.push([startState], 0)
    exploredSet = set()
    actions = PriorityQueue()
    actions.push([0], 0)
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1], posGoals):
            #print(','.join(node_action[1:]).replace(',',''))
            solution = ','.join(node_action[1:]).replace(',','')
            return solution
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            Cost = cost(node_action[1:])
            for action in legalActions(node[-1][0], node[-1][1], posWalls):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox, posWalls, posGoals):
                    continue
                frontier.push(node + [(newPosPlayer, newPosBox)], Cost)
                actions.push(node_action + [action[-1]], Cost)

def aStarSearch(gameState, posWalls, posGoals):
    """Implement aStarSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    start_state = (beginPlayer, beginBox)
    frontier = PriorityQueue()
    frontier.push([start_state], heuristic(beginPlayer, beginBox, posGoals))
    exploredSet = set()
    actions = PriorityQueue()
    actions.push([0], heuristic(beginPlayer, start_state[1], posGoals))
    while not frontier.isEmpty():
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1], posGoals):
            #print(','.join(node_action[1:]).replace(',',''))
            solution = ','.join(node_action[1:]).replace(',','')
            return solution
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            Cost = cost(node_action[1:])
            for action in legalActions(node[-1][0], node[-1][1], posWalls):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox, posWalls, posGoals):
                    continue
                Heuristic = heuristic(newPosPlayer, newPosBox, posGoals)
                frontier.push(node + [(newPosPlayer, newPosBox)], Heuristic + Cost) 
                actions.push(node_action + [action[-1]], Heuristic + Cost)
    #if the search doesnt find the solution, which means deadlock detection didn't find the deadlock, and there's no sloution;
    return [0]


def aStarSearch_detector(beginPlayer, beginBox, posWalls, posGoals):
    """Implement aStarSearch approach"""

    start_state = (beginPlayer, beginBox)
    frontier = PriorityQueue()
    frontier.push([start_state], heuristic(beginPlayer, beginBox, posGoals))
    exploredSet = set()
    actions = PriorityQueue()
    actions.push([0], heuristic(beginPlayer, start_state[1], posGoals))
    while not frontier.isEmpty():
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1], posGoals):
            #print(','.join(node_action[1:]).replace(',',''))
            solution = ','.join(node_action[1:]).replace(',','')
            return solution
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            Cost = cost(node_action[1:])
            for action in legalActions(node[-1][0], node[-1][1], posWalls):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox, posWalls, posGoals):
                    continue
                Heuristic = heuristic(newPosPlayer, newPosBox, posGoals)
                frontier.push(node + [(newPosPlayer, newPosBox)], Heuristic + Cost) 
                actions.push(node_action + [action[-1]], Heuristic + Cost)
    #if the search doesnt find the solution, which means deadlock detection didn't find the deadlock, and there's no sloution;
    return [0]
"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('sokobanLevels/'+options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

#added by Zhao, get the solution given a game state;
def get_solution(gameState, method):
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    posBoxes = PosOfBoxes(gameState)
    #isFailed detection could only detect 5 cases. if it didn't find deadlock, and there is a deadlock, search will return [0].
    if isFailed(posBoxes, posWalls, posGoals):
        #print('Voice from the expert: the game is already a dud, I couldnt do anything...')
        return [0]
    if method == 'astar':
        solution = aStarSearch(gameState, posWalls, posGoals)
    elif method == 'dfs':
        solution = depthFirstSearch(gameState, posWalls, posGoals)
    elif method == 'bfs':
        solution = breadthFirstSearch(gameState, posWalls, posGoals)
    elif method == 'ucs':
        solution = uniformCostSearch(gameState, posWalls, posGoals)
    else:
        raise ValueError('Invalid method.')
    return solution

if __name__ == '__main__':
    time_start = time.time()
    layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState(layout)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    if method == 'astar':
        aStarSearch()
    elif method == 'dfs':
        depthFirstSearch()
    elif method == 'bfs':
        breadthFirstSearch()
    elif method == 'ucs':
        uniformCostSearch()
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))

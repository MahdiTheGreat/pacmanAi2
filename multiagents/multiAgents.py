# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math
import treelib



from game import Agent


def euclideanDistance(position, nextPosition):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = nextPosition
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

def manhattanDistance(position, nextPosition):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = nextPosition
    print()
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        print()
        distanceFuncs=[euclideanDistance,manhattanDistance]
        currentDistanceFunc=distanceFuncs[0]

        foodPositions=[]
        foodGrid=newFood.data
        ghostPositions=[]
        ghostDistances=[]
        sumGhostDistance = 0
        for i in range(len(newGhostStates)):
            ghostPosition=newGhostStates[i].configuration.pos
            ghostPositions.append(ghostPosition)
            ghostDistances.append(currentDistanceFunc(newPos,ghostPosition))
            sumGhostDistance += ghostDistances[len(ghostDistances)-1]

        print()
        for i in range(len(foodGrid)):
           for j in range(len(foodGrid[i])):
               if foodGrid[i][j]:foodPositions.append((i,j))


        sumFoodDistance = 0
        foodDistances=[]

        for i in range(len(foodPositions)):
           foodDistances.append(currentDistanceFunc(newPos, foodPositions[i]))
           sumFoodDistance += foodDistances[len(foodDistances)-1]

        dangerFlag=False
        for i in range(len(ghostPositions)):
            if euclideanDistance(newPos,ghostPositions[i])<3:
             dangerFlag=True
#
        if dangerFlag:return -math.inf
        #else:return successorGameState.getScore()+(math.inf if len(foodDistances) else 1/min(foodDistances))
        else:
            return successorGameState.getScore() + (math.inf if len(foodDistances)==0 else 1 / min(foodDistances))
#
        #elif sumFoodDistance == 0:
        #    return math.inf
        #else:


        print()

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

def stateIdMaker(state, mode=2):

    if mode == 0:
        return id(state)
    elif mode == 1:
        return hash(state)
    elif mode == 2:
        return state



class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    #def getActionsAndSuccessors(self, gameState,agentIndex=0):
    #    "Returns successor states, the actions they require, and a cost of 1."
    #    successors = []
    #    actions = gameState.getLegalActions(self)
    #    for action in actions:
    #        successors.append([action,gameState.generateSuccessor(agentIndex, action)])
    #    print()
    #    return successors


    #def maxValue(self,state,tree):
    #    agentIndex=0
    #    if state.isWin() or state.isLose() or tree.depth()>self.depth: return self.evaluationFunction(state)
    #    v=-math.inf
    #    actionsAndSuccessors=self.getActionsAndSuccessors(state)
    #    for actionAndSuccessor in actionsAndSuccessors:
    #        score=self.minValue(actionAndSuccessor[1],tree,agentIndex)
    #        tree.create_node(identifier=actionAndSuccessor[1],parent=state,data=[score,actionAndSuccessor[0]])
    #        v=max(v,score)
    #    return v

    def terminalStateChecker(self,state,tree):
        return state.isWin() or state.isLose() or tree.depth(stateIdMaker(state))>(self.depth)

    def maxValue(self,state,tree,agentIndex=0):
       if self.terminalStateChecker(state,tree):return self.evaluationFunction(state)
       v=-math.inf
       actions = state.getLegalActions(agentIndex)

       for action in actions:
           successor=state.generateSuccessor(agentIndex, action)
           if not tree.contains(stateIdMaker(successor)):
               tree.create_node(identifier=stateIdMaker(successor), parent=stateIdMaker(state), data=[action])
           score=self.minValue(successor,tree,agentIndex+1)
           tree.get_node(stateIdMaker(successor)).data.append(score)
           v=max(v,score)

       return v

    def minValue(self, state, tree,agentIndex):
        if self.terminalStateChecker(state,tree): return self.evaluationFunction(state)
        v = math.inf
        agentNum=state.getNumAgents()
        currentAgentIndex=agentIndex % (agentNum-1)
        actions = state.getLegalActions(agentIndex)
        print()

        for action in actions:
            successor = state.generateSuccessor(agentIndex, action)

            if not tree.contains(stateIdMaker(successor)):
                tree.create_node(identifier=stateIdMaker(successor), parent=stateIdMaker(state), data=[action])

            if currentAgentIndex:score = self.minValue(successor,tree,currentAgentIndex+1)
            else: score=self.maxValue(successor, tree)

            tree.get_node(stateIdMaker(successor)).data.append(score)
            v = min(v, score)
        return v


    def getAction(self, gameState):

        tree = treelib.Tree()
        if tree.contains(stateIdMaker(gameState)):
            print()

        tree.create_node(identifier=stateIdMaker(gameState),parent=None,data=None)
        self.maxValue(gameState,tree)
        sucessors=tree.children(stateIdMaker(gameState))

        max=sucessors[0]
        for sucessor in sucessors:
            if sucessor.data[1]>=max.data[1]:max=sucessor

        return max.data[0]


        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction

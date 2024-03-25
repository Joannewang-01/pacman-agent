# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint
from queue import PriorityQueue

NUM_TRAINING = 0
TRAINING = False


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]             
##########
# Agents #
##########

from contest.util import nearestPoint,Queue
from queue import PriorityQueue
class Node:
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost
class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

    def search_and_decision_tree(self, game_state):
        """
        A* heuristic search plus a simple decision tree
        """
        # Implement A* heuristic search plus a simple decision tree here
        a_star_path = self.a_star_search(game_state)

        # Decision tree approach
        if self.is_repeating_actions(game_state):
            return self.find_different_attack_path(game_state)
        elif self.has_few_allied_food(game_state):
            return self.defend_remaining_food(game_state)
        elif self.is_agent_pacman(game_state):
            if self.is_agent_being_chased(game_state):
                return self.escape_status(game_state)
            elif self.has_agent_crossed_border_recently(game_state):
                return self.find_different_attack_path(game_state)
            else:
                return self.eat_enemy_food(game_state)
        else:
            return self.eat_enemy_food(game_state)

    def a_star_search(self, game_state):
        """
        A* heuristic search implementation
        """
        start_node = Node(game_state)
        frontier = PriorityQueue()
        frontier.put(start_node)
        explored = set()

        while not frontier.empty():
            current_node = frontier.get()

            if current_node.state.is_win() or current_node.state.is_lose():
                return self.get_path(current_node)

            explored.add(current_node.state)

            for action in current_node.state.get_legal_actions():
                successor_state = current_node.state.generate_successor(action)
                if successor_state not in explored:
                    cost = current_node.cost + 1
                    successor_node = Node(successor_state, current_node, action, cost)
                    frontier.put(successor_node)

    def get_path(self, node):
        """
        Get the path from the root node to the given node
        """
        path = []
        while node.parent is not None:
            path.append(node.action)
            node = node.parent
        return path[::-1]


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def registerInitialState(self, gameState):
        self.epsilon = 0.1
        self.alpha = 0.2
        self.discount = 0.9
        self.numTraining = NUM_TRAINING
        self.episodesSoFar = 0

        self.weights = {'closest-food': -3.099192562140742,
                        'bias': -9.280875042529367,
                        '#-of-ghosts-1-step-away': -16.6612110039328,
                        'eats-food': 11.127808437648863}

        self.start = gameState.getAgentPosition(self.index)
        self.featuresExtractor = FeaturesExtractor(self)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
            Picks among the actions with the highest Q(s,a).
        """
        legalActions = gameState.getLegalActions(self.index)
        if len(legalActions) == 0:
            return None

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in legalActions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        action = None
        if TRAINING:
            for action in legalActions:
                self.updateWeights(gameState, action)
        if not util.flipCoin(self.epsilon):
            # exploit
            action = self.getPolicy(gameState)
        else:
            # explore
            action = random.choice(legalActions)
        return action

    def getWeights(self):
        return self.weights

    def getQValue(self, gameState, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # features vector
        features = self.featuresExtractor.getFeatures(gameState, action)
        return features * self.weights

    def update(self, gameState, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.featuresExtractor.getFeatures(gameState, action)
        oldValue = self.getQValue(gameState, action)
        futureQValue = self.getValue(nextState)
        difference = (reward + self.discount * futureQValue) - oldValue
        # for each feature i
        for feature in features:
            newWeight = self.alpha * difference * features[feature]
            self.weights[feature] += newWeight
        # print(self.weights)

    def updateWeights(self, gameState, action):
        nextState = self.getSuccessor(gameState, action)
        reward = self.getReward(gameState, nextState)
        self.update(gameState, action, nextState, reward)

    def getReward(self, gameState, nextState):
        reward = 0
        agentPosition = gameState.getAgentPosition(self.index)

        # check if I have updated the score
        if self.getScore(nextState) > self.getScore(gameState):
            diff = self.getScore(nextState) - self.getScore(gameState)
            reward = diff * 10

        # check if food eaten in nextState
        myFoods = self.getFood(gameState).asList()
        distToFood = min([self.getMazeDistance(agentPosition, food) for food in myFoods])
        # I am 1 step away, will I be able to eat it?
        if distToFood == 1:
            nextFoods = self.getFood(nextState).asList()
            if len(myFoods) - len(nextFoods) == 1:
                reward = 10

        # check if I am eaten
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(ghosts) > 0:
            minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in ghosts])
            if minDistGhost == 1:
                nextPos = nextState.getAgentState(self.index).getPosition()
                if nextPos == self.start:
                    # I die in the next state
                    reward = -100

        return reward

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        CaptureAgent.final(self, state)
        # print(self.weights)
        # did we finish training?

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def computeValueFromQValues(self, gameState):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        allowedActions = gameState.getLegalActions(self.index)
        if len(allowedActions) == 0:
            return 0.0
        bestAction = self.getPolicy(gameState)
        return self.getQValue(gameState, bestAction)

    def computeActionFromQValues(self, gameState):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = gameState.getLegalActions(self.index)
        if len(legalActions) == 0:
            return None
        actionVals = {}
        bestQValue = float('-inf')
        for action in legalActions:
            targetQValue = self.getQValue(gameState, action)
            actionVals[action] = targetQValue
            if targetQValue > bestQValue:
                bestQValue = targetQValue
        bestActions = [k for k, v in actionVals.items() if v == bestQValue]
        # random tie-breaking
        return random.choice(bestActions)

    def getPolicy(self, gameState):
        return self.computeActionFromQValues(gameState)

    def getValue(self, gameState):
        return self.computeValueFromQValues(gameState)

    class FeaturesExtractor:

        def __init__(self, agentInstance):
            self.agentInstance = agentInstance

        def getFeatures(self, gameState, action):
            # extract the grid of food and wall locations and get the ghost locations
            food = self.agentInstance.getFood(gameState)
            walls = gameState.getWalls()
            enemies = [gameState.getAgentState(i) for i in self.agentInstance.getOpponents(gameState)]
            ghosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
            # ghosts = state.getGhostPositions()

            features = util.Counter()

            features["bias"] = 1.0

            # compute the location of pacman after he takes the action
            agentPosition = gameState.getAgentPosition(self.agentInstance.index)
            x, y = agentPosition
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)

            # count the number of ghosts 1-step away
            features["#-of-ghosts-1-step-away"] = sum(
                (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

            # if len(ghosts) > 0:
            #   minGhostDistance = min([self.agentInstance.getMazeDistance(agentPosition, g) for g in ghosts])
            #   if minGhostDistance < 3:
            #     features["minGhostDistance"] = minGhostDistance

            # successor = self.agentInstance.getSuccessor(gameState, action)
            # features['successorScore'] = self.agentInstance.getScore(successor)

            # if there is no danger of ghosts then add the food feature
            if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
                features["eats-food"] = 1.0

            # capsules = self.agentInstance.getCapsules(gameState)
            # if len(capsules) > 0:
            #   closestCap = min([self.agentInstance.getMazeDistance(agentPosition, cap) for cap in self.agentInstance.getCapsules(gameState)])
            #   features["closestCapsule"] = closestCap

            dist = self.closestFood((next_x, next_y), food, walls)
            if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features["closest-food"] = float(dist) / (walls.width * walls.height)
            features.divideAll(10.0)
            # print(features)
            return features

        def closestFood(self, pos, food, walls):
            """
            closestFood -- this is similar to the function that we have
            worked on in the search project; here its all in one place
            """
            fringe = [(pos[0], pos[1], 0)]
            expanded = set()
            while fringe:
                pos_x, pos_y, dist = fringe.pop(0)
                if (pos_x, pos_y) in expanded:
                    continue
                expanded.add((pos_x, pos_y))
                # if we find a food at this location then exit
                if food[pos_x][pos_y]:
                    return dist
                # otherwise spread out from the location to its neighbours
                nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
                for nbr_x, nbr_y in nbrs:
                    fringe.append((nbr_x, nbr_y, dist + 1))
            # no food found
            return None
     ######## update get_features  function again #########
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}

    def classical_planning(self, game_state):
        """
        Classical planning using PDDL
        """
        # Implement classical planning using PDDL here
        pass


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):

        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

    def approximate_q_learning(self, game_state):
        """
        Approximate Q-Learning
        """
        # Implement Approximate Q-Learning here
        pass

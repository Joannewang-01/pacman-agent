# myTeam_update.py
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


from contest.pacman import Directions
from contest.game import Agent
import pickle
#import numpy as np
import random, time, sys
from contest.capture import GameState,GameStateData
import contest.util as util
import contest.distanceCalculator as distanceCalculator
from contest.captureAgents import CaptureAgent
from contest.game import Actions
from contest.game import Directions, Agent
from contest.util import nearestPoint,Counter 
from queue import PriorityQueue
from collections import defaultdict
from collections import deque
import random

import math

#################
# Team creation #
#################

TRAINING = False


def create_team(first_index, second_index, isRed,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', **args):
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

    # The following line is an example only; feel free to change it.
    return [eval(first)(first_index), eval(second)(second_index)]



class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions.
    """

    def register_initial_state(self, game_state):
        """
        Register the initial state of the agent.
        This method is called at the beginning of each game to set up the necessary variables.
        """
        # Store the agent's initial position
        self.initial_position = game_state.get_agent_position(self.index)

        # Call the parent class's register_initial_state method
        CaptureAgent.register_initial_state(self, game_state)

        # Store the game map walls
        self.walls = game_state.get_walls()

        # Store the list of legal positions on the game map
        self.legal_positions = game_state.get_walls().as_list(False)

        # Initialize the Bayesian Inference for the ghost positions
        self.obs = {enemy: util.Counter() for enemy in self.get_opponents(game_state)}
        for enemy in self.get_opponents(game_state):
            self.obs[enemy][game_state.get_initial_agent_position(enemy)] = 1.0

    def elapse_time(self, enemy, game_state):
        """
        Update the agent's belief about the opponent's position over time.
        This is part of the Bayesian Inference process used to track the positions of the ghosts.
        """
        # Define a lambda function to calculate possible next positions
        possible_positions = lambda x, y: [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        # Initialize a counter to store the updated belief distribution
        all_obs = util.Counter()

        # Iterate over the previous positions and their probabilities
        for prev_pos, prev_prob in self.obs[enemy].items():
            # Calculate the new possible positions for the enemy
            new_obs = util.Counter(
                {pos: 1.0 for pos in possible_positions(prev_pos[0], prev_pos[1]) if pos in self.legal_positions})

            # Normalize the new observation probabilities
            new_obs.normalize()

            # Update the overall belief distribution with the new probabilities
            for new_pos, new_prob in new_obs.items():
                all_obs[new_pos] += new_prob * prev_prob

        # Check for any food eaten by opponents
        foods = self.get_food_you_are_defending(game_state).as_list()
        prev_foods = self.get_food_you_are_defending(
            self.get_previous_observation()).as_list() if self.get_previous_observation() else []

        # If the number of foods has decreased, adjust the belief distribution
        if len(foods) < len(prev_foods):
            eaten_food = set(prev_foods) - set(foods)
            for food in eaten_food:
                all_obs[food] = 1.0 / len(self.get_opponents(game_state))

        # Update the agent's belief about the opponent's positions
        self.obs[enemy] = all_obs

    def observe_action(self, enemy, game_state):
        """
        Updates beliefs based on the distance observation and Pacman's position.
        This is part of the Bayesian Inference process used to track the positions of the ghosts.
        """
        # Get distance observations for all agents
        all_noise = game_state.get_agent_distances()
        noisy_distance = all_noise[enemy]
        my_pos = game_state.get_agent_position(self.index)
        team_idx = [index for index, value in enumerate(game_state.teams) if value]
        team_pos = [game_state.get_agent_position(team) for team in team_idx]

        # Initialize a counter to store the updated belief distribution
        all_obs = util.Counter()

        # Iterate over all legal positions on the board
        for pos in self.legal_positions:
            # Check if any teammate is close to the current position
            team_dist = [team for team in team_pos if util.manhattanDistance(team, pos) <= 5]

            if team_dist:
                # If a teammate is close, set the probability of this position to 0
                all_obs[pos] = 0.0
            else:
                # Calculate the true distance between Pacman and the current position
                true_distance = util.manhattanDistance(my_pos, pos)

                # Get the probability of observing the noisy distance given the true distance
                pos_prob = game_state.get_distance_prob(true_distance, noisy_distance)

                # Update the belief distribution with the calculated probability
                all_obs[pos] = pos_prob * self.obs[enemy][pos]

        # Check if there are any non-zero probabilities in the belief distribution
        if all_obs.totalCount():
            # Normalize the belief distribution if there are non-zero probabilities
            all_obs.normalize()

            # Update the agent's belief about the opponent's positions
            self.obs[enemy] = all_obs

    def choose_action(self, game_state):
        """
        Choose the best action for the agent based on the current game state.
        This method is called during the game to determine the agent's next move.
        """
        # Get the legal actions the agent can take
        actions = game_state.get_legal_actions(self.index)

        # Evaluate the value of each action and store the results
        #start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        #print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        # Select the best action based on the highest Q-value
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        # If there are only 2 or fewer food left, return to the initial position
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.initial_position, pos2)
                if dist < best_dist:
                    bestAction = action
                    best_dist = dist
            return bestAction

        # Return a random best action
        return random.choice(bestActions)

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
        Evaluate the value of an action in a given game state.
        """
        features = self.get_features(game_state, action)
        weights = self.getWeights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Compute and return a set of features that describe the game state after taking a given action.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successorScore'] = self.get_score(successor)
        return features

    def getWeights(self, game_state, action):
        """
        Get the weights for the features used in the evaluation function.
        """
        return {'successorScore': 1.0}


class OffensiveReflexAgent(CaptureAgent):
    def load_weights(self):
        """
        Load the pre-trained weights for the Q-learning algorithm.
        These weights were learned during the training phase and will be used to guide the agent's actions.
        """
        weights = {
            'bias': -9.1234412,
            'nearest_dot': -2.983928392083,
            'nearest_ghosts': -3.65065432233,
            'eaten_dot': 15.12232122121,
            'camp_return': 1.822389123231
        }
        return weights

    def register_initial_state(self, game_state):
        """
        Initialize the agent's state and parameters for the Q-learning algorithm.
        This method is called at the beginning of each game to set up the necessary variables.
        """
        # Set the exploration rate based on whether we are in training mode or not
        if TRAINING:
            self.epsilon = 0.15  # Exploration rate during training
        else:
            self.epsilon = 0  # No exploration during evaluation

        # Set the learning rate and discount factor for the Q-learning algorithm
        self.alpha = 0.2
        self.discount = 0.8

        # Load the pre-trained weights
        self.weights = self.load_weights()

        # Store the agent's initial position and the list of legal positions on the game map
        self.initial_position = game_state.get_agent_position(self.index)
        self.legal_positions = game_state.get_walls().as_list(False)

        # Call the parent class's register_initial_state method
        CaptureAgent.register_initial_state(self, game_state)

        # Initialize the Bayesian Inference for the ghost positions
        self.obs = {enemy: util.Counter() for enemy in self.get_opponents(game_state)}
        for enemy in self.get_opponents(game_state):
            self.obs[enemy][game_state.get_initial_agent_position(enemy)] = 1.0

    def camp_return_action(self, game_state):
        """
        Choose the action that brings the agent closer to its initial position.
        This is useful when the agent is carrying a significant amount of food and needs to return to the home base.
        """
        best_dist = 10000
        for action in game_state.get_legal_actions(self.index):
            successor = self.get_successor(game_state, action)
            pos2 = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(self.initial_position, pos2)
            if dist < best_dist:
                bestAction = action
                best_dist = dist
        return bestAction

    def choose_action(self, game_state):
        """
        Choose an action based on the Q-values and exploration-exploitation strategy.
        This method is called during the game to determine the agent's next move.
        """
        action = None
        legal_actions = game_state.get_legal_actions(self.index)

        # If there are no legal actions, return None
        if len(legal_actions) == 0:
            return None

        # If we are in training mode, update the weights based on the current state and actions
        if TRAINING:
            for action in legal_actions:
                self.update_weights(game_state, action)

        # Determine whether to exploit or explore based on the epsilon value
        if not util.flipCoin(self.epsilon):
            # Exploit: Choose the action with the highest Q-value
            action = self.action_from_q_values(game_state)
        else:
            # Explore: Randomly choose an action
            action = random.choice(legal_actions)

        return action

    def distance_of_ghost(self, agentPos, ghostPos, steps, walls):
        """
        Check if a ghost is within a specified number of steps from the agent.
        This is used to determine the number of ghosts in proximity to the agent.
        """
        distance = self.get_maze_distance(agentPos, ghostPos)
        return distance <= steps

    def number_of_ghost(self, game_state, action):
        """
        Get the number of ghosts in proximity after taking a specific action.
        This information is used to calculate the features for the Q-learning algorithm.
        """
        food = self.get_food(game_state)
        walls = game_state.get_walls()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        enemies_idx = [i for i in self.get_opponents(game_state)]
        ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() is not None]

        max_vals = list()
        if len(ghosts) == 0:
            for e_idx in enemies_idx:
                self.observe_action(e_idx, game_state)
                self.elapse_time(e_idx, game_state)
                belief_dist_e = self.obs[e_idx]
                max_position, max_prob = max(belief_dist_e.items(), key=lambda item: item[1])
                max_vals.append(max_position)
            ghosts = list(set(max_vals))

        agentPosition = game_state.get_agent_position(self.index)
        x, y = agentPosition
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        return sum(self.distance_of_ghost((next_x, next_y), g, 3, walls) for g in ghosts)

    def capture_camp_return_feature(self, game_state, agent_position, action):
        """
        Calculate a feature indicating the desirability of going near home when carrying food.
        This feature is used to encourage the agent to return to its home base when it is carrying a significant amount of food.
        """
        x, y = agent_position
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        original_agent_state = game_state.get_agent_state(self.index)
        amount_of_food_carrying = original_agent_state.num_carrying

        return amount_of_food_carrying / -(
                (game_state.get_walls().width / 3) - self.get_maze_distance(self.initial_position,
                                                                            (next_x, next_y)))

    def get_features(self, game_state, action):
        """
        Compute and return a set of features that describe the game state after taking a given action.
        These features are used by the Q-learning algorithm to determine the best action to take.
        """
        features = util.Counter()
        agent_position = game_state.get_agent_position(self.index)
        x, y = agent_position
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        features["bias"] = 1.0
        features["score"] = 1.0
        features["nearest_ghosts"] = self.number_of_ghost(game_state, action)
        features["eaten_dot"] = 1.0

        dist = self.nearest_dot_action((next_x, next_y), self.get_food(game_state), game_state.get_walls())
        if dist is not None:
            features["nearest_dot"] = float(dist) / (game_state.get_walls().width * game_state.get_walls().height)

        features["camp_return"] = self.capture_camp_return_feature(game_state, agent_position, action)

        return features

    def nearest_dot_action(self, pos, food, walls):
        """
        Find the distance to the nearest food dot.
        This is used to calculate the "nearest_dot" feature for the Q-learning algorithm.
        """
        frontier = [(pos[0], pos[1], 0)]
        expanded = set()
        while frontier:
            pos_x, pos_y, dist = frontier.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))

            if food[pos_x][pos_y]:
                return dist

            nbrs = Actions.get_legal_neighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                frontier.append((nbr_x, nbr_y, dist + 1))
        return None

    def elapse_time(self, enemy, game_state):
        """
        Update the agent's belief about the opponent's position over time.
        This is part of the Bayesian Inference process used to track the positions of the ghosts.
        """
        possible_positions = lambda x, y: [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        all_obs = util.Counter()

        for prev_pos, prev_prob in self.obs[enemy].items():
            new_obs = util.Counter(
                {pos: 1.0 for pos in possible_positions(prev_pos[0], prev_pos[1]) if pos in self.legal_positions})
            new_obs.normalize()
            for new_pos, new_prob in new_obs.items():
                all_obs[new_pos] += new_prob * prev_prob

        foods = self.get_food_you_are_defending(game_state).as_list()
        prev_foods = self.get_food_you_are_defending(
            self.get_previous_observation()).as_list() if self.get_previous_observation() else []

        if len(foods) < len(prev_foods):
            eaten_food = set(prev_foods) - set(foods)
            for food in eaten_food:
                all_obs[food] = 1.0 / len(self.get_opponents(game_state))

        self.obs[enemy] = all_obs

    def observe_action(self, enemy, game_state):
        """
        Observe the opponent's position and update the agent's belief.
        This is part of the Bayesian Inference process used to track the positions of the ghosts.
        """
        all_noise = game_state.get_agent_distances()
        noisy_distance = all_noise[enemy]
        my_pos = game_state.get_agent_position(self.index)
        team_idx = [index for index, value in enumerate(game_state.teams) if value]
        team_pos = [game_state.get_agent_position(team) for team in team_idx]

        all_obs = util.Counter()

        for pos in self.legal_positions:
            team_dist = [team for team in team_pos if team is not None and util.manhattanDistance(team, pos) <= 5]

            if team_dist:
                all_obs[pos] = 0.0
            else:
                true_distance = util.manhattanDistance(my_pos, pos)
                pos_prob = game_state.get_distance_prob(true_distance, noisy_distance)
                all_obs[pos] = pos_prob * self.obs[enemy][pos]

        if all_obs.totalCount():
            all_obs.normalize()
            self.obs[enemy] = all_obs

    def get_q_value(self, game_state, action):
        """
        Calculate the Q-value for a given state and action.
        This is used to determine the best action to take based on the current state.
        """
        features = self.get_features(game_state, action)
        return features * self.weights

    def update(self, game_state, action, nextState, reward):
        """
        Update the weights based on the transition.
        This is the core of the Q-learning algorithm, where the agent updates its weights based on the
        current state, action, next state, and the reward received.
        """
        features = self.get_features(game_state, action)
        oldValue = self.get_q_value(game_state, action)
        futureQValue = self.capture_q_values(game_state)
        difference = (reward + self.discount * futureQValue) - oldValue
        for feature in features:
            if feature not in self.weights:
                self.weights[feature] = 0  # Initialize with a default value, like 0
            newWeight = self.alpha * difference * features[feature]
            self.weights[feature] += newWeight

    def update_weights(self, game_state, action):
        """
        Update the weights based on the current state and action.
        This method is called during the training phase to update the agent's weights.
        """
        nextState = self.get_successor(game_state, action)
        reward = self.get_reward(game_state, nextState)
        self.update(game_state, action, nextState, reward)

    def get_reward(self, game_state, nextState):
        """
        Calculate the total reward for a given state transition.
        This method combines the various reward components, such as the score reward, the distance to food reward,
        the camp return reward, and the defend reward, to determine the overall reward for the transition.
        """
        agent_position = game_state.get_agent_position(self.index)
        go_home_reward = self.camp_return_reward(nextState)
        score_reward = self.score_reward(game_state, nextState)
        dist_to_food_reward = self.dotdistance_reward(game_state, nextState, agent_position)
        enemies_reward = self.defend_reward(game_state, nextState, agent_position)

        # Display individual rewards for debugging purposes
        rewards = {"enemies": enemies_reward, "go_home": go_home_reward, "dist_to_food_reward": dist_to_food_reward,
                   "score": score_reward}
        print("REWARDS:", rewards)

        return sum(rewards.values())

    def camp_return_reward(self, nextState):
        """
        Calculate the reward for going near home when carrying food.
        This reward encourages the agent to return to its home base when it is carrying a significant amount of food.
        """
        original_agent_state = nextState.get_agent_state(self.index)
        amount_of_food_carrying = original_agent_state.num_carrying
        agent_position = nextState.get_agent_position(self.index)

        return amount_of_food_carrying / -(
                (nextState.get_walls().width / 3) - self.get_maze_distance(self.initial_position, agent_position))

    def score_reward(self, game_state, nextState):
        """
        Calculate the reward based on the change in score from the current state to the successor state.
        This reward encourages the agent to take actions that increase the team's score.
        """
        score_reward = 0

        # Check if the score has increased
        if self.get_score(nextState) > self.get_score(game_state):
            # Calculate the difference in score
            diff = self.get_score(nextState) - self.get_score(game_state)

            # Update the score reward based on the team color
            score_reward += diff * 20 if self.red else -diff * 20

        return score_reward

    def dotdistance_reward(self, game_state, nextState, agent_position):
        """
        Calculate the reward based on the change in distance to the nearest food.
        This reward encourages the agent to move towards the nearest food dot.
        """
        dist_to_food_reward = 0

        # Get the list of coordinates of the agent's food in the current state
        my_foods = self.get_food(game_state).as_list()

        # Get the minimum distance to food in the current state
        dist_to_food = min([self.get_maze_distance(agent_position, food) for food in my_foods])

        # Check if the agent reached a food in the next state
        if dist_to_food == 1:
            # Get the list of coordinates of the agent's food in the next state
            next_foods = self.get_food(nextState).as_list()

            # Check if one food was eaten in the next state
            if len(my_foods) - len(next_foods) == 1:
                # Update the dist_to_food_reward
                dist_to_food_reward += 20

        return dist_to_food_reward

    def defend_reward(self, game_state, nextState, agent_position):
        """
        Calculate the reward based on the proximity to enemies (ghosts) in the current and next states.
        This reward encourages the agent to stay away from ghosts and defend its home base.
        """
        enemies_reward = 0

        # Get the states of enemies in the current state
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

        # Get the positions of ghosts among enemies
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        # Check if there are ghosts in the current state
        if len(ghosts) > 0:
            # Get the minimum distance to a ghost in the current state
            min_dist_ghost = min([self.get_maze_distance(agent_position, g.get_position()) for g in ghosts])

            # Check if the agent is one step away from a ghost in the next state and going home
            if min_dist_ghost == 1:
                next_pos = nextState.get_agent_state(self.index).get_position()
                if next_pos == self.initial_position:
                    # Update the enemies_reward
                    enemies_reward = -50

        return enemies_reward

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

    def final(self, state):
        """
        Called at the end of each game.
        """
        CaptureAgent.final(self, state)
        with open('QLearning_trained_weights.pkl', 'wb') as file:
            pickle.dump(self.weights, file)

    def capture_q_values(self, game_state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        allowedActions = game_state.get_legal_actions(self.index)
        if len(allowedActions) == 0:
            return 0.0
        bestAction = self.action_from_q_values(game_state)
        return self.get_q_value(game_state, bestAction)

    def action_from_q_values(self, game_state):
        """
        Returns the best action to take in the current state.
        This method is used to select the action with the highest Q-value.
        """
        legal_actions = game_state.get_legal_actions(self.index)
        if len(legal_actions) == 0:
            return None

        actionVals = {}
        bestQValue = float('-inf')

        for action in legal_actions:
            target_q_value = self.get_q_value(game_state, action)
            actionVals[action] = target_q_value
            if target_q_value > bestQValue:
                bestQValue = target_q_value
        bestActions = [k for k, v in actionVals.items() if v == bestQValue]
 
        return random.choice(bestActions)



class DefensiveReflexAgent(ReflexCaptureAgent):
    def register_initial_state(self, game_state):
        """
        Register the initial state of the agent.
        This method is called at the beginning of each game to set up the necessary variables.
        """
        # Call the parent class's register_initial_state method
        super().register_initial_state(game_state)

        # Calculate the patrol points for the agent
        self.patrol_points = self.patrol_action(game_state)
        self.current_patrol_point = 0  # Index of the current patrol point

    def patrol_action(self, game_state):
        """
        Calculate the patrol points for the agent.
        The patrol points are chosen based on the location of the border that divides the map.
        """
        # Calculate the x-coordinate for the patrol area
        border_x = (game_state.get_walls().width // 2) - 1
        if not self.red:
            border_x += 1  # Adjust for blue team

        # Adjust x-coordinate to stay within safe distance from the border
        patrol_x = border_x - 1 if self.red else border_x + 1

        # Create patrol points focusing on chokepoints
        return self.evaluation_blockers(game_state, patrol_x)

    def evaluation_blockers(self, game_state, patrol_x):
        """
        Identify the chokepoints along the border of the map.
        These chokepoints are used as the patrol points for the agent.
        """
        # Initialize a list to store the identified chokepoints
        points = []

        # Get the height and width of the game map
        wall_matrix = game_state.get_walls()
        height = wall_matrix.height
        width = wall_matrix.width

        # Identify tiles that have gaps in the walls along the border
        if self.red:
            # If the agent is on the red team, search for gaps on the left side of the map
            for y in range(1, height - 1):
                if not wall_matrix[patrol_x][y] and not wall_matrix[patrol_x + 1][y]:
                    points.append((patrol_x, y))
        else:
            # If the agent is on the blue team, search for gaps on the right side of the map
            for y in range(1, height - 1):
                if not wall_matrix[patrol_x][y] and not wall_matrix[patrol_x - 1][y]:
                    points.append((patrol_x, y))

        return points

    def choose_action(self, game_state):
        """
        Choose the best action for the agent based on the current game state.
        This method is called during the game to determine the agent's next move.
        """
        # Get the legal actions the agent can take
        actions = game_state.get_legal_actions(self.index)

        # Get information about enemy agents
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

        # Identify visible invaders
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]

        # Get the agent's state
        my_state = game_state.get_agent_state(self.index)

        # Check if the agent is scared
        scared = my_state.scared_timer > 5

        if scared and invaders:
            # Avoid invaders when scared
            return self.defend_invader(game_state, actions)
        elif len(invaders) == 0:
            # Patrol based on belief distribution when there are no visible invaders
            return self.belief_distribution_action(game_state, actions)
        else:
            # Default behavior
            return super().choose_action(game_state)

    def defend_invader(self, game_state, actions):
        """
        Choose an action to defend against visible invaders.
        The agent will try to maintain a safe distance from the closest invader.
        """
        # Get the agent's current position and the positions of all visible invaders
        my_pos = game_state.get_agent_position(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [(a, a.get_position()) for a in enemies if a.is_pacman and a.get_position() is not None]

        # Safe buffer distance
        safe_distance = 5

        # Calculate distance to closest invader
        closest_invader_distance = float('inf')
        closest_invader_pos = None
        for invader, pos in invaders:
            distance = self.get_maze_distance(my_pos, pos)
            if distance < closest_invader_distance:
                closest_invader_distance = distance
                closest_invader_pos = pos

        # If no invader is found, return a random action
        if closest_invader_pos is None:
            return random.choice(actions)

        # Choose action that maintains the safe buffer distance
        best_action = None
        best_distance_diff = float('inf')
        for action in actions:
            successor = self.get_successor(game_state, action)
            next_pos = successor.get_agent_state(self.index).get_position()
            distance = self.get_maze_distance(next_pos, closest_invader_pos)

            # Calculate the difference from the safe distance
            distance_diff = abs(distance - safe_distance)

            if distance_diff < best_distance_diff:
                best_distance_diff = distance_diff
                best_action = action

        return best_action

    def belief_distribution_action(self, game_state, actions):
        """
        Choose an action based on the agent's belief distribution about the invaders' positions.
        This is used when there are no visible invaders, and the agent needs to patrol the map.
        """
        # Get the agent's current position
        myPos = game_state.get_agent_position(self.index)

        # Initialize variables for tracking the best action and its distance
        best_action = None
        min_dist = float('inf')

        # Determine the x-coordinate of the border that divides the map
        border_x = (game_state.get_walls().width // 2) - 1 if self.red else (game_state.get_walls().width // 2)

        # Identify the most probable invader location based on belief distribution
        most_probable_invader_loc = None
        highest_prob = 0.0
        for enemy in self.get_opponents(game_state):
            for pos, prob in self.obs[enemy].items():
                if prob > highest_prob and not game_state.has_wall(*pos):
                    # Ensure the position is on your side of the map
                    if (self.red and pos[0] <= border_x) or (not self.red and pos[0] >= border_x):
                        highest_prob = prob
                        most_probable_invader_loc = pos

        # If a probable invader location is identified on the agent's side, move towards it
        if most_probable_invader_loc:
            for action in actions:
                successor = self.get_successor(game_state, action)
                nextPos = successor.get_agent_state(self.index).get_position()
                # Ensure the agent doesn't cross into the opposing side
                if (self.red and nextPos[0] <= border_x) or (not self.red and nextPos[0] >= border_x):
                    dist = self.get_maze_distance(nextPos, most_probable_invader_loc)
                    if dist < min_dist:
                        best_action = action
                        min_dist = dist
        else:
            # Default to standard patrol behavior if no probable location is identified
            return self.traverse_action(game_state, actions)

        return best_action if best_action is not None else random.choice(actions)

    def traverse_action(self, game_state, actions):
        """
        Choose an action to traverse the patrol points.
        This is the default behavior when there are no visible invaders and no probable invader locations.
        """
        # Get the agent's current position
        myPos = game_state.get_agent_position(self.index)

        # Get the current patrol point
        patrol_point = self.patrol_points[self.current_patrol_point]

        # Check if reached the current patrol point
        if myPos == patrol_point:
            # Update to the next patrol point in the list, looping back if necessary
            self.current_patrol_point = (self.current_patrol_point + 1) % len(self.patrol_points)
            patrol_point = self.patrol_points[self.current_patrol_point]

        # Choose an action to move towards the patrol point
        best_action = None
        min_dist = float('inf')
        for action in actions:
            successor = self.get_successor(game_state, action)
            nextPos = successor.get_agent_state(self.index).get_position()
            dist = self.get_maze_distance(nextPos, patrol_point)
            if dist < min_dist:
                best_action = action
                min_dist = dist

        # Return the chosen action for patrolling
        return best_action if best_action is not None else random.choice(actions)

    def get_features(self, game_state, action):
        """
        Compute and return a set of features that describe the game state after taking a given action.
        These features are used by the agent to evaluate the quality of the action.
        """
        # Initialize an empty Counter to store the features
        features = util.Counter()

        # Get the successor state after taking the specified action
        successor = self.get_successor(game_state, action)

        # Get the agent's state in the successor state
        myState = successor.get_agent_state(self.index)
        myPos = myState.get_position()

        # Compute whether the agent is on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.is_pacman:
            features['onDefense'] = 0

        # Compute the distance to visible invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]

        enemies_idx = [i for i in self.get_opponents(successor) if successor.get_agent_state(i).is_pacman]

        # Calculate Bayesian probability to see an enemy in further positions
        if len(enemies_idx) > 0:
            if len(invaders) > 0:
                dists = [self.get_maze_distance(myPos, a.get_position()) for a in invaders]
                features['invaderDistance'] = min(dists)
                features['numInvaders'] = len(invaders)
            else:
                dists = []
                for e_idx in enemies_idx:
                    self.observe_action(e_idx, game_state)
                    self.elapse_time(e_idx, game_state)
                    belief_dist_e = self.obs[e_idx]
                    max_position, max_prob = max(belief_dist_e.items(), key=lambda item: item[1])
                    dists.append(self.get_maze_distance(myPos, max_position))
                features['invaderDistance'] = min(dists)
                features['numInvaders'] = len(enemies_idx)

        # Check if the action is STOP and set the 'stop' feature
        if action == Directions.STOP:
            features['stop'] = 1

        # Check if the action is a reverse action and set the 'reverse' feature
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features
    def getWeights(self, game_state, action):
        """
        Get the weights for the features used in the Q-learning algorithm.
        These weights determine the importance of each feature in the agent's decision-making process.
        """
        return {
            'numInvaders': -1000,  # Weight for the number of invaders (penalize more invaders)
            'onDefense': 100,  # Weight for being on defense (favor being on defense)
            'invaderDistance': -10,  # Weight for the distance to invaders (penalize longer distances)
            'stop': -100,  # Weight for choosing the STOP action (strongly penalize STOP)
            'reverse': -2  # Weight for choosing reverse actions (penalize reverse actions)
        }


"""Search Algos: MiniMax, AlphaBeta
"""
import numpy as np
import utils
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
import copy

# TODO: you can import more modules, if needed

class SearchAlgos:
    def __init__(self, utility, succ, perform_move, goal=None, real_state=None):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move
        self.goal = goal

    def search(self, state, depth, maximizing_player):
        pass

class MiniMax(SearchAlgos):

    def heuristic(self, state):

        # min_dist_to_fruit = self.get_min_dist_to_fruit(state.board)
        # if len(self.get_min_dist_to_fruit(state.board)) == 0:
        #     return 0
        #
        # min_dist_fruit_location = min(min_dist_to_fruit,
        #                               key=lambda fruit_loc: fruit_loc[1] + fruit_loc[2])
        #
        # return (min_dist_fruit_location[0]) / (min_dist_fruit_location[1] + min_dist_fruit_location[2])
        # res = state.scores[0] + self.moves_score(state.board, state.pos)

        # w = 1
        # moves_score = self.moves_score(state.board, state.pos)
        # fruits_score = [1 / self.get_md(state.pos, fruit[1]) for fruit in self.reachable_fruits_positions(state.board)]
        # reachable_fruits_positions = self.reachable_fruits_positions(state)
        # fruits_score = [(fruit[0] / self.get_md(state.pos, fruit[1])) * (0.1 * self.get_md(self.get_rival_pos(state.board), fruit[1])) for fruit in reachable_fruits_positions]

        # if len(reachable_fruits_positions) == 0:
        #     return moves_score
        # else:
        #    return state.score[0]*10 + w * max(fruits_score, default=0) + (1 - w) * moves_score

        # print("md: ", [self.get_md(state.pos, fruit[1]) for fruit in self.reachable_fruits_positions(state.board)])
        # print("fruits pos: ", self.reachable_fruits_positions(state.board))
        # print("fruits_score: ", fruits_score)
        # res = state.scores[0] + w * max(fruits_score, default=0) + (1 - w) * moves_score
        # print("res: ", res)
        # print("moves counter: ", state.moves_counter)
        # print("in heuristic; my score: ", state.scores[0], " rivals score: ", state.scores[1])

        # reachable_fruits_positions = self.reachable_fruits_positions(state)
        # if len(reachable_fruits_positions) == 0:  # if no fruits - everything is deterministic
        #     print("heuristic no fruits", state.scores[0] - state.scores[1])
        #     return state.scores[0] - state.scores[1]  # my score - rivals score
        # else:
        #     w = 1
        #     relative_score = (state.scores[0] - state.scores[1])
        #     fruits_score = [(fruit[0] / self.get_md(state.player_pos, fruit[1])) * (0.1 * self.get_md(state.rival_pos, fruit[1])) for fruit in reachable_fruits_positions]
        #     print("heuristic", w * max(fruits_score, default=0))
        #     return w * max(fruits_score, default=0) + (1-w) * relative_score

        # w1 =
        # w2 =
        # w3 =


        #
        # current_scores = state.scores
        #
        # # fruits
        # def get_fruit_score(fruit):
        #     return fruit[0]
        #
        # w1, w2, w3 = 1, 1, 1
        #
        # reachable_fruits_positions = self.reachable_fruits_positions(state)
        # max_fruit_by_value = reachable_fruits_positions.sort(key=get_fruit_score)[0]
        #
        # fruit_indicator = 0
        # if len(reachable_fruits_positions) > 0:
        #     fruit_indicator = 1 if current_scores[0] + max_fruit_by_value - state.pentaly_score > current_scores[1] else 0
        #
        #
        # # free tiles
        # num_of_free_tiles = self.number_of_future_moves(state.player_pos, state.board)
        #
        # # closeness to rival
        # #closeness_to_rival =
        #
        # return w1 * num_of_free_tiles + w2 * max_fruit_by_value * fruit_indicator  # + w3 * closeness_to_rival
        return self.number_of_future_moves(state.player_pos, state.board)

    # returns: (best score, best direction)
    def search(self, state, depth, maximizing_player):
        """Start the MiniMax algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        if self.goal(state):
            return self.utility(state), None
        if depth == 0:
            return self.heuristic(state), None

        if maximizing_player:
            current_max_score, current_max_direction = float('-inf'), None
            children = self.succ(state, maximizing_player=True)
            for child in children:
                child_score, child_direction = self.search(child, depth - 1, maximizing_player=False)
                if child_score > current_max_score:  # choose max child
                    current_max_score, current_max_direction = child_score, state.get_player_directions(child)
                if current_max_score == float('inf'):
                    break
            return current_max_score, current_max_direction

        else:  # maximizing_player == False
            current_min_score, current_min_direction = float('inf'), None
            children = self.succ(state, maximizing_player=False)
            for child in children:
                child_score, child_direction = self.search(child, depth - 1, maximizing_player=True)
                if child_score < current_min_score:  # choose min child
                    current_min_score, current_min_direction = child_score, None
                if current_min_score == float('-inf'):
                    break
            return current_min_score, current_min_direction

    # for heuristics
    # returns a list of:  (value, fruit pos) only if they are reachable!
    def reachable_fruits_positions(self, state):
        fruits_location = []
        for r in range(len(state.board)):
            for c in range(len(state.board[0])):
                if state.board[r][c] > 2 and self.calc_indicator(state, (r, c)):  # only if they are reachable!
                    fruits_location.append((state.board[r][c], (r, c)))  # (fruit value, row index, col index)

        return fruits_location

    def get_md(self, from_pos, to_pos):
        md = abs(to_pos[0] - from_pos[0]) + abs(to_pos[1] - from_pos[1])
        assert md != 0
        return abs(to_pos[0] - from_pos[0]) + abs(to_pos[1] - from_pos[1])

    # we have rival's pos in state.rival_pos
    # def get_rival_pos(self, board):
    #     pos = np.where(board == 2)
    #     return tuple(ax[0] for ax in pos)

    # returns 1 if we can get to the fruit before it disappears, 0 otherwise
    def calc_indicator(self, state, fruit_pos):
        number_of_moves_to_fruit_pos = self.get_md(state.player_pos, fruit_pos)
        min_rectangle_side = min(len(state.board), len(state.board[0]))
        number_of_moves_until_fruit_disappears = min_rectangle_side - state.moves_counter
        # res = number_of_moves_until_fruit_disappears >= number_of_moves_to_fruit_pos
        # print("fruit in pos:", fruit_pos, "will remain until I get there? :", res)
        return number_of_moves_until_fruit_disappears >= number_of_moves_to_fruit_pos

    # gets a COPY of the board
    # return number of free tiles ahead
    def number_of_future_moves(self, pos, board):
        future_moves_counter = 0
        copy_board = board
        all_next_positions = [utils.tup_add(pos, direction) for direction in utils.get_directions()]
        possible_next_positions = [next_position for next_position in all_next_positions if
                                   self.is_valid_pos(next_position, board)]  # save only valid positions

        board[pos] = -1  # mark current pos as grey

        if len(possible_next_positions) == 0:
            return future_moves_counter

        # for every possible next pos counter += 1 and call recursively
        for next_pos in possible_next_positions:
            future_moves_counter += 1  # next pos is a valid future move
            future_moves_counter += self.number_of_future_moves(next_pos, board)  # get all number of future moves from next pos

        return future_moves_counter

    # gets a position to check, returns True if we can move to this position, False otherwise
    def is_valid_pos(self, checking_pos, board):
        on_board = ((0 <= checking_pos[0] < len(board)) and (0 <= checking_pos[1] < len(board[0])))
        if not on_board:
            return False
        is_valid_cell = board[checking_pos] not in [-1, 1, 2]
        return is_valid_cell

class AlphaBeta(SearchAlgos):
    def heuristic(self, state):
        pass  # TODO same as minimax or different?

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        # TODO: erase the following line and implement this function.
        # raise NotImplementedError

        if self.goal(state):
            return self.utility((state, maximizing_player))
        if depth == 0:
            return (self.heuristic(state), None)

        children = self.succ(state)

        if maximizing_player:
            current_max = (float('-inf'), None)
            for child in children:
                child_result = self.search(child, depth - 1, maximizing_player, alpha, beta)
                if child_result[0] > current_max[0]:
                    current_max = (child_result[0], state.get_directions(child))
                alpha = max(current_max[0], alpha)
                if current_max[0] >= beta:
                    return (float('inf'), None)
            return current_max

        else:  # agent_to_move != maximizing_player
            current_min = (float('inf'), None)
            for child in children:
                child_result = self.search(child, depth - 1, not maximizing_player, alpha, beta)
                if child_result[0] < current_min[0]:
                    current_min = (child_result[0], None)
                beta = min(current_min[0], beta)
                if current_min[0] <= alpha:
                    return (float('-inf'), None)
            return current_min

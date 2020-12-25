"""Search Algos: MiniMax, AlphaBeta
"""
import numpy as np
import utils
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
from operator import  itemgetter
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

        # does not always win in situation it should have won
        # consider changing weights
        # consider normalizing the result

        w_fruit = 1
        w_rival = 1
        w_tiles = 1

        scores = state.scores

        max_fruit_value, fruit_indicator = 0, 0
        reachable_fruits_positions = self.reachable_fruits_positions(state)
        if len(reachable_fruits_positions) > 0:
            max_fruit_value = max(reachable_fruits_positions, key=itemgetter(0))[0]
            fruit_indicator = scores[0] + max_fruit_value - state.penalty_score > scores[1]

        score_of_free_tiles = self.number_of_future_moves(state.player_pos, copy.deepcopy(state.board)) - self.number_of_future_moves(state.rival_pos, copy.deepcopy(state.board))

        closeness_to_rival = self.get_md(state.player_pos, state.rival_pos)
        #consider adding indicator if reachable

        return w_fruit * fruit_indicator * max_fruit_value + w_rival * 1/closeness_to_rival + w_tiles * score_of_free_tiles



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

    # return number of free tiles ahead
    def number_of_future_moves(self, pos, board):
        future_moves_counter = 0
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

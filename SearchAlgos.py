"""Search Algos: MiniMax, AlphaBeta
"""
import collections
from operator import itemgetter

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

    # like original heuristic, BUT:
    # reachable_fruits is calculated by manhattan distance instead of shortest path (BFS)
    # score_of_tiles doesnt care about rival's score of tiles

    def heuristic(self, state):
        max_fruit_value, fruit_helpful = 0, False
        relative_score = state.scores[0] - state.scores[1]

        # only if there are fruits on board (we get fruits_pos from outside)
        if len(state.fruits_pos) > 0:
            reachable_fruits = self.reachable_fruits_positions(state)
            if len(reachable_fruits) > 0:
                # check if fruit will lead to a win
                max_fruit = max(reachable_fruits, key=itemgetter(0))
                max_fruit_value, max_fruit_pos = max_fruit[0], max_fruit[1]
                fruit_near_max_fruit = [fruit[0] for fruit in reachable_fruits if fruit[1] != max_fruit_pos and self.get_md(max_fruit_pos, fruit[1]) == 1]  # get best value near fruit near max fruit, but not max fruit value itself
                max_fruit_value += max(fruit_near_max_fruit, default=0)  # add fruit_near_max_fruit value
                fruit_helpful = state.scores[0] + max_fruit_value - state.penalty_score > state.scores[1]  # check if it fruit score will help

        # if fruit will make us win
        if fruit_helpful:
            # print("all fruits:", state.fruits_pos)
            # print("fruit value", max_fruit_value, "at:", max_fruit_pos)
            # print("max_fruit_value + relative_score", max_fruit_value, "+", relative_score)
            return max_fruit_value + relative_score

        else:  # try to avoid penalty score
            score_of_tiles = self.number_of_future_moves(state.player_pos,
                                                         copy.deepcopy(state.board)) - self.number_of_future_moves(
                state.rival_pos, copy.deepcopy(state.board))
            return score_of_tiles

    # returns: (best score, best direction)

    # return state.scores[0] - state.scores[1]

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
                if state.board[r][c] > 2 and self.is_reachable(state, (r, c)):  # only if they are reachable!
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
    def is_reachable(self, state, fruit_pos):
        number_of_moves_to_fruit_pos = self.calc_shortest_path(state.board, state.player_pos, fruit_pos)
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
            future_moves_counter += self.number_of_future_moves(next_pos,
                                                                board)  # get all number of future moves from next pos

        return future_moves_counter

    # gets a position to check, returns True if we can move to this position, False otherwise
    def is_valid_pos(self, checking_pos, board):
        on_board = ((0 <= checking_pos[0] < len(board)) and (0 <= checking_pos[1] < len(board[0])))
        if not on_board:
            return False
        is_valid_cell = board[checking_pos] not in [-1, 1, 2]
        return is_valid_cell

    # gets the board, fruit pos and player_pos returns the len of shortest to the fruit from players_pos on the board
    # practically a BFS
    def calc_shortest_path(self, board, start, end):
        queue = [[start]]
        seen = [start]
        while queue:
            path = queue.pop(0)
            if path[-1] == end:
                return len(path) - 1  # -1 because without start node

            for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                next_pos = utils.tup_add(copy.deepcopy(path[-1]), d)
                if self.is_valid_pos(next_pos, board) and next_pos not in seen:
                    queue.append(path + [next_pos])
                    seen.append(next_pos)

        return float('inf')
        # return len(board) + len(board[0])  # if not reachable return the max moves in board

    def number_of_white_tiles_in_board(self, board):
        counter = 0
        for row in board:
            for cell in row:
                if cell not in [-1, 1, 2]:
                    counter += 1
        return counter


class AlphaBeta(SearchAlgos):

    # same heuristics as minimax
    def heuristic(self, state):
        max_fruit_value, fruit_helpful = 0, False
        relative_score = state.scores[0] - state.scores[1]

        # only if there are fruits on board (we get fruits_pos from outside)
        if len(state.fruits_pos) > 0:
            reachable_fruits = self.reachable_fruits_positions(state)
            if len(reachable_fruits) > 0:
                # check if fruit will lead to a win
                max_fruit = max(reachable_fruits, key=itemgetter(0))
                max_fruit_value, max_fruit_pos = max_fruit[0], max_fruit[1]
                fruit_near_max_fruit = [fruit[0] for fruit in reachable_fruits if fruit[1] != max_fruit_pos and self.get_md(max_fruit_pos, fruit[1]) == 1]  # get best value near fruit near max fruit, but not max fruit value itself
                max_fruit_value += max(fruit_near_max_fruit, default=0)  # add fruit_near_max_fruit value
                fruit_helpful = state.scores[0] + max_fruit_value - state.penalty_score > state.scores[1]  # check if it fruit score will help

        # if fruit will make us win
        if fruit_helpful:
            # print("all fruits:", state.fruits_pos)
            # print("fruit value", max_fruit_value, "at:", max_fruit_pos)
            # print("max_fruit_value + relative_score", max_fruit_value, "+", relative_score)
            return max_fruit_value + relative_score

        else:  # try to avoid penalty score
            score_of_tiles = self.number_of_future_moves(state.player_pos,
                                                         copy.deepcopy(state.board)) - self.number_of_future_moves(
                state.rival_pos, copy.deepcopy(state.board))
            return score_of_tiles

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
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
                child_score, child_direction = self.search(child, depth - 1, maximizing_player=False, alpha=alpha,
                                                           beta=beta)
                if child_score > current_max_score:  # choose max child
                    current_max_score, current_max_direction = child_score, state.get_player_directions(child)
                alpha = max(alpha, current_max_score)
                if current_max_score >= beta:  # if the current max score is greater or equal to beta, min node won't choose, so cut it
                    return float('inf'), current_max_direction
                if current_max_score == float('inf'):
                    break
            return current_max_score, current_max_direction

        else:  # maximizing_player == False
            current_min_score, current_min_direction = float('inf'), None
            children = self.succ(state, maximizing_player=False)
            for child in children:
                child_score, child_direction = self.search(child, depth - 1, maximizing_player=True, alpha=alpha,
                                                           beta=beta)
                if child_score < current_min_score:  # choose min child
                    current_min_score, current_min_direction = child_score, None
                beta = min(beta, current_min_score)
                if current_min_score <= alpha:  # if the current min score is less or equal to alpha, max node won't choose it, so cut it
                    return float('-inf'), current_min_direction
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
            future_moves_counter += self.number_of_future_moves(next_pos,
                                                                board)  # get all number of future moves from next pos

        return future_moves_counter

    # gets a position to check, returns True if we can move to this position, False otherwise
    def is_valid_pos(self, checking_pos, board):
        on_board = ((0 <= checking_pos[0] < len(board)) and (0 <= checking_pos[1] < len(board[0])))
        if not on_board:
            return False
        is_valid_cell = board[checking_pos] not in [-1, 1, 2]
        return is_valid_cell

    def reachable_fruits_positions_simple(self, state):
        fruits_location = []
        for r in range(len(state.board)):
            for c in range(len(state.board[0])):
                if state.board[r][c] > 2 and self.is_reachable(state, (r, c)):  # only if they are reachable!
                    fruits_location.append((state.board[r][c], (r, c)))  # (fruit value, row index, col index)

        return fruits_location

    def is_reachable_simple(self, state, fruit_pos):
        number_of_moves_to_fruit_pos = self.get_md(state.player_pos, fruit_pos)
        min_rectangle_side = min(len(state.board), len(state.board[0]))
        number_of_moves_until_fruit_disappears = min_rectangle_side - state.moves_counter
        # res = number_of_moves_until_fruit_disappears >= number_of_moves_to_fruit_pos
        # print("fruit in pos:", fruit_pos, "will remain until I get there? :", res)
        return number_of_moves_until_fruit_disappears >= number_of_moves_to_fruit_pos


class AlphaBeta_simple_heuristic(SearchAlgos):
    # same heuristics as minimax
    def simple_heuristic(self, state):
        # print("player score:", state.scores[0], "rival score", state.scores[1])
        max_fruit_value, fruit_helpful = 0, False
        # only if there are fruits on board (we get fruits_pos from outside)
        # we check twice because first check with state.fruits_pos is really short, the other check takes more time
        if len(state.fruits_pos) > 0:
            reachable_fruits = self.reachable_fruits_positions_simple(state)
            if len(reachable_fruits) > 0:
                # print("reachable_fruits:", reachable_fruits)
                max_fruit_value = max(reachable_fruits, key=itemgetter(0))[0]
                fruit_helpful = state.scores[0] + max_fruit_value - state.penalty_score > state.scores[
                    1]  # check if it fruit score will help

        # if fruit will make us win
        if fruit_helpful:
            # print("heuristic fruit:", max_fruit_value)
            return max_fruit_value + state.scores[0]

        else:  # try to avoid penalty score
            score_of_tiles = self.number_of_future_moves(state.player_pos, copy.deepcopy(state.board))
            # print("heuristic score_of_tiles:", score_of_tiles)
            # print("player score:", state.scores[0], "rival score", state.scores[1])
            return score_of_tiles + state.scores[0]

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """

        if self.goal(state):
            return self.utility(state), None
        if depth == 0:
            return self.simple_heuristic(state), None

        if maximizing_player:
            current_max_score, current_max_direction = float('-inf'), None
            children = self.succ(state, maximizing_player=True)
            for child in children:
                child_score, child_direction = self.search(child, depth - 1, maximizing_player=False, alpha=alpha,
                                                           beta=beta)
                if child_score > current_max_score:  # choose max child
                    current_max_score, current_max_direction = child_score, state.get_player_directions(child)
                alpha = max(alpha, current_max_score)
                if current_max_score >= beta:  # if the current max score is greater or equal to beta, min node won't choose, so cut it
                    return float('inf'), current_max_direction
                if current_max_score == float('inf'):
                    break
            return current_max_score, current_max_direction

        else:  # maximizing_player == False
            current_min_score, current_min_direction = float('inf'), None
            children = self.succ(state, maximizing_player=False)
            for child in children:
                child_score, child_direction = self.search(child, depth - 1, maximizing_player=True, alpha=alpha,
                                                           beta=beta)
                if child_score < current_min_score:  # choose min child
                    current_min_score, current_min_direction = child_score, None
                beta = min(beta, current_min_score)
                if current_min_score <= alpha:  # if the current min score is less or equal to alpha, max node won't choose it, so cut it
                    return float('-inf'), current_min_direction
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
            future_moves_counter += self.number_of_future_moves(next_pos,
                                                                board)  # get all number of future moves from next pos

        return future_moves_counter

    # gets a position to check, returns True if we can move to this position, False otherwise
    def is_valid_pos(self, checking_pos, board):
        on_board = ((0 <= checking_pos[0] < len(board)) and (0 <= checking_pos[1] < len(board[0])))
        if not on_board:
            return False
        is_valid_cell = board[checking_pos] not in [-1, 1, 2]
        return is_valid_cell

    def reachable_fruits_positions_simple(self, state):
        fruits_location = []
        for r in range(len(state.board)):
            for c in range(len(state.board[0])):
                if state.board[r][c] > 2 and self.is_reachable_simple(state, (r, c)):  # only if they are reachable!
                    fruits_location.append((state.board[r][c], (r, c)))  # (fruit value, row index, col index)

        return fruits_location

    def is_reachable_simple(self, state, fruit_pos):
        number_of_moves_to_fruit_pos = self.get_md(state.player_pos, fruit_pos)
        min_rectangle_side = min(len(state.board), len(state.board[0]))
        number_of_moves_until_fruit_disappears = min_rectangle_side - state.moves_counter
        # res = number_of_moves_until_fruit_disappears >= number_of_moves_to_fruit_pos
        # print("fruit in pos:", fruit_pos, "will remain until I get there? :", res)
        return number_of_moves_until_fruit_disappears >= number_of_moves_to_fruit_pos

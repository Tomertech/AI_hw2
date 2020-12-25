"""
MiniMax Player
"""
from SearchAlgos import MiniMax
from players.AbstractPlayer import AbstractPlayer
import numpy as np  # TODO consider importing only in AbstractPlayer
import utils
import copy


# TODO: you can import more modules, if needed


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time,
                                penalty_score)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        # TODO: initialize more fields, if needed, and the Minimax algorithm from SearchAlgos.py
        self.board = None
        self.player_pos = None
        self.rival_pos = None
        self.minimax = MiniMax(self.utility, self.succ, self.perform_move, self.goal)
        self.scores = [0, 0]
        self.moves_counter = 0

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        # TODO: erase the following line and implement this function.
        # raise NotImplementedError
        self.board = board
        player_pos = np.where(board == 1)
        rival_pos = np.where(board == 2)
        self.player_pos = tuple(ax[0] for ax in player_pos)
        self.rival_pos = tuple(ax[0] for ax in rival_pos)

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        self.scores = players_score
        self.update_players_pos()  # get the current pos of players from board
        best_new_move_score, best_new_move_direction = float('-inf'), self.get_a_valid_move()  # just a valid move so it won't be None
        player_state = self.state(self.board, self.player_pos, self.rival_pos, self.scores, self.penalty_score,
                                  self.moves_counter)
        for depth in range(1, 8):
            score, move = self.minimax.search(copy.deepcopy(player_state), depth, maximizing_player=True)
            if score > best_new_move_score:  # we update it there is a better score OR if best_new_move_direction is None to get at least one valid move
                best_new_move_score, best_new_move_direction = score, move

        next_pos = self.player_pos[0] + best_new_move_direction[0], self.player_pos[1] + best_new_move_direction[1]

        # update Player's fields: scores, player pos, moves counter
        previous_position = self.player_pos
        self.player_pos = next_pos
        self.moves_counter += 1  # in order to keep track on moves number for heuristic (fruits disappear after x turns)

        # update Player's board: score
        self.board[previous_position] = -1
        self.board[next_pos] = 1
        return best_new_move_direction

    def set_rival_move(self, pos):
        """Updaet your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is exp+`1`ected
        """
        # TODO: erase the following line and implement this function.
        # raise NotImplementedError

        self.board[self.rival_pos] = -1
        self.rival_pos = pos
        self.board[self.rival_pos] = 2

    ########## helper functions in class ##########
    # TODO: add here helper functions in class, if needed
    class state:
        def __init__(self, board, player_pos, rival_pos, scores, penalty_score, moves_counter):
            self.board = board
            self.player_pos = player_pos  # player 1 position
            self.rival_pos = rival_pos  # player 2 position
            self.scores = scores  # [my score, rival's score]
            self.penalty_score = penalty_score
            self.moves_counter = moves_counter

        def get_player_directions_by_pos(self, pos):
            # print("get_player_directions:", state.player_pos[0] - self.player_pos[0], state.player_pos[1] - self.player_pos[1])
            return pos[0] - self.player_pos[0], pos[1] - self.player_pos[1]

        def get_player_directions(self, state):
            # print("get_player_directions:", state.player_pos[0] - self.player_pos[0], state.player_pos[1] - self.player_pos[1])
            return state.player_pos[0] - self.player_pos[0], state.player_pos[1] - self.player_pos[1]

        def update_players_pos(self):
            self.player_pos = tuple(ax[0] for ax in np.where(self.board == 1))
            self.rival_pos = tuple(ax[0] for ax in np.where(self.board == 2))

        ########## helper functions for MiniMax algorithm ##########

        def delete_unreachable_fruits(self):
            """Update your info on the current fruits on board (if needed).
            input:
                - fruits_on_board_dict: dict of {pos: value}
                                        where 'pos' is a tuple describing the fruit's position on board,
                                        'value' is the value of this fruit.
            No output is expected.
            """
            # TODO: erase the following line and implement this function. In case you choose not to use it, use 'pass' instead of the following line.
            # raise NotImplementedError
            for r in range(len(self.board)):
                for c in range(len(self.board[0])):
                    if self.board[r][c] > 2 and self.is_fruit_disappear(
                            (r, c)):  # if there is a fruit and it should disappear
                        self.board[r][c] = 0  # delete fruit

        def is_fruit_disappear(self, fruit_pos):
            number_of_moves_to_fruit_pos = self.get_md(self.player_pos, fruit_pos)
            min_rectangle_side = min(len(self.board), len(self.board[0]))
            number_of_moves_until_fruit_disappears = min_rectangle_side - self.moves_counter
            # returns True if fruit will disappear before we are able to get them
            return number_of_moves_until_fruit_disappears < number_of_moves_to_fruit_pos

        def get_md(self, from_pos, to_pos):
            md = abs(to_pos[0] - from_pos[0]) + abs(to_pos[1] - from_pos[1])
            assert md != 0
            return abs(to_pos[0] - from_pos[0]) + abs(to_pos[1] - from_pos[1])

    # TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm
    # we only care about the relative score - meaning if we win we don't care by how much
    # if we win return infinity, if we lose return -infinity
    def utility(self, state):
        # print("~~~~~~~~~ in utility ~~~~~~~~~")
        win_value, lose_value, tie_value = float('inf'), float('-inf'), 0
        my_score = state.scores[0] - state.penalty_score  # I'm here because it's a leaf (goal) - meaning I have no moves
        rival_score = (state.scores[1] - state.penalty_score) if self.is_stuck(state.rival_pos, state.board) else state.scores[1]

        # print("my_score", my_score, "rival_score", rival_score)
        if my_score > rival_score:
            return win_value
        elif my_score < rival_score:
            return lose_value
        else:  # it's a Tie
            return tie_value

    # develop next states
    # if maximizing player == True: create next states as player 1 played
    # if maximizing player == False: create next states as player 2 played
    def succ(self, state, maximizing_player):
        player = 1 if maximizing_player else 2  # set the right next player
        current_player_pos = state.player_pos if maximizing_player else state.rival_pos  # find current player position
        all_next_positions = [utils.tup_add(current_player_pos, direction) for direction in utils.get_directions()]
        possible_next_positions = [next_position for next_position in all_next_positions if
                                   self.is_valid_pos(next_position, state.board)]  # save only valid positions
        children = []

        for next_pos in possible_next_positions:
            # child = self.state(copy.deepcopy(state.board), copy.deepcopy(state.player_pos), copy.deepcopy(state.rival_pos), copy.deepcopy(state.scores), copy.deepcopy(state.penalty_score), copy.deepcopy(state.moves_counter) + 1)  # create a child like its parent state, then update it
            child = copy.deepcopy(state)
            child.board[current_player_pos] = -1  # mark visited in previous position
            child.scores[player - 1] += state.board[next_pos]  # add score from next position on board to relevant player
            child.board[next_pos] = player  # move player to next position
            if maximizing_player:
                child.moves_counter += 1
            child.update_players_pos()
            # child.delete_unreachable_fruits()
            children.append(child)

            #TODO delete it
            if not (np.count_nonzero(child.board == 1) == 1 and np.count_nonzero(child.board == 2) == 1):
                print("(np.count_nonzero(child.board == 1)", np.count_nonzero(child.board == 1))
                print("(np.count_nonzero(child.board == 2)", np.count_nonzero(child.board == 2))
            assert np.count_nonzero(child.board == 1) == 1 and np.count_nonzero(child.board == 2) == 1

        return children

    def perform_move(self):
        pass

    # gets a position to check, returns True if we can move to this position, False otherwise
    def is_valid_pos(self, checking_pos, board):
        on_board = ((0 <= checking_pos[0] < len(board)) and (0 <= checking_pos[1] < len(board[0])))
        if not on_board:
            return False
        is_valid_cell = board[checking_pos] not in [-1, 1, 2]
        return is_valid_cell

    # returns True if state is goal
    def goal(self, state):
        return self.is_stuck(state.player_pos, state.board)

    # return True if there is no place to go from given position (aka goal)
    def is_stuck(self, pos, board):
        all_next_positions = [utils.tup_add(pos, direction) for direction in utils.get_directions()]
        possible_next_positions = [next_pos for next_pos in all_next_positions if self.is_valid_pos(next_pos, board)]
        return len(possible_next_positions) == 0

    def update_players_pos(self):
        self.player_pos = tuple(ax[0] for ax in np.where(self.board == 1))
        self.rival_pos = tuple(ax[0] for ax in np.where(self.board == 2))

    # return any valid move
    def get_a_valid_move(self):
        all_next_positions = [utils.tup_add(self.player_pos, direction) for direction in utils.get_directions()]
        possible_next_positions = [next_position for next_position in all_next_positions if
                                   self.is_valid_pos(next_position, self.board)]
        pos = possible_next_positions[0]
        return pos[0] - self.player_pos[0], pos[1] - self.player_pos[1]

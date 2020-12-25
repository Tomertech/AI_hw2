"""
MiniMax Player with AlphaBeta pruning
"""
from SearchAlgos import AlphaBeta
from players.AbstractPlayer import AbstractPlayer
import numpy as np #TODO consider importing only in AbstractPlayer
import utils#TODO: you can import more modules, if needed


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time, penalty_score) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        #TODO: initialize more fields, if needed, and the AlphaBeta algorithm from SearchAlgos.py
        # From HW - we will not use the parameter game_time
        self.board = None
        self.pos = None
        self.time_left = None
        self.alphabeta = AlphaBeta(self.utility, self.succ, self.perform_move, self.goal)
        self.score = 0

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        #TODO: erase the following line and implement this function.
        # raise NotImplementedError
        self.board = board
        pos = np.where(board == 1)
        # convert pos to tuple of ints
        self.pos = tuple(ax[0] for ax in pos)

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        #TODO: erase the following line and implement this function.
        # raise NotImplementedError
        state = self.state(self.board, self.pos, time_limit, players_score, True, self.penalty_score)
        best_new_move = (float('-inf'), None)
        for depth in range(1, 5):
            res = self.alphabeta.search(state, depth, True)
            if res[0] > best_new_move[0]:
                best_new_move = res

        self.score += self.board[best_new_move[1]]
        self.board[self.pos] = -1
        self.board[best_new_move[1]] = 1

        return best_new_move[1]


    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        #TODO: erase the following line and implement this function.
        # raise NotImplementedError
        self.board[pos] = -1


    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        #TODO: erase the following line and implement this function. In case you choose not to use this function, 
        # use 'pass' instead of the following line.
        # raise NotImplementedError
        pass


    ########## helper functions in class ##########
    #TODO: add here helper functions in class, if needed
    class state():
        def __init__(self, board, pos, time_left, score, turn, penalty_score):
            self.board = board
            self.pos = pos
            self.time_left = time_left
            self.score = score  # [0, 0]
            self.turn = turn  # boolean if true then it's our player, and false if it's a rival player
            self.penalty_score = penalty_score

        # def get_directions(self, state):
        #     return (self.pos[0] - state.pos[0], self.pos[1] - state.pos[1])

    ########## helper functions for AlphaBeta algorithm ##########
    #TODO: add here the utility, succ, and perform_move functions used in AlphaBeta algorithm
    def utility(self, state):
        # if both players can't move then the player doesn't receive a penalty score
        if not self.can_move(np.where(state.board == 2), state.board):
            return state.score
        else:
            return state.score - state.penalty_score

    def succ(self, state):
        all_next_positions = [utils.tup_add(state.pos, direction) for direction in utils.get_directions()]
        possible_next_positions = [pos for pos in all_next_positions if self.can_move(pos, self.board)]
        children = []
        for pos in possible_next_positions:
            board = np.copy(state.board)
            child_score = state.score + board[pos[0]][pos[1]]
            board[state.pos[0]][state.pos[1]] = -1  # TODO check if self.map[prev_pos[1]][prev_pos[0]] = -1
            board[pos[0]][pos[1]] = 1  # TODO if we assume the player is always 1
            child = self.state(board, pos, state.time_left, child_score, not state.turn,
                               self.penalty_score)  # TODO should we update the time left?
            children.append(child)
        return children

    def perform_move(self):  # TODO find out what does it do
        pass

    def can_move(self, pos, board):
        on_board = (0 <= pos[0] < len(board) and 0 <= pos[1] < len(board[0]))
        if not on_board:
            return False

        value_in_pos = board[pos[0]][pos[1]]
        free_cell = (value_in_pos not in [-1, 1, 2])
        return free_cell

    def goal(self, state):
        # how to know if there is no more time left?
        all_next_positions = [utils.tup_add(state.pos, direction) for direction in utils.get_directions()]
        possible_next_positions = [pos for pos in all_next_positions if self.can_move(pos, state.board)]
        return len(possible_next_positions) == 0
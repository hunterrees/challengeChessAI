import sys
sys.path.append("/Users/Brennen/sunfish")
import train
import pickle
import theano
import theano.tensor as T
import math
import chess, chess.pgn
from parse_game import bb2array
import heapq
import time
import re
import string
import numpy
import sunfish
import pickle
import random
import traceback
import MySQLdb
import socket
import pdb

gameStates = {}

def get_model_from_pickle(fn):
    f = open(fn)
    Ws, bs = pickle.load(f)

    Ws_s, bs_s = train.get_parameters(Ws=Ws, bs=bs)
    x, p = train.get_model(Ws_s, bs_s)

    predict = theano.function(
        inputs=[x],
        outputs=p)

    return predict

strip_whitespace = re.compile(r"\s+")
translate_pieces = string.maketrans(".pnbrqkPNBRQK", "\x00" + "\x01\x02\x03\x04\x05\x06" + "\x08\x09\x0a\x0b\x0c\x0d")

def sf2array(pos, flip):
# Create a numpy array from a sunfish representation
    pos = strip_whitespace.sub('', pos.board) # should be 64 characters now
    pos = pos.translate(translate_pieces)
    m = numpy.fromstring(pos, dtype=numpy.int8)
    if flip:
        m = numpy.fliplr(m.reshape(8, 8)).reshape(64)
    return m

CHECKMATE_SCORE = 1e6

def negamax(pos, depth, alpha, beta, color, func):
    moves = []
    X = []
    pos_children = []
    for move in pos.gen_moves():
        pos_child = pos.move(move)
        moves.append(move)
        X.append(sf2array(pos_child, flip=(color==1)))
        pos_children.append(pos_child)

    if len(X) == 0:
        return Exception('eh?')

    # Use model to predict scores
    scores = func(X)

    for i, pos_child in enumerate(pos_children):
        if pos_child.board.find('K') == -1:
            scores[i] = CHECKMATE_SCORE

    child_nodes = sorted(zip(scores, moves), reverse=True)

    best_value = float('-inf')
    best_move = None

    for score, move in child_nodes:
        if depth == 1 or score == CHECKMATE_SCORE:
            value = score
        else:
            pos_child = pos.move(move)
            neg_value, _ = negamax(pos_child, depth-1, -beta, -alpha, -color, func)
            value = -neg_value

        if value > best_value:
            best_value = value
            best_move = move

        if value > alpha:
            alpha = value

        if alpha > beta:
            break

    return best_value, best_move

class machine_learning_ai():
    def __init__(self, func, player_1=False, player_2=True, maxd=3,):
        self._func = func
        self._pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
        self._player1 = player_1
        self._player2 = player_2
        self._maxd = maxd
    
    def isPlayer1(self):
        if self._player1 is True:
            return True
        else: return False

    def isPlayer2(self):
        if self._player2 is True:
            return True
        else: return False

    def setPlayerTurns(player_1):
        if player_1 is True:
            self._player1 = True
            self._player2 = False
        else:
            self._player1 = False
            self.player_2 = True

    def get_move(self, gn_current):
        pdb.set_trace()
        if gn_current is not None:
            if self._player1 is True:
                move = str(gn_current.move)
                formattedMove = (119 - sunfish.parse(move[0:2]), 119 - sunfish.parse(move[2:4]))
                self._pos = self._pos.move(formattedMove)
            elif self._player2 is True:
                move = str(gn_current.move)
                formatedMove = (sunfish.parse(move[0:2]), sunfish.parse(move[2:4]))
                self._pos = self._pos.move(formatedMove)

        alpha = float('-inf')
        beta = float('inf')
        depth = self._maxd

        best_value, best_move = negamax(self._pos, depth, alpha, beta, 1, self._func)
        new_move = sunfish.render(119 - best_move[0]) + sunfish.render(119 - best_move[1])
        self._pos = self._pos.move(best_move)
        return new_move

    def simulate_game(self, moveList):
        player1 = True
        for move in moveList:
            if player1:
                formatedMove = (sunfish.parse(move[0:2]), sunfish.parse(move[2:4]))
            else:
                formatedMove = (119 - sunfish.parse(move[0:2]), 119 - sunfish.parse(move[2:4]))
            player1 = not(player1)
            self._pos = self._pos.move(formatedMove)


class game():
    def __init__(self,machine,gn_current, game_id, moves=[]):
        self._machine = machine
        self._gn_current = gn_current
        self._game_id = game_id
        self._moves = moves

    def isTurn_Player1(self):
        if len(self._moves) % 2 == 1:
            return False
        else: return True

    def isTurn_Player2(self):
        if len(self._moves) % 2 == 0:
            return False
        else: return True

    def setMove(self, move):
        gn_new = chess.pgn.GameNode()
        gn_new.parent = self._gn_current
        gn_new.move = move
        self._gn_current = gn_new
        self._moves.append(move)

    def playMachine(self):
        move = self._machine.get_move(self._gn_current)
        self.setMove(move)
        return move

    def checkMove(self, move_str):
        bb = self._gn_current.board()
        move = chess.Move.from_uci(move_str)
        if move not in bb.legal_moves:
            return False
        return True

    def clientMoved(self, move):
        self.setMove(move)

    def addMove(move):
        self._moves.append(move)

def load_games():
    func = get_model_from_pickle('model.pickle')
    db = MySQLdb.connect("localhost","root","password","challengeChess")
    cursor = db.cursor()

    """Get player id for Machine Learning AI"""
    query = "SELECT id FROM player WHERE name = 'Pamala'"
    cursor.execute(query)
    results = cursor.fetchall()
    machine_id = results[0][0]
    """******************* END ***********************"""

    query = "SELECT * FROM game WHERE status = 'inProgress' AND (player_1 = %s || player_2 = %s)" % (machine_id,machine_id)
    cursor.execute(query)
    results = cursor.fetchall()
    for res in results:
        gn_current = chess.pgn.Game()
        all_moves = []
        game_id = res[0]
        player_1_id = res[1]
        player_2_id = res[2]
        q = "SELECT * FROM move WHERE game_id = %s" % (game_id) 
        cursor.execute(q)
        moves = cursor.fetchall()
        for mv in moves:
            from_pos = mv[1]
            to_pos = mv[2]
            piece = mv[4]
            total_move = str(from_pos) + str(to_pos)
            all_moves.append(total_move)
        for move in all_moves:
            m = chess.Move.from_uci(move)
            gn_new = chess.pgn.GameNode()
            gn_new.parent = gn_current
            gn_new.move = m
            gn_current = gn_new
        
        gameAI = None
        if player_1_id == machine_id:
            gameAI = machine_learning_ai(func, True, False)
        elif player_2_id == machine_id:
            gameAI = machine_learning_ai(func, False, True)

        gameAI.simulate_game(all_moves)
        g = game(gameAI, gn_current, game_id, all_moves)
        gameStates[game_id] = g    



def play():
    func = get_model_from_pickle('model.pickle')
    while True:
        inp = raw_input("Which player do you want to be?")
        if inp == "1":
            bot = machine_learning_ai(func, False, True)
            gn_current = chess.pgn.Game()
            gm = game(bot, gn_current, 1,)
            while True:
                clientMove = raw_input("Please enter a move: ")
                pdb.set_trace()
                while gm.checkMove(clientMove) is False:
                    clientMove = raw_input("Invalid Move!! Please enter a valid move, idiot: ")
                gm.clientMoved(clientMove)
                machineMove = gm.playMachine()
                print "Machine made the following move: ", machineMove





# def play():
#     # load_games()
#     # func = get_model_from_pickle('model.pickle')
#     s = socket.socket()
#     host = socket.gethostname() 
#     port = 12345                
#     s.connect((host, port))                
#     while True:
#         c, addr = s.accept()
#         stuff = s.accept     
#         print 'Got connection from', addr
#         print stuff
#         c.send('Thank you for connecting')
#         # c.close() 


    # bot = machine_learning_ai(func)
    # gn_current = chess.pgn.Game()

    # while True:
    #     pdb.set_trace()
    #     crdn = raw_input("Please enter a move: ")
    #     bb = gn_current.board()
    #     move = chess.Move.from_uci(crdn)
    #     if move in bb.legal_moves:
    #         gn_new = chess.pgn.GameNode()
    #         gn_new.parent = gn_current
    #         gn_new.move = move
    #         new_move = bot.get_move(crdn)
    #         print "The made the following move: ", new_move
    #         gn_new_2 = chess.pgn.GameNode()
    #         gn_new_2.parent = gn_current
    #         gn_new_2.move = new_move
    #     else:
    #         print "Invalid Move!"


if __name__ == '__main__':
    play()

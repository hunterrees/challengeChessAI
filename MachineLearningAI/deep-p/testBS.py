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
import pdb

gameState = {}

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
            # print 'ok will recurse', sunfish.render(move[0]) + sunfish.render(move[1])
            pos_child = pos.move(move)
            neg_value, _ = negamax(pos_child, depth-1, -beta, -alpha, -color, func)
            value = -neg_value

        # value += random.gauss(0, 0.001)

        # crdn = sunfish.render(move[0]) + sunfish.render(move[1])
        # print '\t' * (3 - depth), crdn, score, value

        if value > best_value:
            best_value = value
            best_move = move

        if value > alpha:
            alpha = value

        if alpha > beta:
            break

    return best_value, best_move

class machine_learning_ai():
    def __init__(self, func, maxd=3):
        self._func = func
        self._pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
        self._maxd = maxd

    def get_move(self, crdn):
        print "Parsed: ", sunfish.parse(crdn[0:2])
        move = (119 - sunfish.parse(crdn[0:2]), 119 - sunfish.parse(crdn[2:4]))
        self._pos = self._pos.move(move)
        alpha = float('-inf')
        beta = float('inf')
        depth = self._maxd
        best_value, best_move = negamax(self._pos, depth, alpha, beta, 1, self._func)
        new_move = sunfish.render(best_move[0]) + sunfish.render(best_move[1])
        print "new_move: ", new_move
        self._pos = self._pos.move(best_move)
        print "pos: ", self._pos

        return new_move





def play():
    func = get_model_from_pickle('model.pickle')
    bot = machine_learning_ai(func)

    while True:
        crdn = raw_input("Please enter a move: ")
        # move = (119 - sunfish.parse(crdn[0:2]), 119 - sunfish.parse(crdn[2:4]))
        new_move = bot.get_move(crdn)
        print "The made the following move: ", new_move


# while True:
#     side, times = game(func)
#     f = open('stats.txt', 'a')
#     f.write('%s %f %f\n' % (side, times['A'], times['B']))
#     f.close()


if __name__ == '__main__':
    play()

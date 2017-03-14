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
from pprint import pprint


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

model = get_model_from_pickle('model.pickle')


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
            #print 'ok will recurse', sunfish.render(move[0]) + sunfish.render(move[1])
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

def getMove(moveList):
    learnedModel = model
    currentBoardPosition = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
    #apply all moves in movelist to the currentBoardPosition
    player1 = True
    for move in moveList:
        if player1:
            formatedMove = (sunfish.parse(move[0:2]), sunfish.parse(move[2:4]))
        else:
            formatedMove = (119 - sunfish.parse(move[0:2]), 119 - sunfish.parse(move[2:4]))
        print 'formatedMove', formatedMove
        player1 = not(player1)
        currentBoardPosition = currentBoardPosition.move(formatedMove)

    # for depth in xrange(1, self._maxd+1):
    alpha = float('-inf')
    beta = float('inf')

    depth = 2
    t0 = time.time()

    best_value, best_move = negamax(currentBoardPosition, depth, alpha, beta, 1, learnedModel)
    moveStandardFormat = sunfish.render(best_move[0]) + sunfish.render(best_move[1])
    return moveStandardFormat


def run():
    moveList = ["d2d4", "a7a6", 'g1f3', 'g8f6']


    print getMove(moveList)


if __name__ == '__main__':
    run()
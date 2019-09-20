from itertools import combinations
import numpy as np


PAIR           = 1000000
TWO_PAIR       = 2000000
SET            = 3000000
STRAIGHT       = 4000000
FLUSH          = 5000000
FULL_HOUSE     = 6000000
QUADS          = 7000000
STRAIGHT_FLUSH = 8000000


"""
Expects hole_cards to be dimension 2x2.
Returns 1 if player 1 wins, 0 if draw, -1 if player 2 wins.
"""
def compute_winner(hole_cards, board):
    score_p1 = best_hand(hole_cards[0], board)
    score_p2 = best_hand(hole_cards[1], board)
    if score_p1 == score_p2:
        return 0
    elif score_p1 > score_p2:
        return 1
    else:
        return -1


"""
Expects hole_cards to be the hand for a single player, i.e. length 2.
Returns 1 if player 1 wins, 0 if draw, -1 if player 2 wins.
"""
def best_hand(hole_cards, board):
    score = 0
    for board_set in combinations(range(5), 3):
        board_set = np.array(board_set)
        hand  = [
            np.append(hole_cards[0], board[0][board_set]),
            np.append(hole_cards[1], board[1][board_set]),
        ]
        score = max(score, score_hand(hand))
    return score


"""
hand format: hand[0] is card values, hand[1] is suits
"""
def score_hand(hand):
    h = sorted(hand[0])
    if is_straight(h) and is_flush(hand):
        return straight_flush_val(h)
    elif is_quads(h):
        return quads_val(h)
    elif is_full_house(h):
        return full_house_val(h)
    elif is_flush(hand):
        return flush_val(h)
    elif is_straight(h):
        return straight_val(h)
    elif is_set(h):
        return set_val(h)
    elif is_two_pair(h):
        return two_pair_val(h)
    elif is_pair(h):
        return pair_val(h)
    else:
        return high_card_val(h)


def is_quads(hand):
    middle_equal = hand[1] == hand[2] and hand[2] == hand[3]
    return middle_equal and (hand[0] == hand[1] or hand[4] == hand[1])


def is_full_house(hand):
    a1 = hand[0] == hand[1] and\
         hand[1] == hand[2] and\
         hand[3] == hand[4]

    a2 = hand[0] == hand[1] and\
         hand[2] == hand[3] and\
         hand[3] == hand[4]

    return(a1 or a2);


def is_flush(hand):
    return len(set(hand[1])) == 1


def is_straight(hand):
    return hand == range(hand[0], hand[4]+1) or \
        hand[0] == 2 and hand[1] == 3 and hand[2] == 4 and hand[3] == 5 and hand[4] == 14


"""
Assumes that we already checked for quads and full house.
"""
def is_set(hand):
    a1 = hand[0] == hand[1] and hand[1] == hand[2]
    a2 = hand[1] == hand[2] and hand[2] == hand[3]
    a3 = hand[2] == hand[3] and hand[3] == hand[4]
    return a1 or a2 or a3


"""
Assumes that we already checked for sets, quads and full house.
"""
def is_two_pair(hand):
    a1 = hand[0] == hand[1] and hand[2] == hand[3]
    a2 = hand[0] == hand[1] and hand[3] == hand[4]
    a3 = hand[1] == hand[2] and hand[3] == hand[4]
    return a1 or a2 or a3


"""
Assumes that we already checked for two pair, sets, quads and full house.
"""
def is_pair(hand):
    return hand[0] == hand[1] or hand[1] == hand[2] or \
        hand[2] == hand[3] or hand[3] == hand[4]


def high_card_val(hand):
    val = 0
    for idx, card in enumerate(hand):
        val += card * (14**idx)
    return val


def pair_val(hand):
    for idx, card in enumerate(hand):
        if hand[idx] == hand[idx+1]:
            return PAIR + card * (14**3) + high_card_val(hand[0:idx] + hand[idx+2:len(hand)])


def two_pair_val(hand):
    if hand[0] == hand[1] and hand[2] == hand[3]:
        return 14**2 * hand[2] + 14 * hand[0] + hand[4] + TWO_PAIR
    elif hand[0] == hand[1] and hand[3] == hand[4]:
        return 14**2 * hand[3] + 14 * hand[0] + hand[2] + TWO_PAIR
    else:
        return 14**2 * hand[3] + 14 * hand[2] + hand[0] + TWO_PAIR


def set_val(hand):
    return SET + hand[2]


def flush_val(hand):
    return FLUSH + high_card_val(hand)


def straight_val(hand):
    if hand[4] == 14 and hand[0] == 2: # ace is bottom of straight
        return STRAIGHT + hand[3]
    return STRAIGHT + hand[4]


def full_house_val(hand):
    return FULL_HOUSE + hand[2]


def quads_val(hand):
    return QUADS + hand[2]


def straight_flush_val(hand):
    return STRAIGHT_FLUSH + hand[4]


def _test_hands():
    hole_cards = [[[14, 13], [0, 0]], [[12, 12], [1, 2]]]

    # p1 ace-high straight, p2 12-12-12-4-4 boat
    board = [np.array([12, 11, 10, 4, 4]), np.array([0, 2, 1, 0, 1])]
    assert compute_winner(hole_cards, board) == -1

    # p1 ace-high straight, p2 12-12-12-11-10 trips
    board = [np.array([12, 11, 10, 4, 5]), np.array([0, 2, 1, 0, 1])]
    assert compute_winner(hole_cards, board) == 1

    # p1 two pair, p2 pair
    board = [np.array([14, 13, 10, 4, 5]), np.array([0, 2, 1, 0, 1])]
    assert compute_winner(hole_cards, board) == 1

    # p1 two pair, p2 trips
    board = [np.array([14, 13, 12, 4, 5]), np.array([0, 2, 1, 0, 1])]
    assert compute_winner(hole_cards, board) == -1

    # p1 high card, p2 pair
    board = [np.array([11, 10, 7, 4, 5]), np.array([0, 2, 1, 0, 1])]
    assert compute_winner(hole_cards, board) == -1

    # p1 high card, p2 high card
    hole_cards = [[[14, 13], [0, 0]], [[12, 6], [1, 2]]]
    board = [np.array([11, 10, 7, 4, 5]), np.array([0, 2, 1, 0, 1])]
    assert compute_winner(hole_cards, board) == 1

    # p1 high card, p2 high card
    hole_cards = [[[14, 2], [0, 0]], [[12, 13], [1, 2]]]
    board = [np.array([11, 10, 7, 4, 5]), np.array([0, 2, 1, 0, 1])]
    assert compute_winner(hole_cards, board) == 1

    # p1 high card, p2 high card
    hole_cards = [[[14, 2], [0, 0]], [[14, 13], [1, 2]]]
    board = [np.array([11, 10, 7, 4, 5]), np.array([0, 2, 1, 0, 1])]
    assert compute_winner(hole_cards, board) == -1

    # p1 high card, p2 high card
    hole_cards = [[[14, 2], [0, 0]], [[14, 2], [1, 2]]]
    board = [np.array([11, 10, 7, 4, 5]), np.array([0, 2, 1, 0, 1])]
    assert compute_winner(hole_cards, board) == 0

    # p1 high card, p2 high card
    hole_cards = [[[14, 3], [0, 0]], [[14, 2], [1, 2]]]
    board = [np.array([11, 10, 7, 4, 5]), np.array([0, 2, 1, 0, 1])]
    assert compute_winner(hole_cards, board) == 1

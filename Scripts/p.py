import random

class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.rank} of {self.suit}"
    
    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank
    
    def __repr__(self):
        return self.__str__()
    
  
class Deck:
    def __init__(self):
        self.cards = []
        SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [Card(suit, rank) for suit in SUITS for rank in RANKS]


    def shuffle(self):
        random.shuffle(self.cards)
        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop() if self.cards else None

RANK_TO_VALUE = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13,'A':14}
RANK_TO_VALUE_ACELOW = dict(RANK_TO_VALUE)
RANK_TO_VALUE_ACELOW['A'] = 1



def FiveConsecutive(sorted_ranks):
    foundStraight = None
    for i in range(len(sorted_ranks) - 4):
        position = i
        for j in range(position, position + 4):
            if sorted_ranks[j] + 1 != sorted_ranks[j + 1]:
                j -= 1
                break
        if j == position + 3:
            foundStraight = i
    return foundStraight


def is_Straight(all_cards):
    if (len(all_cards) < 5):
        return False
    ranks = [card.rank for card in all_cards]
    rank_values1 = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 1}
    rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    sorted_ranks = sorted([rank_values[rank] for rank in ranks])
    sorted_ranks_ace_low = sorted([rank_values1[rank] for rank in ranks])
    if FiveConsecutive(sorted_ranks):
        return True
    elif FiveConsecutive(sorted_ranks_ace_low):
        return True
    else:
        return False

def is_StraightFlush(all_cards):
    if (len(all_cards) < 5):
        return False
    straight_flush = False
    RANK_ORDER = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    ACELOW_RANK_ORDER = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
    '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 1} 
    sorted_cards = sorted(all_cards, key=lambda card: RANK_ORDER[card.rank])
    suits_sorted = [card.suit for card in sorted_cards]
    sorted_cards_ace_low = sorted(all_cards, key=lambda card: ACELOW_RANK_ORDER[card.rank])
    suits_sorted_ace_low = [card.suit for card in sorted_cards_ace_low]
    val_cards = [RANK_ORDER[card.rank] for card in sorted_cards]
    val_cards_ace_low = [ACELOW_RANK_ORDER[card.rank] for card in sorted_cards_ace_low]

    if (FiveConsecutive(val_cards) is not None):
        start_index = FiveConsecutive(val_cards)
        flush_suit = suits_sorted[start_index]
        for i in range(start_index, start_index + 5):
            if suits_sorted[i] != flush_suit:
                i -= 1
                break
        if i == start_index + 4:
            straight_flush = True
    elif (FiveConsecutive(val_cards_ace_low) is not None):
        start_index = FiveConsecutive(val_cards_ace_low)
        flush_suit = suits_sorted_ace_low[start_index]
        for i in range(start_index, start_index + 5):
            if suits_sorted_ace_low[i] != flush_suit:
                i -= 1
                break
        if i == start_index + 4:
            straight_flush = True
    
    return straight_flush


def is_RoyalFlush(all_cards):
    if (len(all_cards) < 5):
        return False
    ranks = ["10", "J", "Q", "K", "A"]
    diamonds = [Card('Diamonds', rank) for rank in ranks]
    hearts = [Card('Hearts', rank) for rank in ranks]
    clubs = [Card('Clubs', rank) for rank in ranks]
    spades = [Card('Spades', rank) for rank in ranks]
    if (all(card in all_cards for card in diamonds) or
        all(card in all_cards for card in hearts) or
        all(card in all_cards for card in clubs) or
        all(card in all_cards for card in spades)):
        return True
    return False

def evaluate_hand(all_cards):
    ranks = [card.rank for card in all_cards]
    suits = [card.suit for card in all_cards]
    suit_count = {suit: suits.count(suit) for suit in set(suits)}
    rank_count = {rank: ranks.count(rank) for rank in set(ranks)}
    if is_RoyalFlush(all_cards):
        return "Royal Flush"
    elif is_StraightFlush(all_cards):
        return "Straight Flush"
    elif 4 in rank_count.values():
        return "Four of a Kind"
    elif 3 in rank_count.values() and 2 in rank_count.values():
        return "Full House"
    elif 5 in suit_count.values():
        return "Flush"
    elif is_Straight(all_cards):
        return "Straight"
    elif 3 in rank_count.values():
        return "Three of a Kind"
    elif list(rank_count.values()).count(2) >= 2:
        return "Two Pair"
    elif 2 in rank_count.values():
        return "One Pair"
    else:
        return "High Card" 


HAND_SCORE = {
    "Royal Flush": 1000,
    "Straight Flush": 900,
    "Four of a Kind": 800,
    "Full House": 700,
    "Flush": 600,
    "Straight": 500,
    "Three of a Kind": 400,
    "Two Pair": 300,
    "One Pair": 200,
    "High Card": 100
}

def hand_strength_score(cards):
    name = evaluate_hand(cards)
    base = HAND_SCORE[name]
    top = max(RANK_TO_VALUE[c.rank] for c in cards)
    return base + top/100.0


import numpy as np
from collections import Counter

def extract_features_learned(hole_cards, community_cards):
    visible = hole_cards + community_cards
    stage_map = {0:0, 3:1, 4:2, 5:3}
    stage = stage_map.get(len(community_cards), 0) / 3.0

    hole_vals = sorted([RANK_TO_VALUE[c.rank] for c in hole_cards])
    hole_high = hole_vals[-1] / 14.0
    hole_low = hole_vals[0] / 14.0
    pocket_pair = 1.0 if hole_cards[0].rank == hole_cards[1].rank else 0.0
    suited_hole = 1.0 if hole_cards[0].suit == hole_cards[1].suit else 0.0

    rank_counts = Counter([c.rank for c in visible]) if visible else Counter()
    suit_counts = Counter([c.suit for c in visible]) if visible else Counter()

    max_rank_count = max(rank_counts.values()) if rank_counts else 0
    num_pairs = sum(1 for v in rank_counts.values() if v >= 2)

    max_suit_count = max(suit_counts.values()) if suit_counts else 0

    uniq_vals = sorted({RANK_TO_VALUE[c.rank] for c in visible}) if visible else []
    if len(uniq_vals) >= 2:
        span = uniq_vals[-1] - uniq_vals[0]
    else:
        span = 0
    straight_potential = max(0.0, 1.0 - (span / 12.0))

    categories = list(HAND_SCORE.keys())
    current_cat = evaluate_hand(visible) if visible else "High Card"
    cat_onehot = [1.0 if current_cat == k else 0.0 for k in categories]

    feats = [
        stage,
        hole_high,
        hole_low,
        pocket_pair,
        suited_hole,
        max_rank_count / 4.0,
        num_pairs / 3.0,
        max_suit_count / 7.0,
        straight_potential
    ] + cat_onehot

    return np.array(feats, dtype=np.float32)

def monte_carlo_win_prob_learned(hole_cards, community_cards, num_opponents=4, iters=200):
    full_deck = Deck()
    used = {(c.suit, c.rank) for c in (hole_cards + community_cards)}
    full_deck.cards = [c for c in full_deck.cards if (c.suit, c.rank) not in used]

    wins = 0.0
    for _ in range(iters):
        random.shuffle(full_deck.cards)
        idx = 0
        opps = []
        for _ in range(num_opponents):
            opps.append([full_deck.cards[idx], full_deck.cards[idx+1]])
            idx += 2
        need = 5 - len(community_cards)
        board = community_cards + full_deck.cards[idx:idx+need]
        my_score = hand_strength_score(hole_cards + board)
        opp_scores = [hand_strength_score(h + board) for h in opps]
        best = max([my_score] + opp_scores)
        num_best = sum(1 for s in ([my_score] + opp_scores) if abs(s - best) < 1e-9)
        if abs(my_score - best) < 1e-9:
            wins += 1.0 / num_best
    return wins / iters

import tensorflow as tf
from tensorflow.keras import layers, models

def train_keras_model(X, y, epochs=15, batch_size=128, lr=1e-3):
    net = models.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["accuracy"]
    )
    net.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return net

def keras_predict_proba(net, x_np):
    x_np = np.asarray(x_np, dtype=np.float32)
    if x_np.ndim == 1:
        x_np = np.expand_dims(x_np, axis=0)  # [1, D]
    p = net.predict(x_np, verbose=0)[0, 0]
    return float(p)

class KerasWrapper:
    def __init__(self, net):
        self.net = net
    def predict_proba(self, x):
        return keras_predict_proba(self.net, x)

def build_training_set_learned(samples=1500, num_opponents=4, mc_iters=80):
    X_list = []
    y_list = []
    stages = [0, 3, 4, 5]
    for _ in range(samples):
        d = Deck()
        d.shuffle()
        hole = [d.deal(), d.deal()]
        comm_n = random.choice(stages)
        community = []
        for _k in range(comm_n):
            if d.cards:
                d.deal() 
            if d.cards:
                community.append(d.deal())
        winp = monte_carlo_win_prob_learned(hole, community, num_opponents=num_opponents, iters=mc_iters)
        label = winp
        X_list.append(extract_features_learned(hole, community))
        y_list.append(label)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    return X, y



def hole_strength_heuristic(hole_cards):
    v0 = RANK_TO_VALUE[hole_cards[0].rank]
    v1 = RANK_TO_VALUE[hole_cards[1].rank]
    high = max(v0, v1) / 14.0
    low = min(v0, v1) / 14.0
    pocket = 1.0 if hole_cards[0].rank == hole_cards[1].rank else 0.0
    suited = 1.0 if hole_cards[0].suit == hole_cards[1].suit else 0.0
    gap = abs(v0 - v1)
    connector_bonus = 0.12 if gap == 1 else (0.06 if gap == 2 else 0.0)
    broadway = 0.18 if v0 >= 11 and v1 >= 11 else 0.0
    base = (high + low) / 2.0
    score = base * 0.6 + pocket * 0.25 + suited * 0.08 + connector_bonus + broadway * 0.0
    return max(0.0, min(1.0, score))

def learned_ai_action(model, hole_cards, community_cards, player_name=None):
    x = extract_features_learned(hole_cards, community_cards)
    p = float(model.predict_proba(x))

    hs = hole_strength_heuristic(hole_cards)

    noise = (random.random() - 0.5) * 0.04 

    stage = len(community_cards) 

    def has_flush_draw(hole, board):
        suits = [c.suit for c in hole + board]
        for suit, cnt in Counter(suits).items():
            if cnt >= 4 and len(board) >= 3:
                return True
        return False

    def straight_draw_potential(hole, board):
        vals = sorted({RANK_TO_VALUE[c.rank] for c in hole + board})
        if len(vals) < 4:
            return 0.0
        for i in range(len(vals) - 3):
            window = vals[i:i+4]
            if window[0]+1 == window[1] and window[1]+1 == window[2] and window[2]+1 == window[3]:
                return 1.0 
        vals_ace_low = sorted({RANK_TO_VALUE_ACELOW[c.rank] for c in hole + board})
        for i in range(len(vals_ace_low) - 3):
            window = vals_ace_low[i:i+4]
            if window[0]+1 == window[1] and window[1]+1 == window[2] and window[2]+1 == window[3]:
                return 1.0
        return 0.25

    flush_draw = has_flush_draw(hole_cards, community_cards)
    straight_draw_score = straight_draw_potential(hole_cards, community_cards)

    #PRE-FLOP logic
    if stage == 0:
        if hs >= 0.45:
            if hs >= 0.8 and p > 0.08:
                return "raise", p
            return "call", p

        combined = 0.7 * hs + 0.3 * p + noise
        if player_name == "You":
            combined += 0.06
        if combined < 0.12:
            return "fold", p
        elif combined < 0.62:
            return ("call", p)
        else:
            return ("raise", p)

    # POST-FLOP logic
    combined = 0.5 * p + 0.45 * hs + noise

    if flush_draw:
        combined = max(combined, 0.48) + 0.08
    if straight_draw_score >= 1.0:
        combined = max(combined, 0.46) + 0.06
    elif straight_draw_score > 0:
        combined += 0.03
    if stage == 3 and (flush_draw or straight_draw_score >= 1.0):
        if combined < 0.38:
            return ("call", p)
    if stage >= 4 and (flush_draw or straight_draw_score >= 1.0):
        if combined < 0.40:
            return ("fold", p)

    if combined < 0.26:
        return "fold", p
    elif combined < 0.52:
        return "call", p
    else:
        return "raise", p

def get_human_action(model, hole_cards, community_cards, pot, to_call=10):
    print("Your hand:", hole_cards)
    print("Community:", community_cards)
    print("Hand evaluation:", evaluate_hand(hole_cards + community_cards))
    x = extract_features_learned(hole_cards, community_cards)
    p = float(model.predict_proba(x))
    hs = hole_strength_heuristic(hole_cards)
    print(f"Model P(win) = {p:.3f}, hole_strength = {hs:.3f}")
    print(f"Pot: {pot}. To call: {to_call}. Raise amount: 30 (fixed).")
    print("Choose action: [f]old, [c]all, [r]aise  (you can type full word)")

    # input loop until valid
    while True:
        choice = input("> ").strip().lower()
        if choice in ('f','fold'):
            return "fold", p
        if choice in ('c','call'):
            return "call", p
        if choice in ('r','raise'):
            return "raise", p
        print("Invalid choice — type 'f' (fold), 'c' (call), or 'r' (raise).")

def play_round(model, num_bots=4):
    d = Deck()
    d.shuffle()
    names = ["You"] + [f"Bot{i+1}" for i in range(num_bots)]
    hands = {n: [d.deal(), d.deal()] for n in names}
    community = []
    active = {n: True for n in names}
    pot = 10 * len(names)
    to_call = 10
    print("=== New Round ===")
    print("Your hand:", hands["You"])
    print("Starting pot:", pot)

    def betting(stage_name):
        nonlocal pot, to_call
        print(f"\n--- {stage_name} ---")
        for n in names:
            if not active[n]:
                continue
            if n == "You":
                action, p = get_human_action(model, hands[n], community, pot, to_call=to_call)
            else:
                action, p = learned_ai_action(model, hands[n], community, player_name=n)
            if action == "fold":
                active[n] = False
                print(f"{n} folds (p={p:.2f})")
            elif action == "call":
                pot += to_call
                print(f"{n} calls (+{to_call} to pot).")
            elif action == "raise":
                pot += to_call + 20 
                print(f"{n} raises (+{to_call + 20} to pot).")
            else:
                pot += to_call
                print(f"{n} calls (+{to_call} to pot).")

    # Preflop
    betting("Preflop")
    if sum(active.values()) == 1:
        winner = [n for n in names if active[n]][0]
        print(f"\n{winner} wins by folds, pot={pot}")
        print ("\nFinal hands:")
        for n in names:
            print(f"{n} had: {hands[n]}")
            print(f"Evaluated Hand: {evaluate_hand(hands[n] + community)}")
        return

    # Flop
    d.deal()  
    community += [d.deal(), d.deal(), d.deal()]
    betting("Flop")
    if sum(active.values()) == 1:
        winner = [n for n in names if active[n]][0]
        print(f"\n{winner} wins by folds, pot={pot}")
        print ("\nFinal hands:")
        for n in names:
            print(f"{n} had: {hands[n]}")
            print(f"Evaluated Hand: {evaluate_hand(hands[n] + community)}")
        return

    # Turn
    d.deal()
    community.append(d.deal())
    betting("Turn")
    if sum(active.values()) == 1:
        winner = [n for n in names if active[n]][0]
        print(f"\n{winner} wins by folds, pot={pot}")
        print ("\nFinal hands:")
        for n in names:
            print(f"{n} had: {hands[n]}")
            print(f"Evaluated Hand: {evaluate_hand(hands[n] + community)}")
        return

    # River
    d.deal()
    community.append(d.deal())
    betting("River")

    # Showdown
    contenders = [n for n in names if active[n]]
    print("\n--- Showdown ---")
    best = -1
    winners = []
    for n in contenders:
        score = hand_strength_score(hands[n] + community)
        name = evaluate_hand(hands[n] + community)
        print(f"{n} final hand:", hands[n])
        print(f" -> {name} (score {score:.2f})")
        if score > best:
            best = score
            winners = [n]
        elif abs(score - best) < 1e-9:
            winners.append(n)

    if len(winners) == 1:
        print(f"\nWinner: {winners[0]} (pot={pot})")
    else:
        print(f"\nTie: {winners} split pot {pot}")

    print ("\nFinal hands:")
    for n in names:
        print(f"{n} had: {hands[n]}")
        print(f"Evaluated Hand: {evaluate_hand(hands[n] + community)}")

if __name__ == "__main__":
    random.seed()
    np.random.seed(0)
    tf.random.set_seed(0)

    print("Building training set (this may take a bit)...")
    X, y = build_training_set_learned(samples=1200, num_opponents=4, mc_iters=70)

    net = train_keras_model(X, y, epochs=18, batch_size=64, lr=1e-3)
    model = KerasWrapper(net)

    play_round(model, num_bots=4)
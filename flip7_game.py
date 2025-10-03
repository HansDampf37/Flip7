import random
from collections import OrderedDict


class Flip7Game:
    NUM_CARDS = 22

    def __init__(self):
        self._init_deck_definition()
        self.reset()

    def _init_deck_definition(self):
        """Define full deck with frequencies of all card types."""
        self.card_frequencies = {}

        # Number cards: 0 appears once, 1 appears once, ..., 12 appears 12 times
        for i in range(13):
            self.card_frequencies[i] = max(1, i)

        # Modifier cards (example counts, can be tweaked)
        self.card_frequencies.update({
            "+2": 4,
            "+4": 3,
            "+6": 2,
            "+8": 2,
            "+10": 1,
            "x2": 1
        })

        # Action cards
        self.card_frequencies.update({
            "Second Chance": 3,
            "Flip Three": 2,
            "Freeze": 2
        })

    def reset(self):
        """Resets the game state: full stack, empty hand, score = 0."""
        self.stack = []
        for card, count in self.card_frequencies.items():
            self.stack.extend([card] * count)
        random.shuffle(self.stack)

        self.round_over = False
        self.hand = []
        self.card_counts_in_hand = OrderedDict({ card: 0 for card in self.card_frequencies.keys() })
        self.unique_numbers = set()
        self.has_second_chance = False
        self.score = 0
        self.additive_modifier = 0
        self.multiplicative_modifier = 1
        self.last_score_diff = 0
        self.pending_flip_three = 0

    def get_score(self):
        return self.score * self.multiplicative_modifier + self.additive_modifier

    def in_(self):
        """Draws one card from the stack and updates game state accordingly."""
        if self.round_over:
            raise RuntimeError("Round is already over")

        prev_score = self.get_score()
        card = self.stack.pop()
        self.hand.append(card)
        self.card_counts_in_hand[card] += 1

        # Action: Flip Three
        if card == "Flip Three":
            self.pending_flip_three += 3
            self._draw_pending_flip_three()

        # Action: Freeze
        elif card == "Freeze":
            self.round_over = True

        # Action: Second Chance
        elif card == "Second Chance":
            self.has_second_chance = True

        # Modifier cards
        elif isinstance(card, str) and card.startswith("+"):
            self.additive_modifier += int(card[1:])

        elif isinstance(card, str) and card.startswith("x"):
            self.multiplicative_modifier *= int(card[1:])

        # Number card
        elif isinstance(card, int):
            if card in self.unique_numbers:
                if self.has_second_chance:
                    self.has_second_chance = False
                    # Do not add card to unique set, ignore duplicate
                else:
                    # Bust: round ends, score reset
                    self.round_over = True
                    self.score = 0
                    self.multiplicative_modifier = 1
                    self.additive_modifier = 0
            else:
                self.unique_numbers.add(card)
                self.score += card

        # Check Flip 7
        if not self.round_over and len(self.unique_numbers) == 7:
            self.score += 15  # Flip 7 bonus
            self.round_over = True

        self.last_score_diff = self.get_score() - prev_score

    def _draw_pending_flip_three(self):
        """Handles multiple draws when Flip Three is triggered."""
        while self.pending_flip_three > 0 and not self.round_over and self.stack:
            self.pending_flip_three -= 1
            self.in_()  # Recursive draw, actions handled individually

    def get_score_difference(self):
        """Returns the change in score caused by the last in_() call."""
        return self.last_score_diff


if __name__ == "__main__":
    game = Flip7Game()
    game.reset()

    while not game.round_over:
        print("Current hand: ", game.hand)
        print("Current hand: ", game.card_counts_in_hand)
        if input("Continue (Y/N)?").lower() in ["yes", "y"]:
            game.in_()
            print("Last score diff:", game.get_score_difference())
        else:
            break

    print("Final score:", game.score)
    print("Cards in hand:", game.hand)
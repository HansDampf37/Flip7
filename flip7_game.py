import random
from collections import OrderedDict


class Flip7Game:
    """Flip7 card game simulation environment."""

    # Explicit card ordering for consistent feature encoding
    CARD_TYPES = (
        list(range(13))  # 0–12 numbers
        + ["+2", "+4", "+6", "+8", "+10", "x2"]
        + ["Second Chance", "Flip Three", "Freeze"]
    )
    NUM_CARDS = len(CARD_TYPES)

    def __init__(self):
        self._init_deck_definition()
        self.reset()

    def _init_deck_definition(self):
        """Define full deck with frequencies of all card types."""
        self.card_frequencies = {}

        # Number cards: 0 appears once, 1 appears once, ..., 12 appears 12 times
        for i in range(13):
            self.card_frequencies[i] = max(1, i)

        # Modifier cards
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
        """Resets the game state: full shuffled stack, empty hand, zeroed score."""
        # Create fresh shuffled stack
        self.stack = []
        for card, count in self.card_frequencies.items():
            self.stack += [card] * count
        random.shuffle(self.stack)

        # Reset state
        self.round_over = False
        self.hand = []
        self.card_counts_in_hand = OrderedDict({card: 0 for card in self.CARD_TYPES})
        self.unique_numbers = set()
        self.has_second_chance = False
        self.score = 0
        self.additive_modifier = 0
        self.multiplicative_modifier = 1
        self.last_score_diff = 0
        self.pending_flip_three = 0

    def get_score(self) -> int:
        """Returns current score after applying modifiers."""
        return self.score * self.multiplicative_modifier + self.additive_modifier

    def in_(self):
        """Draw one card and update state accordingly."""
        if self.round_over:
            raise RuntimeError("Round is already over")
        if not self.stack:
            self.round_over = True
            return

        prev_score = self.get_score()
        card = self.stack.pop()
        self.hand.append(card)
        self.card_counts_in_hand[card] += 1

        # Action: Flip Three
        if card == "Flip Three":
            self.pending_flip_three += 3

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
                    # Ignore duplicate safely
                else:
                    # Bust: reset score and end round
                    self.round_over = True
                    self.score = 0
                    self.multiplicative_modifier = 1
                    self.additive_modifier = 0
            else:
                self.unique_numbers.add(card)
                self.score += card

        # Check Flip7 bonus
        if not self.round_over and len(self.unique_numbers) == 7:
            self.score += 15
            self.round_over = True

        # Handle any pending Flip Three draws iteratively
        while self.pending_flip_three > 0 and not self.round_over and self.stack:
            self.pending_flip_three -= 1
            self.in_()  # safe because loop ensures bounded recursion

        self.last_score_diff = self.get_score() - prev_score

    def get_score_difference(self) -> int:
        """Returns the score change caused by the last in_() call."""
        return self.last_score_diff


if __name__ == "__main__":
    game = Flip7Game()
    game.reset()

    while not game.round_over:
        print("Current hand:", game.hand)
        print("Card counts:", dict(game.card_counts_in_hand))
        if input("Continue (Y/N)? ").strip().lower() in ["yes", "y"]:
            game.in_()
            print("Last score diff:", game.get_score_difference())
            print("Score:", game.get_score())
        else:
            break

    print("Final score:", game.get_score())
    print("Cards in hand:", game.hand)

import numpy as np
import torch
from colorama import Fore, Style, init

from src.flip7_game import Flip7Game
from src.regressor import Flip7Model

init(autoreset=True)


class Flip7Helper:
    def __init__(self):
        self.game = Flip7Game()
        self.model = Flip7Model()
        self.model.load_state_dict(torch.load("models/model", weights_only=True))
        self.model.eval()

    def on_draw_card(self, card):
        """
        Move card to the top of the deck and invoke the in method
        :param card: The card that was drawn in reality
        :return:
        """
        self.game.stack.remove(card)
        self.game.stack.append(card)
        self.game.in_()

    def on_opp_draw_card(self, card):
        """
        When any opponent draws a card it is no longer available to us
        :param card: the card
        :return:
        """
        self.game.stack.remove(card)

    def print_info(self):
        score = self.game.get_score()

        expected_score_change = self.game.mc_sample_expected_score_difference_of_in(3000)

        color = Fore.GREEN if expected_score_change >= 0 else Fore.RED
        recommended_act = f"{color}{"IN" if expected_score_change > 0 else "OUT"}{Style.RESET_ALL}"
        print(f"Approximated score: {score} {color}{expected_score_change:+.2f}{Style.RESET_ALL} -> {recommended_act}")

        obs = np.array([self.game.card_counts_in_hand[c] for c in Flip7Game.CARD_TYPES], dtype=np.float32)
        observation = torch.tensor(obs).unsqueeze(0)
        expected_score_change = self.model(observation).item()

        color = Fore.GREEN if expected_score_change >= 0 else Fore.RED
        recommended_act = f"{color}{"IN" if expected_score_change > 0 else "OUT"}{Style.RESET_ALL}"
        print(f"Predicted score: {score} {color}{expected_score_change:+.2f}{Style.RESET_ALL} -> {recommended_act}")


def main():
    print("Welcome to FLIP7-Helper!\n"
          "--------------------------------------------------------------------------------------------------\n"
          "Add cards to your hand with 'add <card1>,<card2>,<card3>' for example 'add +2', 'add 7,12' or 'add 10,Freeze'.\n"
          "Instead of 'd Flip Three' you should just add the three cards directly.\n"
          "Remove cards from the stack with 'rm <card1>,<card2>,<card3>,...' for example 'rm Second Chance', or 'rm 6,7'")

    def preprocess_card(card: str):
        if card.startswith("+") or card.startswith("x"):
            return card.strip()
        try:
            return int(card.strip())
        except ValueError:
            return card.title().strip()

    flip7 = Flip7Helper()
    while True:
        inp = input()
        try:
            command, args = inp.split(" ", 1)
        except ValueError:
            print(f"Separate command and arguments by a space")
            continue
        if command == "add":
            cards = [preprocess_card(card) for card in args.split(",")]
            for card in cards:
                flip7.on_draw_card(card)
        elif command == "rm":
            cards = [preprocess_card(card) for card in args.split(",")]
            for card in cards:
                flip7.on_opp_draw_card(card)
        else:
            print("Invalid command")
        flip7.print_info()


if __name__ == "__main__":
    main()

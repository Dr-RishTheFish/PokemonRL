# max_damage.py
from poke_env.player.player import Player

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            return self.create_order(max(battle.available_moves, key=lambda move: move.base_power))
        else:
            return self.choose_random_move(battle)

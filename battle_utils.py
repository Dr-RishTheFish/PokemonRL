import torch
from poke_env.environment.move_category import MoveCategory

def calculate_damage(move, attacker, defender, pessimistic, is_bot_turn):
    if move is None:
        print("Why is move none?")
        return 0
    if move.category == MoveCategory.STATUS:
        return 0

    damage = move.base_power
    ratio = 1
    
    if move.category == MoveCategory.PHYSICAL:
        ratio = calculate_physical_ratio(attacker, defender, is_bot_turn)
    elif move.category == MoveCategory.SPECIAL:
        ratio = calculate_special_ratio(attacker, defender, is_bot_turn)
    
    damage = damage * ratio
    level_multiplier = ((2 * attacker.level) / 5 ) + 2
    damage = damage * level_multiplier
    damage = (damage / 50) + 2
    
    if pessimistic:
        damage = damage * 0.85
    if move.type == attacker.type_1 or move.type == attacker.type_2:
        damage = damage * 1.5
    type_multiplier = defender.damage_multiplier(move)
    damage = damage * type_multiplier

    return damage

def calculate_physical_ratio(attacker, defender, is_bot_turn):
    if is_bot_turn:
        attack = attacker.stats["atk"]
        defense = 2 * defender.base_stats["def"] + 36
        defense = ((defense * defender.level) / 100) + 5
    else:
        defense = defender.stats["def"]
        attack = 2 * attacker.base_stats["atk"] + 36
        attack = ((attack * attacker.level) / 100) + 5
    return attack / defense   

def calculate_special_ratio(attacker, defender, is_bot_turn):
    if is_bot_turn:
        spatk = attacker.stats["spa"]
        spdef = 2 * defender.base_stats["spd"] + 36
        spdef = ((spdef * defender.level) / 100) + 5
    else: 
        spdef = defender.stats["spd"]
        spatk = 2 * attacker.base_stats["spa"] + 36
        spatk = ((spatk * attacker.level) / 100) + 5
    return spatk / spdef

def opponent_can_outspeed(my_pokemon, opponent_pokemon):
    my_speed = my_pokemon.stats["spe"]
    opponent_max_speed = 2 * opponent_pokemon.base_stats["spe"] + 52
    opponent_max_speed = ((opponent_max_speed * opponent_pokemon.level) / 100) + 5
    return opponent_max_speed > my_speed

def calculate_total_HP(pokemon, is_dynamaxed): 
    HP = pokemon.base_stats["hp"] * 2 + 36
    HP = ((HP * pokemon.level) / 100) + pokemon.level + 10
    if is_dynamaxed: 
        HP = HP * 2
    return HP

def get_defensive_type_multiplier(my_pokemon, opponent_pokemon):
    first_multiplier = my_pokemon.damage_multiplier(opponent_pokemon.type_1)
    second_multiplier = my_pokemon.damage_multiplier(opponent_pokemon.type_2) if opponent_pokemon.type_2 else 1
    return max(first_multiplier, second_multiplier)

def embed_battle(battle):
    opp_remaining_pokemon = len([mon for mon in battle.opponent_team.values() if not mon.fainted]) / 6
    player_remaining_pokemon = len([mon for mon in battle.team.values() if not mon.fainted]) / 6

    moves_base_power = torch.zeros(4)
    move_type_multiplier = torch.ones(4)

    for i, move in enumerate(battle.available_moves):
        moves_base_power[i] = get_move_base_power(move) / 100
        move_type_multiplier[i] = get_move_type_multiplier(move, battle.opponent_active_pokemon)

    return torch.cat([
        moves_base_power,
        move_type_multiplier,
        torch.tensor([opp_remaining_pokemon, player_remaining_pokemon])
    ])

def compute_reward(battle, fainted_value=2, hp_value=1, status_value=0.5, victory_value=30):
    reward = 0
    reward += fainted_value * (len([mon for mon in battle.opponent_team.values() if mon.fainted]) - len([mon for mon in battle.team.values() if mon.fainted]))
    reward += hp_value * (sum([mon.current_hp for mon in battle.team.values()]) - sum([mon.current_hp for mon in battle.opponent_team.values()]))
    reward += status_value * (len([mon for mon in battle.opponent_team.values() if mon.status]) - len([mon for mon in battle.team.values() if mon.status]))
    if battle.won:
        reward += victory_value
    elif battle.lost:
        reward -= victory_value
    return reward

def get_move_base_power(move):
    return move.base_power

def get_move_type_multiplier(move, opponent):
    return move.type.damage_multiplier(opponent.type_1, opponent.type_2)

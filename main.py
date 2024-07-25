# -*- coding: utf-8 -*-
import asyncio
import torch
from poke_env import AccountConfiguration
from max_damage import MaxDamagePlayer
from smart_damage import SmartDamagePlayer
from reinforcmentLearningBot import RLPlayer, DQN, train  # Corrected the module name
from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

account_config = AccountConfiguration('Tha1231232', 'pokemoniscool')
account_config_2 = AccountConfiguration('Sasnakisadoggo', 'joebobiscool')

# Cynthia's Team
cynthiaBaddieTeam = [
    TeambuilderPokemon(
        species="Garchomp",
        item="Focus Sash",
        ability="Rough Skin",
        moves=["Stealth Rock", "Outrage", "Earthquake", "Fire Blast"],
        nature="Naive",
        evs=[0, 252, 0, 4, 252, 0],
        ivs=[31, 31, 31, 31, 31, 31]
    ),
    TeambuilderPokemon(
        species="Spiritomb",
        item="Black Glasses",
        ability="Infiltrator",
        moves=["Pursuit", "Shadow Sneak", "Sucker Punch", "Will-o-Wisp"],
        nature="Adamant",
        evs=[252, 252, 0, 0, 4, 0],
        ivs=[31, 31, 31, 31, 31, 31]
    ),
    TeambuilderPokemon(
        species="Togekiss",
        item="Leftovers",
        ability="Serene Grace",
        moves=["Nasty Plot", "Air Slash", "Thunder Wave", "Roost"],
        nature="Calm",
        evs=[252, 0, 120, 0, 132, 4],
        ivs=[31, 31, 31, 31, 31, 31]
    ),
    TeambuilderPokemon(
        species="Milotic",
        item="Life Orb",
        ability="Marvel Scale",
        moves=["Hydro Pump", "Ice Beam", "Recover", "Dragon Tail"],
        nature="Modest",
        evs=[252, 0, 0, 252, 4, 0],
        ivs=[31, 31, 31, 31, 31, 31]
    ),
    TeambuilderPokemon(
        species="Lucario",
        item="Air Balloon",
        ability="Justified",
        moves=["Swords Dance", "Extremespeed", "Bullet Punch", "Close Combat"],
        nature="Adamant",
        evs=[0, 252, 4, 0, 252, 0],
        ivs=[31, 31, 31, 31, 31, 31]
    ),
    TeambuilderPokemon(
        species="Roserade",
        item="Choice Scarf",
        ability="Natural Cure",
        moves=["Leaf Storm", "Sludge Bomb", "Toxic Spikes", "Sleep Powder"],
        nature="Timid",
        evs=[0, 252, 4, 0, 252, 0],
        ivs=[31, 31, 31, 31, 31, 31]
    )
]

# Convert TeambuilderPokemon objects to Pokémon Showdown format strings
def pokemon_to_string(pokemon):
    evs_str = ' / '.join(f"{ev} {stat}" for ev, stat in zip(pokemon.evs, ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]))
    ivs_str = ' / '.join(f"{iv} {stat}" for iv, stat in zip(pokemon.ivs, ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]))
    moves_str = '\n- '.join(pokemon.moves)
    
    return f"""
{pokemon.species} @ {pokemon.item}
Ability: {pokemon.ability}
EVs: {evs_str}
IVs: {ivs_str}
{pokemon.nature} Nature
- {moves_str}
"""

team_string = '\n'.join(pokemon_to_string(pokemon) for pokemon in cynthiaBaddieTeam)

# Create a ConstantTeambuilder with the defined team
team_builder = ConstantTeambuilder(team=team_string)

# Print the team to verify
print(team_string)

random_player = SmartDamagePlayer(account_configuration=account_config_2, battle_format='gen9nationaldex', accept_open_team_sheet=True, team=team_builder)
player = SmartDamagePlayer(account_configuration=account_config, battle_format='gen9nationaldex', accept_open_team_sheet=True, team=team_builder)
reinforcement_player = RLPlayer(opponent=random_player)  # Provide the opponent here

n_action = 10  # Number of possible actions, adjust according to your game
model = DQN(n_action)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
loss_fn = torch.nn.MSELoss()

async def main():
    for _ in range(10):
        await player.battle_against(random_player)
    print(f"Tharun won {player.n_won_battles} / 10 battles")

    # Train the reinforcement learning player
    train(reinforcement_player, model, optimizer, loss_fn, num_steps=10000)

if __name__ == "__main__":
    asyncio.run(main())

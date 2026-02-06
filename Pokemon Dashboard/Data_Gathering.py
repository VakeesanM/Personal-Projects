import requests
import pandas as pd
import numpy as np


pokemon_name = []
pokemon_url =[]
pokemon_id = []
pokemon_type_1 = []
pokemon_type_2 = []
pokemon_weight = []
pokemon_height = []
pokemon_hp = []
pokemon_att = []
pokemon_def = []
pokemon_specatt = []
pokemon_specdef =[]
pokemon_speed = []
pokemon_gen = []
pokemon_region = []
def GetPokemonname():
    response = requests.get('https://pokeapi.co/api/v2/pokemon?limit=100000&offset=0').json()
    for pokemon in response['results']:
        pokemon_name.append(pokemon['name'])
        pokemon_url.append(pokemon['url'])

def pokemon_data(Pokemon_Url):
    for url in Pokemon_Url:
        response = requests.get(url).json()
        pokemon_id.append(response['id'])
        types = response['types']
        pokemon_type_1.append(types[0]['type']['name'])
        if len(types) > 1:
            pokemon_type_2.append(types[1]['type']['name'])
        else:
            pokemon_type_2.append('No Type')
        pokemon_weight.append(response['weight'])
        pokemon_height.append(response['height'])
        pokemon_hp.append(response['stats'][0]['base_stat'])
        pokemon_att.append(response['stats'][1]['base_stat'])
        pokemon_def.append(response['stats'][2]['base_stat'])
        pokemon_specatt.append(response['stats'][3]['base_stat'])
        pokemon_specdef.append(response['stats'][4]['base_stat'])
        pokemon_speed.append(response['stats'][5]['base_stat'])

def generation(data):
    for x in data:
        if x > 0 and x < 152: #
            pokemon_gen.append(1)
        elif x> 151 and x <= 251: #
            pokemon_gen.append(2)
        elif x> 251 and x < 387: #
            pokemon_gen.append(3)
        elif x> 386 and x < 494:
            pokemon_gen.append(4)
        elif x> 493 and x < 650:
            pokemon_gen.append(5)
        elif x> 649 and x < 722:
            pokemon_gen.append(6)
        elif x> 721 and x < 810:
            pokemon_gen.append(7)
        elif x> 809 and x < 905:
            pokemon_gen.append(8)
        elif x> 904 and x< 1026:
            pokemon_gen.append(9)
        else:
            pokemon_gen.append('Transformation')



gen_reg_dict = {1:'Kanto',
                2:'Johto',
                3:'Hoenn',
                4:'Sinnoh',
                5:'Unova',
                6:'Kalos',
                7:'Alola',
                8:'Galar',
                9:'Paldea',
                'Transformation': 'No Region'}

def region(data):
    for x in data:
        pokemon_region.append(gen_reg_dict[x])

GetPokemonname()
pokemon_data(pokemon_url)
generation(pokemon_id)
region(pokemon_gen)



data = {'Name':pokemon_name, 
                             'ID':pokemon_id,
                             'Primary Type': pokemon_type_1,  
                             'Secondary Type': pokemon_type_2,
                             'Height': pokemon_height,
                             'Weight': pokemon_weight,
                             'HP':pokemon_hp,
                             'Attack': pokemon_att,
                             'Defense': pokemon_def,
                             'Special Attack': pokemon_specatt,
                             'Special Defense':pokemon_specdef,
                             'Speed':pokemon_speed,
                             'Generation': pokemon_gen,
                             'Region': pokemon_region
                             }

Pokemon_data = pd.DataFrame(data)



Pokemon_data.to_csv('Pokemon.csv', index=False)


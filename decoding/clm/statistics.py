characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ '." + "\n"  # no blank

character_list = [c for c in characters]
character_set = dict(zip(character_list, range(len(character_list))))
inverse_character_set = dict(zip(range(len(character_list)), character_list))

print("hello world")
#sastojci koji fale useru
import ast
from text_processing import (normalise_lemma, split_ing, normalise)
# Example usage:
# my_ingredients = ['chicken', 'oil', 'salt', 'pepper']
# missing_report = analyze_missing_ingredients(my_ingredients, df)


def analiziraj(user_ingredients, recipe):
    user_set = set(ing.strip().lower() for ing in user_ingredients)
    recipe_ing_list = [i.strip().lower() for i in recipe["ingredients"]]
    #recipe_ing_list = [i.strip().lower() for i in ast.literal_eval(recipe["ingredients"])]
    recipe_set = set(recipe_ing_list)

    missing = recipe_set - user_set
    match_count = len(recipe_set) - len(missing)
    match_percent = (match_count / len(recipe_set)) * 100 if recipe_set else 0

    return {
        "missing_ingredients": list(missing),
        "missing_count": len(missing),
        "match_percentage": round(match_percent, 2)
    }



#nema u datasetu
def report_missing_ingredients(user_ingredients, vocabulary):
    """
    Reports which user ingredients were found in the database
    and which are missing, giving clear feedback to the user.
    """
    found = []
    missing = []
    for ing in user_ingredients:
        for part in split_ing(ing):
            c = normalise(part)
            if c and len(c) > 2:
                if c in vocabulary:
                    found.append(c)
                else:
                    missing.append(c)
    print(f"Recognised ingredients ({len(found)}): {found}")
    if missing:
        print(f"Ingredients not found in our database ({len(missing)}): {missing}")
    if not found:
        print("None of your ingredients were recognised. Please try different ingredients.")
        return None
    return found

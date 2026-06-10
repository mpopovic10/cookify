"""
User feedback and ingredient analysis utilities.
"""

from src.preprocessing import split_ingredients, normalise, get_tokens


def report_missing_ingredients(user_ingredients, vocabulary, lemma=False):
    """
    Reports which user ingredients were found in the database
    and which are missing, giving clear feedback to the user.

    :param user_ingredients: list of ingredient strings from user
    :param vocabulary: set of valid ingredient tokens (from Counter keys)
    :param lemma: if True, applies lemmatization during normalisation
    :return: set of found ingredients, or None if none found
    """
    found = set()
    missing = set()

    for ing in user_ingredients:
        for part in split_ingredients(ing):
            tokens = get_tokens([part], lemma=lemma)
            for token in tokens:
                if token in vocabulary:
                    found.add(token)
                else:
                    missing.add(token)

    print(f"✓ Recognised ingredients ({len(found)}): {', '.join(sorted(found))}")
    if missing:
        print(f"✗ Ingredients not found in database ({len(missing)}): {', '.join(sorted(missing))}")
    if not found:
        print("None of your ingredients were recognised. Please try different ingredients.")
        return None

    return found


def analyze_missing_ingredients(user_ingredients, df, lemma=False):
    """
    Analyzes which ingredients the user is missing for each recipe.
    Returns recipes ranked by how many ingredients the user needs to add.

    :param user_ingredients: list of ingredient strings from user
    :param df: recipe DataFrame with Cleaned_Ingredients column
    :param lemma: if True, applies lemmatization during normalisation
    :return: sorted list of recipe analysis dicts
    """
    user_tokens = set()
    for ing in user_ingredients:
        tokens = get_tokens([ing], lemma=lemma)
        user_tokens.update(tokens)

    analysis_results = []

    for index, row in df.iterrows():
        recipe_tokens = set(get_tokens(row['Cleaned_Ingredients'], lemma=lemma))

        # Calculate what the user is MISSING
        missing = recipe_tokens - user_tokens

        # Calculate match percentage
        match_count = len(recipe_tokens) - len(missing)
        match_percent = (match_count / len(recipe_tokens) * 100) if len(recipe_tokens) > 0 else 0

        analysis_results.append({
            'title': row['Title'],
            'missing_ingredients': sorted(list(missing)),
            'missing_count': len(missing),
            'match_percentage': round(match_percent, 2),
            'total_ingredients': len(recipe_tokens)
        })

    # Sort by fewest missing ingredients (best matches first)
    analysis_results = sorted(analysis_results, key=lambda x: (x['missing_count'], -x['match_percentage']))

    return analysis_results


def print_analysis_results(analysis_results, top_n=5):
    """
    Pretty-prints the analysis results in a readable format.

    :param analysis_results: output from analyze_missing_ingredients()
    :param top_n: number of top recipes to display
    """
    print(f"\n{'=' * 80}")
    print(f"Top {top_n} Closest Recipe Matches")
    print(f"{'=' * 80}\n")

    for i, recipe in enumerate(analysis_results[:top_n], 1):
        print(f"{i}. {recipe['title']}")
        print(
            f"   Match: {recipe['match_percentage']}% ({recipe['total_ingredients'] - recipe['missing_count']}/{recipe['total_ingredients']} ingredients)")
        if recipe['missing_ingredients']:
            print(f"   Missing: {', '.join(recipe['missing_ingredients'][:5])}", end="")
            if len(recipe['missing_ingredients']) > 5:
                print(f" + {len(recipe['missing_ingredients']) - 5} more")
            else:
                print()
        print()
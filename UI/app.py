import streamlit as st
import base64
from pathlib import Path
from recommender import (recommend_pipeline, df_5k)
from preprocessing import (report_missing_ingredients, analiziraj)

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

if "results" not in st.session_state:
    st.session_state.results = []

if "selected_recipe" not in st.session_state:
    st.session_state.selected_recipe = None

#pozadina stranice
bg_img = get_base64("../../Downloads/UI - Cookify/UI - Cookify/assests/background3.png")
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True)

#kartica za recepte
st.markdown("""
    <style>
    
    .recipe-card {
        background-color: white;
        padding: 20px;
        border-radius: 20px;
        margin-bottom: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        border-left: 8px solid #ff914d;
    }
    
    .recipe-title {
        font-size: 24px;
        font-weight: bold;
    }
    
    .recipe-info {
        font-size: 16px;
        color: #555;
    }
    
    </style>
    """, unsafe_allow_html=True)

#expendeder
st.markdown(
    """
    <style>

    /* Expander container */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        background-color: #442203;
        border: 1px solid #333333;
    }

    /* Expander header */
    div[data-testid="stExpander"] summary {
        background-color: #734110;
        color: white;
        padding: 10px;
        border-radius: 12px;
        font-weight: 600;
    }

    /* Expander content */
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        background-color: #734110;
        color: white;
        padding: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

#konfiguracija stranice (sta ce biti na tabu na vrhu)
st.set_page_config(
    page_title="Recipe Recommender",
    page_icon="Cookify_logo.png",
    layout="wide")



#pravljenje 3 kolone za front page, da bi mogao logo da se uklopi
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("Cookify_logo_2tekst.png", width=680)

#preuzimanje google fontova
st.markdown(
    '<link rel="stylesheet" href="https://googleapis.com">',
    unsafe_allow_html=True
)

#naslov odmah ispod loga
st.markdown("<h1 style='text-align: left; margin-top: -195px; font-size: 75px; color: #4b2d23;'>Hi, Chef!</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 24px; margin-top: -85px; color: #4b2d23'"
            ">You keep throwing away ingredients because you don't know what to cook? I have a solution for you!<br>"
            "Enter the ingredients you wish to use and I will recommend you what to make.</p>"
            , unsafe_allow_html=True)

#TF-IDF koristi:
all_ingredients = set()
vocabulary = set(all_ingredients)

#prolazi kroz sve recepte u datasetu i izvlači sve jedinstvene sastojke
for row in df_5k["Cleaned_Ingredients"]:
    try:
        ingredients = eval(row) #pretvara string u listu
        for ing in ingredients: #prolazi kroz svaki sastojak u listi
            all_ingredients.add(str(ing)) #dodaje ga u set, duplikati se ignorišu jer je set
    except:
        pass #rezultat su svi sastojci iz all_ingredients koji nisu dupli
ingredient_list = sorted(list(all_ingredients)) #pravi sortiranu listu



st.markdown("<h2 style='color: #4b2d23;'>Select Ingredients:</h2>", unsafe_allow_html=True)
st.markdown(#CSS za multiselector
    """
    <style>
        div[data-baseweb="select"] > div {
            background-color: #d59e58 !important;
            color: #d59e58 !important;
        }
        div[data-baseweb="select"] * {
            color: #FFFFFF !important;
        }
    </style>
""", unsafe_allow_html=True)
selected = st.multiselect( #multiselect, choosing an ingredient from ingredient_list
    "",
    ingredient_list,
    key="selected_ingredients",
    placeholder="Choose ingredients..."
)

#button style
st.markdown("""
    <style>
        .stButton button {
            background-color: #8B4513 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)
#tag (selected ingredient) style
st.markdown("""
    <style>
        span[data-baseweb="tag"] {
            background-color: #8B4513 !important;
        }
    </style>
""", unsafe_allow_html=True)



#when button clicked
find_clicked = st.button("Find Recipes")
if find_clicked:
    if len(selected) == 0:
        st.warning("Select at least one ingredient.")
    else:
        output = recommend_pipeline( #recommendation function call, from recommender.py
            selected,
            top_n=5
        )
        st.session_state.results = output["results"]
        st.session_state.selected_recipe = None

        if st.session_state.results:
            st.markdown("<h3 style='color: #4b2d23;'>Recommended Recipes:</h3>", unsafe_allow_html=True)
            not_in_dataset = report_missing_ingredients(selected, vocabulary)
            for i, recipe in enumerate(st.session_state.results):
                #print(recipe.keys())
                #print(recipe.get("instructions"))
                matched = ", ".join(recipe["matched_tokens"])
                image_path = f"images/Food Images/{recipe['image_name']}.jpg"
                missing = analiziraj(selected, recipe)

                if Path(image_path).exists():
                    image_base64 = image_to_base64(image_path)
                else:
                    image_base64 = ""
                st.markdown(
                    f"""
                    <div class="recipe-card">
                        <div style="
                            display:flex;
                            justify-content:space-between;
                            align-items:center;">
                            <div style="width:70%;">
                                <div class="recipe-title">
                                    🍽 {recipe['title']}
                                </div><br>
                                <div class="recipe-info">
                                    <b>Similarity:</b> {recipe['sbert_score'] * 100:.1f}%<br>
                                    <b>Ingredients you don't have ({missing['missing_count']}):</b> {', '.join(missing['missing_ingredients'])}<br>
                                </div>
                            </div>
                            <div style="width:25%; text-align:right;">
                                <img
                                    src="data:image/jpeg;base64,{image_base64}"
                                    style="
                                        width:230px;
                                        height:150px;
                                        object-fit:cover;
                                        border-radius:15px;">
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Expander samo za detalje
                with st.expander("View Recipe"):

                    st.subheader("Ingredients")

                    for ing in recipe["ingredients"]:
                        st.write(f"• {ing}")

                    # ako imaš instructions
                    if "instructions" in recipe:
                        st.subheader("Preparation")
                        st.write(recipe["instructions"])

if st.session_state.selected_recipe:
    recipe = st.session_state.selected_recipe
    st.markdown("---")
    st.header(recipe["title"])
    st.subheader("Ingredients")
    for ing in recipe["ingredients"]:
        st.write(f"• {ing}")

                # kasnije ćemo dodati instructions

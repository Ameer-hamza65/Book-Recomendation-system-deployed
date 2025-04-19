import streamlit as st
import pickle
import numpy as np

# Load the saved model and pivot table
with open('book_recommender_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('final_pivot.pkl', 'rb') as f:
    final_pivot = pickle.load(f)

# Set page config
st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f6f6f6;
        }
        .main {
            padding: 2rem;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.05);
        }
        .book-card {
            background-color: #f0f2f6;
            padding: 15px 20px;
            margin: 1px 0;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #4a4a4a;
        }
        .subtitle {
            font-size: 18px;
            color: #6c757d;
        }
    </style>
""", unsafe_allow_html=True)

# Recommend function
def recommend_books(book_name): 
    if book_name not in final_pivot.index:
        return ["Book not found in the dataset"]

    book_id = np.where(final_pivot.index == book_name)[0][0]
    distances, suggestions = model.kneighbors(final_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=10)
    
    recommended_books = [final_pivot.index[suggestion] for suggestion in suggestions[0]]
    return recommended_books

# UI
st.markdown('<div class="main">', unsafe_allow_html=True)

st.markdown('<div class="title">📚 Book Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find your next favorite book based on what you love!</div>', unsafe_allow_html=True)
st.write("")

book_name = st.text_input("Enter the book name:")

if st.button("Get Recommendations"):
    if book_name:
        recommendations = recommend_books(book_name)
        if recommendations == ["Book not found in the dataset"]:
            st.error("❌ Book not found in the dataset. Please try another one.")
        else:
            st.success("✅ Here are some books you might like:")
            for i, book in enumerate(recommendations, 1): 
                st.markdown(f'<div class="book-card">{i}. {book}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a book name.")

st.markdown('</div>', unsafe_allow_html=True)

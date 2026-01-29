import pandas as pd
import numpy as np
import gradio as gr
import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

df = pd.read_csv('data/books_with_emotions.csv')

# --- THUMBNAIL ---

GITHUB_NO_COVER_URL = "https://raw.githubusercontent.com/meleknurb/llm_book_recommender/main/cover_not_found.jpg"

df["large_thumbnail"] = df["thumbnail"] + "&fife=w800"
df["large_thumbnail"] = np.where(
    df["large_thumbnail"].isna(),
    GITHUB_NO_COVER_URL,
    df["large_thumbnail"],
)

# --- VECTOR DATABASE ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

if os.path.exists("db_chroma"):
    db_books = Chroma(persist_directory="db_chroma", embedding_function=embeddings)
else:
    txt_path = "data/tagged_descriptions.txt"
    raw_documents = TextLoader(txt_path, encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(documents, embeddings, persist_directory="db_chroma")

# --- LOGIC ---
def retrieve_semantic_recommendations(query, category, tone):
    recs_with_scores = db_books.similarity_search_with_relevance_scores(query, k=50)
    books_list = [{"isbn13": int(doc.page_content.strip('"').split()[0]), "score": s} for doc, s in recs_with_scores if s > 0.3]

    if not books_list:
        return pd.DataFrame()

    book_recs = df.merge(pd.DataFrame(books_list), on="isbn13")
    
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    tone_map = {"Happy": "joy", "Surprising": "surprise", "Angry": "anger", "Suspenseful": "fear", "Sad": "sadness"}
    if tone in tone_map:
        emotion_col = tone_map[tone]
        book_recs["final_score"] = (book_recs["score"] * 0.7) + (book_recs[emotion_col] * 0.3)
        book_recs.sort_values(by="final_score", ascending=False, inplace=True)
    else:
        book_recs.sort_values(by="score", ascending=False, inplace=True)

    return book_recs.head(min(len(book_recs), 16))

def on_search(query, category, tone):
    recs = retrieve_semantic_recommendations(query, category, tone)

    if recs is None or recs.empty:
        return [], pd.DataFrame()
    
    gallery_items = [(row["large_thumbnail"], row["title"]) for _, row in recs.iterrows()]
    return gallery_items, recs

def reset_interface():
    empty_html = "<div style='text-align: center; padding-top: 100px; color: #94a3b8; border: 2px dashed #e2e8f0; border-radius: 12px; height: 300px;'>Select a book cover to view details.</div>"
    return "", "All", "All", [], empty_html

def get_book_details(recs_state, evt: gr.SelectData):
    selected_index = evt.index
    row = recs_state.iloc[selected_index]
    cover_img = row['large_thumbnail']
    authors = str(row["authors"]).replace(";", ", ")
    full_description = row["description"] if pd.notna(row["description"]) else "No description available."
    year = int(row['published_year']) if pd.notna(row['published_year']) else "Unknown"
    rating = row['average_rating'] if pd.notna(row['average_rating']) else "N/A"
    pages = int(row['num_pages']) if pd.notna(row['num_pages']) else "?"
    
    details_html = f"""
    <div style="background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <img src="{cover_img}" style="width: 100%; max-height: 250px; object-fit: contain; border-radius: 8px; margin-bottom: 15px;">
        
        <h3 style="color: #1e293b; margin-bottom: 5px; line-height: 1.2;">{row['title']}</h3>
        <p style="color: #6366f1; font-weight: 600; font-size: 0.9em; margin-bottom: 6px;">✍️ {authors}</p>
        
        <div style="display: flex; gap: 10px; font-size: 0.8em; color: #94a3b8; margin-bottom: 10px;">
            <span>⭐ {rating} Rating</span> | 
            <span>📄 {pages} Pages</span> | 
            <span>📅 {year}</span>
        </div>

        <hr style="border: 0; border-top: 1px solid #f1f5f9; margin: 9px 0;">
        
        <div style="padding-right: 5px; margin-bottom: 15px;">
            <p style="color: #475569; line-height: 1.6; font-size: 0.95em;">{full_description}</p>
        </div>
        
        <div style="margin-top: 15px;">
            <span style="background: #e0e7ff; color: #4338ca; padding: 5px 12px; border-radius: 20px; font-size: 0.75em; font-weight: bold;">{row['simple_categories']}</span>
        </div>
    </div>
    """
    return details_html

# --- INTERFACE ---
categories = ["All"] + sorted(df["simple_categories"].unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
).set(
    body_background_fill="#e5e7eb",
    background_fill_secondary="#f3f4f6",
    block_background_fill="#f9fafb",
    block_border_width="1px"
)

with gr.Blocks(theme=custom_theme, title="Libro") as dashboard:
    current_recs = gr.State()

    gr.Markdown("""
            # 📚 Libro AI
            *Your intelligent gateway to the next favorite story*
    """)
    
    with gr.Row():
        # Left Side: Search Box
        with gr.Column(scale=2):
            user_query = gr.Textbox(
                label="Search by Description", 
                placeholder="Describe the story or feeling you're looking for...", 
                lines=4
            )
        
        # Right Side: Filters and Buttons
        with gr.Column(scale=2):
            with gr.Row():
                cat_drop = gr.Dropdown(choices=categories, label="Genre / Category", value="All")
                tone_drop = gr.Dropdown(choices=tones, label="Emotional Tone", value="All")
            
            with gr.Row():
                gr.Markdown(" ")
                search_btn = gr.Button("🔍 Find Books", variant="primary", scale=1)
                reset_btn = gr.Button("🔄 Reset", variant="primary", scale=1)
                gr.Markdown(" ")

    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=2):
            out_gallery = gr.Gallery(
                label="Recommendations", 
                columns=4, 
                height="700px", 
                object_fit="contain", 
                allow_preview=False,
                show_label=False
            )
        with gr.Column(scale=1):
            details_output = gr.HTML(
                "<div style='text-align: center; padding-top: 100px; color: #94a3b8; border: 2px dashed #e2e8f0; border-radius: 12px; height: 300px;'>Select a book cover to view details.</div>"
            )

    empty_details = "<div style='text-align: center; padding-top: 100px; color: #94a3b8; border: 2px dashed #e2e8f0; border-radius: 12px; height: 300px;'>Select a book cover to view details.</div>"
    search_btn.click(fn=on_search, inputs=[user_query, cat_drop, tone_drop], outputs=[out_gallery, current_recs])
    search_btn.click(fn=lambda: empty_details, outputs=details_output)
    reset_btn.click(None, js="window.location.reload()")
    out_gallery.select(fn=get_book_details, inputs=[current_recs], outputs=details_output)

if __name__ == "__main__":
    dashboard.launch()
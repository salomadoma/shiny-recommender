from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# === Загрузка данных и модели ===
df_clean = pd.read_pickle("df_clean.pkl")
df_unlabeled = pd.read_pickle("df_unlabeled.pkl")

# Загрузка RuBERT
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruBert-base")
model = AutoModel.from_pretrained("sberbank-ai/ruBert-base")
model.eval()

def encode_query(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def find_best_semantic_story(query_text, emotion_filter, min_rating):
    query_vector = encode_query(query_text)
    df_filtered = df_clean.copy()
    if min_rating:
        df_filtered = df_filtered[df_filtered['rating'] >= min_rating]
    if emotion_filter:
        for emo, val in emotion_filter.items():
            df_filtered = df_filtered[df_filtered[emo] >= val]
    vectors = np.vstack(df_filtered['bert_vector'])
    sims = cosine_similarity([query_vector], vectors).flatten()
    return df_filtered.iloc[sims.argmax()]

def recommend_from_story(story_row, method="semantic", top_n=5):
    if method == "semantic":
        story_vec = np.array(story_row['bert_vector']).reshape(1, -1)
        vectors = np.vstack(df_unlabeled['bert_vector'])
    else:
        story_vec = np.array(story_row['hybrid_vector']).reshape(1, -1)
        vectors = np.vstack(df_unlabeled['hybrid_vector'])
    sims = cosine_similarity(story_vec, vectors).flatten()
    top_idx = sims.argsort()[::-1][:top_n]
    result = df_unlabeled.iloc[top_idx][['story_id', 'title', 'text']].copy()
    result['similarity'] = sims[top_idx]
    return result

# === Интерфейс ===
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_text("query", "Введите запрос:", placeholder="Например, война и разлука"),
        ui.input_slider("rating", "Минимальный рейтинг:", min=0, max=10, value=0, step=0.1),
        ui.input_slider("happiness", "Уровень счастья:", 0, 2, 0, step=0.1),
        ui.input_slider("sadness", "Уровень грусти:", 0, 2, 0, step=0.1),
        ui.input_slider("disgust", "Уровень отвращения:", 0, 2, 0, step=0.1),
        ui.input_slider("surprise", "Уровень удивления:", 0, 2, 0, step=0.1),
        ui.input_slider("anger", "Уровень гнева:", 0, 2, 0, step=0.1),
        ui.input_slider("fear", "Уровень страха:", 0, 2, 0, step=0.1),
        ui.input_radio_buttons("method", "Метод рекомендаций:", choices=["semantic", "hybrid"], selected="semantic"),
        ui.input_action_button("run", "Подобрать рекомендации")
    ),
    ui.output_ui("rec_table")
)

# === Серверная логика ===
def server(input, output, session):

    @reactive.event(input.run)
    def compute():
        emo_dict = {
            "happiness": input.happiness(),
            "sadness": input.sadness(),
            "disgust": input.disgust(),
            "surprise": input.surprise(),
            "anger": input.anger(),
            "fear": input.fear(),
        }
        emo_dict = {k: v for k, v in emo_dict.items() if v > 0}
        story = find_best_semantic_story(input.query(), emo_dict, input.rating())
        result = recommend_from_story(story, method=input.method())
        return result

    @output
    @render.ui
    def rec_table():
        df = compute()
        cards = []
        for i, row in df.iterrows():
            cards.append(
                ui.panel_well(
                    ui.tags.h4(f"{row['title']} (ID: {row['story_id']}) — similarity: {row['similarity']:.2f}"),
                    ui.tags.details(
                        ui.tags.summary("Показать/Скрыть текст рассказа"),
                        ui.tags.p(row['text'])
                    )
                )
            )
        return ui.div(*cards)

app = App(app_ui, server)

import math
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-embedding-2"
BATCH_SIZE = 32


@st.cache_data(show_spinner=False)
def _embed_batch(texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY was not found in the .env file.")

    client = genai.Client(api_key=API_KEY)

    contents = [
        types.Content(parts=[types.Part.from_text(text=text)])
        for text in texts
    ]

    result = client.models.embed_content(
        model=MODEL,
        contents=contents,
    )

    if not result.embeddings:
        raise RuntimeError("Gemini returned no embeddings.")

    vectors = []

    for embedding in result.embeddings:
        if not embedding.values:
            raise RuntimeError("Gemini returned an empty embedding.")
        vectors.append(tuple(embedding.values))

    return tuple(vectors)


def _similarity(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    dot_product = sum(a * b for a, b in zip(first, second))
    first_length = math.sqrt(sum(value * value for value in first))
    second_length = math.sqrt(sum(value * value for value in second))

    if first_length == 0 or second_length == 0:
        return 0.0

    return dot_product / (first_length * second_length)


def search_chunks(
    chunks: list[dict[str, int | str]],
    question: str,
    limit: int = 3,
) -> list[dict[str, int | str]]:
    if not question.strip() or not chunks:
        return []

    chunk_texts = tuple(str(chunk["text"]) for chunk in chunks)
    chunk_embeddings = []

    for start in range(0, len(chunk_texts), BATCH_SIZE):
        batch = chunk_texts[start : start + BATCH_SIZE]
        chunk_embeddings.extend(_embed_batch(batch))

    question_embedding = _embed_batch((question.strip(),))[0]

    scored_chunks = [
        (_similarity(question_embedding, chunk_embedding), chunk)
        for chunk_embedding, chunk in zip(chunk_embeddings, chunks)
    ]

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:limit]]
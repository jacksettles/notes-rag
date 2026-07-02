# app.py

from __future__ import annotations

from datetime import datetime, date
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, TOP_K
from rag_core import ReadingRAG


st.set_page_config(
    page_title="Local Reading RAG",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def load_rag(embedding_model_name: str, llm_model_name: str) -> ReadingRAG:
    return ReadingRAG(
        embedding_model_name=embedding_model_name,
        llm_model_name=llm_model_name,
    )


def format_metadata(metadata: dict) -> str:
    parts = []
    for key in ["author", "book", "page", "topic", "note_type"]:
        value = metadata.get(key, "")
        if value:
            parts.append(f"**{key.replace('_', ' ').title()}:** {value}")
    return " | ".join(parts) if parts else "_No metadata_"

def format_created_date(metadata: dict) -> str:
    created_at = metadata.get("created_at", "")
    if not created_at:
        return "Date added: unknown"

    try:
        dt = datetime.fromisoformat(created_at)
        return f"Date added: {dt.strftime('%B %-d, %Y')}"
    except Exception:
        return "Date added: unknown"
    
def get_unique_metadata_values(notes: list[dict], field: str) -> list[str]:
    values = set()

    for note in notes:
        metadata = note.get("metadata", {})
        value = metadata.get(field, "")
        if value:
            values.add(value)

    return sorted(values, key=str.lower)

def parse_created_at(metadata: dict) -> datetime | None:
    created_at = metadata.get("created_at", "")

    if not created_at:
        return None

    try:
        return datetime.fromisoformat(created_at)
    except Exception:
        return None


def filter_notes(
    notes: list[dict],
    selected_author: str,
    selected_book: str,
    date_range,
) -> list[dict]:
    start_date = None
    end_date = None

    if isinstance(date_range, tuple):
        if len(date_range) == 2:
            start_date, end_date = date_range
        elif len(date_range) == 1:
            start_date = date_range[0]
            end_date = date_range[0]
    elif date_range:
        start_date = date_range
        end_date = date_range

    filtered_notes = []

    for note in notes:
        metadata = note.get("metadata", {})

        author = metadata.get("author", "")
        book = metadata.get("book", "")

        if selected_author != "All authors" and author != selected_author:
            continue

        if selected_book != "All books / sources" and book != selected_book:
            continue

        created_dt = parse_created_at(metadata)
        created_date = created_dt.date() if created_dt else None

        if start_date or end_date:
            if created_date is None:
                continue

            if start_date and created_date < start_date:
                continue

            if end_date and created_date > end_date:
                continue

        filtered_notes.append(note)

    return filtered_notes

    
@st.dialog("Confirm delete")
def confirm_delete_note(note_id: str, note_preview: str):
    st.warning("Are you sure you want to delete this note?")
    st.write(note_preview)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Yes, delete", type="primary", key=f"confirm-delete-{note_id}"):
            rag.delete_note(note_id)
            st.success("Deleted note.")
            st.rerun()

    with col2:
        if st.button("Cancel", key=f"cancel-delete-{note_id}"):
            st.rerun()


st.title("📚 Local Reading RAG")
st.caption("A minimal local-first app for storing notes, embedding them, and chatting with a local LLM.")

with st.sidebar:
    st.header("Settings")

    embedding_model_name = st.text_input(
        "Embedding model",
        value=DEFAULT_EMBEDDING_MODEL,
        help="If you change this later, you should reset/rebuild the vector DB.",
    )

    llm_model_name = st.text_input(
        "Ollama chat model",
        value=DEFAULT_LLM_MODEL,
        help="Must match a model you have pulled with Ollama.",
    )

    top_k = st.slider(
        "Retrieved notes",
        min_value=1,
        max_value=10,
        value=TOP_K,
    )

    st.divider()

    rag = load_rag(embedding_model_name, llm_model_name)

    st.metric("Notes stored", rag.count_notes())

    with st.expander("Danger zone"):
        st.warning("This deletes all stored notes in the local Chroma collection.")
        if st.button("Reset vector database"):
            rag.reset_collection()
            st.success("Vector database reset.")
            st.rerun()


tab_add, tab_chat, tab_search, tab_library, tab_map = st.tabs(
    ["Add Note", "Chat", "Search", "Library", "Similarity Map"]
)

with tab_add:
    st.subheader("Add a quote, passage, or note")

    existing_notes = rag.list_notes(limit=1000)
    existing_authors = get_unique_metadata_values(existing_notes, "author")
    existing_books = get_unique_metadata_values(existing_notes, "book")

    with st.form("add_note_form", clear_on_submit=True):
        text = st.text_area(
            "Text",
            height=220,
            placeholder="Paste a quote, passage, or your own reading note here...",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            author = st.selectbox(
                "Author",
                options=existing_authors,
                index=None,
                placeholder="Type or select an author...",
                accept_new_options=True,
            )

            book = st.selectbox(
                "Book / Source",
                options=existing_books,
                index=None,
                placeholder="Type or select a book/source...",
                accept_new_options=True,
            )

        with col2:
            page = st.text_input("Page", placeholder="42")
            topic = st.text_input("Topic", placeholder="absurdism, rationalism")

        with col3:
            note_type = st.selectbox(
                "Type",
                options=["", "quote", "summary", "reflection", "question", "other"],
            )

        submitted = st.form_submit_button("Embed and save")

    if submitted:
        try:
            note_id = rag.add_note(
                text=text,
                author=author or "",
                book=book or "",
                page=page,
                topic=topic,
                note_type=note_type,
            )
            st.success(f"Saved note: {note_id}")
            st.rerun()
        except Exception as e:
            st.error(f"Could not save note: {e}")


with tab_chat:
    st.subheader("Chat with your reading notes")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about your notes...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving notes and asking local model..."):
                try:
                    result = rag.chat(
                        user_question=question,
                        top_k=top_k,
                        llm_model_name=llm_model_name,
                    )

                    answer = result["answer"]
                    retrieved_notes = result["retrieved_notes"]

                    st.markdown(answer)

                    with st.expander("Retrieved context"):
                        for i, item in enumerate(retrieved_notes, start=1):
                            st.markdown(f"### Note {i}")
                            st.markdown(format_metadata(item["metadata"]))
                            st.write(item["text"])
                            if item.get("similarity") is not None:
                                st.caption(f"Similarity: {item['similarity']:.3f}")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    st.error(f"Chat failed: {e}")


with tab_search:
    st.subheader("Semantic search")

    search_query = st.text_input(
        "Search your notes",
        placeholder="Example: rationalism, absurdism, ethics, language games...",
    )

    if search_query:
        try:
            results = rag.search(search_query, top_k=top_k)

            if not results:
                st.info("No results found.")

            for i, item in enumerate(results, start=1):
                with st.container(border=True):
                    st.markdown(f"### Result {i}")
                    st.markdown(format_metadata(item["metadata"]))
                    st.write(item["text"])

                    if item.get("similarity") is not None:
                        st.caption(f"Similarity: {item['similarity']:.3f}")

        except Exception as e:
            st.error(f"Search failed: {e}")


with tab_library:
    st.subheader("Stored notes")

    all_notes = rag.list_notes(limit=500)

    all_notes = sorted(
        all_notes,
        key=lambda item: item["metadata"].get("created_at", ""),
        reverse=True,
    )

    authors = get_unique_metadata_values(all_notes, "author")
    books = get_unique_metadata_values(all_notes, "book")

    with st.expander("Filters", expanded=True):
        if st.button("Clear filters", key="clear_library_filters"):
            st.session_state.library_author_filter = "All authors"
            st.session_state.library_book_filter = "All books / sources"
            st.session_state.library_date_filter = ()
            st.rerun()

        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            selected_author = st.selectbox(
                "Author",
                options=["All authors"] + authors,
                key="library_author_filter",
            )

        with filter_col2:
            selected_book = st.selectbox(
                "Book / Source",
                options=["All books / sources"] + books,
                key="library_book_filter",
            )

        with filter_col3:
            date_range = st.date_input(
                "Date added",
                value=(),
                help="Filter by the date notes were added.",
                key="library_date_filter",
            )

    notes = filter_notes(
        notes=all_notes,
        selected_author=selected_author,
        selected_book=selected_book,
        date_range=date_range,
    )

    st.caption(
        f"Showing {len(notes)} of {len(all_notes)} stored note"
        f"{'s' if len(all_notes) != 1 else ''}."
    )

    if "editing_note_id" not in st.session_state:
        st.session_state.editing_note_id = None

    if not all_notes:
        st.info("No notes stored yet.")
    elif not notes:
        st.info("No notes match the current filters.")
    else:
        for item in notes:
            note_id = item["id"]
            metadata = item["metadata"]
            is_editing = st.session_state.editing_note_id == note_id

            with st.container(border=True):
                st.caption(format_created_date(item["metadata"]))
                st.markdown(f":blue[{format_metadata(item["metadata"])}]")
                st.write(item["text"])

                col1, col2, col3 = st.columns([5,1,1])
                with col1:
                    st.caption(f"ID: {item['id']}")
                with col2:
                    if st.button("Edit Note", key=f"edit-{item['id']}"):
                        st.session_state.editing_note_id = note_id
                        st.rerun()
                with col3:
                    if st.button("Delete", key=f"delete-{item['id']}"):
                        preview = item["text"][:300]
                        if len(item["text"]) > 300:
                            preview += "..."

                        confirm_delete_note(
                            note_id=item["id"],
                            note_preview=preview,
                        )
                
                if is_editing:
                    st.divider()

                    new_text = st.text_area(
                        "Note text",
                        value=item["text"],
                        key=f"text-{note_id}",
                        height=150,
                    )

                    new_author = st.text_input(
                        "Author",
                        value=metadata.get("author", ""),
                        key=f"author-{note_id}",
                    )

                    new_book = st.text_input(
                        "Book",
                        value=metadata.get("book", ""),
                        key=f"book-{note_id}",
                    )

                    new_page = st.text_input(
                        "Page",
                        value=metadata.get("page", ""),
                        key=f"page-{note_id}",
                    )

                    new_topic = st.text_input(
                        "Topic",
                        value=metadata.get("topic", ""),
                        key=f"topic-{note_id}",
                    )

                    new_note_type = st.text_input(
                        "Note type",
                        value=metadata.get("note_type", ""),
                        key=f"note-type-{note_id}",
                    )

                    save_col, cancel_col = st.columns([1, 1])
                    with save_col:
                        if st.button("Save changes", key=f"save-{note_id}"):
                            rag.edit_note(
                                note_id=note_id,
                                text=new_text,
                                author=new_author,
                                book=new_book,
                                page=new_page,
                                topic=new_topic,
                                note_type=new_note_type,
                            )
                            st.session_state.editing_note_id = None
                            st.success("Edited note.")
                            st.rerun()

                    with cancel_col:
                        if st.button("Cancel", key=f"cancel-{note_id}"):
                            st.session_state.editing_note_id = None
                            st.rerun()
with tab_map:
    st.subheader("2D similarity map")

    notes = rag.list_notes(limit=1000)

    if len(notes) < 2:
        st.info("Add at least two notes to visualize similarity.")
    else:
        map_level = st.radio(
            "Map level",
            ["Individual notes", "Author bundles"],
            horizontal=True,
        )

        reduction_method = st.selectbox(
            "Dimensionality reduction method",
            ["PCA", "UMAP"],
        )

        show_clusters = st.checkbox("Color by hierarchical clusters", value=False)

        if show_clusters:
            max_clusters = min(10, len(notes))
            n_clusters = st.slider(
                "Number of clusters",
                min_value=2,
                max_value=max_clusters,
                value=min(4, max_clusters),
            )
        else:
            n_clusters = None

        try:
            rows = []

            if map_level == "Individual notes":
                texts = [note["text"] for note in notes]
                embeddings = [rag.embed_text(text) for text in texts]

                source_items = notes

            else:
                author_groups = {}

                for note in notes:
                    meta = note["metadata"]
                    author = meta.get("author", "").strip() or "Unknown author"

                    if author not in author_groups:
                        author_groups[author] = {
                            "texts": [],
                            "books": set(),
                            "topics": set(),
                            "note_count": 0,
                        }

                    author_groups[author]["texts"].append(note["text"])
                    author_groups[author]["note_count"] += 1

                    book = meta.get("book", "").strip()
                    topic = meta.get("topic", "").strip()

                    if book:
                        author_groups[author]["books"].add(book)
                    if topic:
                        author_groups[author]["topics"].add(topic)

                if len(author_groups) < 2:
                    st.info("Add notes from at least two authors to visualize author bundles.")
                    st.stop()

                source_items = []

                for author, group in author_groups.items():
                    combined_text = "\n\n".join(group["texts"])

                    source_items.append(
                        {
                            "text": combined_text,
                            "metadata": {
                                "author": author,
                                "book": ", ".join(sorted(group["books"])),
                                "topic": ", ".join(sorted(group["topics"])),
                                "note_count": group["note_count"],
                            },
                        }
                    )

                texts = [item["text"] for item in source_items]

                with st.spinner("Embedding author bundles..."):
                    embeddings = [rag.embed_text(text) for text in texts]

            if len(embeddings) < 2:
                st.info("Add at least two items to visualize similarity.")
                st.stop()

            if reduction_method == "PCA":
                reducer = PCA(n_components=2)
                coords = reducer.fit_transform(embeddings)

            elif reduction_method == "UMAP":
                try:
                    import umap
                except ImportError:
                    st.error("UMAP is not installed. Run: pip install umap-learn")
                    st.stop()

                n_neighbors = min(15, max(2, len(embeddings) - 1))

                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric="cosine",
                    random_state=42,
                )

                coords = reducer.fit_transform(embeddings)

            for item, coord in zip(source_items, coords):
                meta = item["metadata"]

                label_parts = [
                    meta.get("author", ""),
                    meta.get("book", ""),
                    meta.get("topic", ""),
                ]

                label = " | ".join([part for part in label_parts if part])

                if not label:
                    label = item["text"][:60]

                row = {
                    "x": coord[0],
                    "y": coord[1],
                    "label": label,
                    "text": item["text"],
                    "author": meta.get("author", ""),
                    "book": meta.get("book", ""),
                    "topic": meta.get("topic", ""),
                }

                if map_level == "Author bundles":
                    row["note_count"] = meta.get("note_count", 0)

                rows.append(row)

            df = pd.DataFrame(rows)

            if show_clusters:
                from sklearn.cluster import AgglomerativeClustering

                cluster_count = min(n_clusters, len(df))

                clustering = AgglomerativeClustering(
                    n_clusters=cluster_count,
                    metric="cosine",
                    linkage="average",
                )

                cluster_labels = clustering.fit_predict(embeddings)
                df["cluster"] = [
                    f"Cluster {label + 1}" for label in cluster_labels
                ]

                color_col = "cluster"
            else:
                color_col = "author" if "author" in df.columns else None

            hover_data = {
                "text": True,
                "author": True,
                "book": True,
                "topic": True,
                "x": False,
                "y": False,
            }

            if map_level == "Author bundles":
                hover_data["note_count"] = True

            fig = px.scatter(
                df,
                x="x",
                y="y",
                color=color_col,
                hover_name="label",
                hover_data=hover_data,
                title=f"{reduction_method} projection of {map_level.lower()} embeddings",
            )

            fig.update_traces(marker=dict(size=10, opacity=0.8))

            fig.update_layout(
                height=650,
                xaxis_title=f"{reduction_method} 1",
                yaxis_title=f"{reduction_method} 2",
            )

            st.plotly_chart(fig, width="stretch")

            if map_level == "Individual notes":
                map_description = "Each point is one note."
            else:
                map_description = (
                    "Each point is one author. All notes by that author are combined "
                    "and re-embedded before dimensionality reduction."
                )

            st.caption(
                f"{map_description} This is a rough 2D {reduction_method} projection. "
                "Nearby points may be semantically related, but treat this as an exploratory view, "
                "not a rigorous semantic map."
            )

        except Exception as e:
            st.error(f"Could not create similarity map: {e}")
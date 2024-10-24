import pandas as pd
from langchain.text_splitter import MarkdownTextSplitter
import hashlib
import os
import requests
from psycopg2.extras import Json
import numpy as np
import torch


def convert_pdf_markdown(url, file_path):
    with open(file_path, "rb") as pdf_file:
        files = {"pdf_file": pdf_file}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        result = response.json()
        return result["full_text"], result["metadata"]
    else:
        raise Exception(f"Error converting PDF: {response.text}")


def retrieve_relevant_documents(embeddings, cursor, query, top_k=3):
    query_vector = embeddings.embed_query(query)

    cursor.execute("SELECT id, embedding, meta FROM max_kb_embedding")
    all_embeddings = cursor.fetchall()

    doc_ids = []
    doc_embeddings = []
    meta_data = []

    for row in all_embeddings:
        doc_id = row[0]
        embedding_str = row[1].replace("[", "").replace("]", "")
        embedding = np.array([float(x) for x in embedding_str.split(",")])
        doc_ids.append(doc_id)
        doc_embeddings.append(embedding)
        meta_data.append(row[2])

    doc_embeddings = np.array(doc_embeddings)

    similarities = torch.cosine_similarity(torch.tensor([query_vector]), torch.tensor(doc_embeddings)).numpy()
    if isinstance(similarities, np.float64):
        similarities = np.array([similarities])

    top_k = min(len(similarities), top_k)
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    relevant_docs = [(doc_ids[i], similarities[i], meta_data[i]) for i in top_k_indices]

    return relevant_docs


def split_markdown_words(url, pdf_file):
    # Step 1: Convert PDF to Markdown using external service
    file_path = pdf_file.name
    full_text, out_metadata = convert_pdf_markdown(url, file_path)

    # Step 2: Save the Markdown content locally
    markdown_file_name = os.path.splitext(os.path.basename(file_path))[0] + ".md"
    markdown_file_path = os.path.join(os.path.dirname(file_path), markdown_file_name)
    with open(markdown_file_path, "w", encoding="utf-8") as md_file:
        md_file.write(full_text)

    # Step 3: Split the Markdown content into segments, using the MarkdownHeaderTextSplitter

    # text_splitter = MarkdownHeaderTextSplitter(
    #     headers_to_split_on=[
    #         ("#", "Header 1"),
    #         ("##", "Header 2"),
    #         ("###", "Header 3"),
    #     ]
    # )
    text_splitter = MarkdownTextSplitter()
    docs = text_splitter.split_text(full_text)

    # Step 4: Return the split segments for display
    data = [
        {"Segment": f"Segment {i + 1}", "Word Count": len(doc.split()), "Preview": " ".join(doc.split())}
        for i, doc in enumerate(docs)
    ]

    return pd.DataFrame(data), markdown_file_path


# Calculate MD5 hash for file deduplication
def calculate_md5(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# Delete local file
def delete_local_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted local file: {file_path}")


def upload_to_oss(bucket, file_path, object_name):
    oss_object_name = f"zhaokai/{object_name}"
    bucket.put_object_from_file(oss_object_name, file_path)
    print(f"File {file_path} uploaded to OSS as {oss_object_name}")
    return oss_object_name


def delete_file_action(conn, cursor, bucket, file_name):
    cursor.execute("SELECT id FROM max_kb_file WHERE filename = %s", (file_name,))
    file_id = cursor.fetchone()
    if file_id:
        file_id = file_id[0]
        cursor.execute(
            "DELETE FROM max_kb_embedding WHERE source_id IN "
            "(SELECT id FROM max_kb_paragraph WHERE document_id IN "
            "(SELECT id FROM max_kb_document WHERE files::jsonb @> %s::jsonb))",
            (Json([file_id]),),
        )
        cursor.execute(
            "DELETE FROM max_kb_paragraph WHERE document_id IN "
            "(SELECT id FROM max_kb_document WHERE files::jsonb @> %s::jsonb)",
            (Json([file_id]),),
        )
        cursor.execute("DELETE FROM max_kb_document WHERE files::jsonb @> %s::jsonb", (Json([file_id]),))
        cursor.execute("DELETE FROM max_kb_file WHERE id = %s", (file_id,))

        # Delete file from OSS
        oss_object_name = f"zhaokai/{file_name}.md"
        bucket.delete_object(oss_object_name)
        print(f"File {oss_object_name} deleted from OSS.")

        conn.commit()
        return f"File '{file_name}' and related data have been deleted."
    else:
        return "File not found."


def chat_with_model_rag(embeddings, cursor, client, prompt, history, llm_model="none", temperature=0.0, top_k=3):
    relevant_docs = retrieve_relevant_documents(embeddings, cursor, prompt, top_k=top_k)

    context_texts = []
    for i, doc in enumerate(relevant_docs):
        paragraph_id = doc[2]["paragraph_id"]
        cursor.execute("SELECT content FROM max_kb_paragraph WHERE id = %s", (int(paragraph_id),))
        paragraph = cursor.fetchone()
        if paragraph:
            context_texts.append(f"Document {i + 1}: {paragraph[0]}")

    related_docs_df = pd.DataFrame(
        {"Document ID": [f"Document {i + 1}" for i in range(len(context_texts))], "Content": context_texts}
    )

    system_message = {
        "role": "system",
        "content": f"You are a task completion expert. Based on the provided history and the relevant documents"
        f"below (which may not be helpful), answer the question.\n"
        f"Relevant Documents:\n{context_texts}\n"
        f"Let's begin the conversation!",
    }
    history_messages = [system_message]
    for idx, (user_msg, model_msg) in enumerate(history):
        if user_msg:
            history_messages.append({"role": "user", "content": user_msg})
        if model_msg:
            history_messages.append({"role": "assistant", "content": model_msg})

    history_messages.append({"role": "user", "content": prompt})
    history.append([prompt, None])
    response = client.chat.completions.create(
        model=llm_model, messages=history_messages, temperature=temperature, top_p=0.1
    )

    output = response.choices[0].message.content.strip()

    history[-1][1] = output

    return history, related_docs_df

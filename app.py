import gradio as gr
from langchain_openai import OpenAIEmbeddings
import psycopg2
from psycopg2.extras import execute_values, Json
import datetime
import os
from openai import OpenAI
import pandas as pd
import threading
import time
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from utils import calculate_md5, split_markdown_words, upload_to_oss, delete_file_action, chat_with_model_rag

conn = None
cursor = None
embeddings = None
snowflake = None
client = None
bucket = None

default_api_key = "your api_keys"
default_api_url = "https://open.bigmodel.cn/api/paas/v4"
default_llm_model = "glm-4-airx"
default_embedding_model = "embedding-3"
flask_pdf_converter_url = "http://localhost:5001/convert_pdf"

os.environ["OSS_ACCESS_KEY_ID"] = "your access key id"
os.environ["OSS_ACCESS_KEY_SECRET"] = "your access key secret"
bucket_name = "zrzrzr"
endpoint = "oss-cn-beijing.aliyuncs.com"


# Initialize the database connection, models, and OSS
def initialize(api_key=default_api_key, api_url=default_api_url, embedding_model=default_embedding_model):
    global conn, cursor, embeddings, snowflake, client, bucket

    PG_HOST = "localhost"
    PG_PORT = "54320"
    PG_USER = "postgres"
    PG_PASSWORD = "dianjiao29"  # Replace with your PostgreSQL password
    PG_DATABASE = "zrkb"

    conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, database=PG_DATABASE)
    cursor = conn.cursor()

    client = OpenAI(api_key=api_key, base_url=api_url)

    embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_base=api_url, openai_api_key=api_key)

    class SnowflakeGenerator:
        # Snowflake ID generator for unique identifiers
        EPOCH = 1288834974657
        WORKER_ID_BITS = 5
        DATA_CENTER_ID_BITS = 5
        SEQUENCE_BITS = 12
        MAX_WORKER_ID = -1 ^ (-1 << WORKER_ID_BITS)
        MAX_DATA_CENTER_ID = -1 ^ (-1 << SEQUENCE_BITS)
        MAX_SEQUENCE = -1 ^ (-1 << SEQUENCE_BITS)
        WORKER_ID_SHIFT = SEQUENCE_BITS
        DATA_CENTER_ID_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS
        TIMESTAMP_LEFT_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS + DATA_CENTER_ID_BITS

        def __init__(self, worker_id, data_center_id):
            self.worker_id = worker_id
            self.data_center_id = data_center_id
            self.sequence = 0
            self.last_timestamp = -1
            self.lock = threading.Lock()

        def _time_gen(self):
            return int(time.time() * 1000)

        def _til_next_millis(self, last_timestamp):
            timestamp = self._time_gen()
            while timestamp <= last_timestamp:
                timestamp = self._time_gen()
            return timestamp

        def get_id(self):
            with self.lock:
                timestamp = self._time_gen()
                if timestamp < self.last_timestamp:
                    raise Exception("Clock moved backwards.")
                if self.last_timestamp == timestamp:
                    self.sequence = (self.sequence + 1) & self.MAX_SEQUENCE
                    if self.sequence == 0:
                        timestamp = self._til_next_millis(self.last_timestamp)
                else:
                    self.sequence = 0
                self.last_timestamp = timestamp
                new_id = (
                    ((timestamp - self.EPOCH) << self.TIMESTAMP_LEFT_SHIFT)
                    | (self.data_center_id << self.DATA_CENTER_ID_SHIFT)
                    | (self.worker_id << self.WORKER_ID_SHIFT)
                    | self.sequence
                )
                return new_id

    snowflake = SnowflakeGenerator(worker_id=1, data_center_id=1)

    # Initialize OSS
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, endpoint, bucket_name)


def vectorize_markdown(embeddings, cursor, snowflake, conn, bucket, docs, markdown_file_path):
    results = []

    file_size = os.path.getsize(markdown_file_path)
    file_md5 = calculate_md5(markdown_file_path)
    file_name = os.path.basename(markdown_file_path)

    cursor.execute("SELECT 1 FROM max_kb_file WHERE md5 = %s OR filename = %s", (file_md5, file_name))
    if cursor.fetchone() is not None:
        return f"The file '{file_name}' already exists, skipping vectorization."

    # Step 1: Upload Markdown to OSS
    oss_object_name = upload_to_oss(bucket, markdown_file_path, file_name)

    # Step 2: Directly use the already split docs for vectorization
    vectors = embeddings.embed_documents(docs)

    # Step 3: Record file information in the database
    file_id = snowflake.get_id()
    file_record = (
        file_id,
        file_md5,
        file_name,
        file_size,
        "user_id",
        "aliyun",
        "oss-cn-beijing",
        bucket.bucket_name,
        oss_object_name,
        f"zhaokai/{file_name}",
        Json({}),
        "",
        datetime.datetime.now(),
        "",
        datetime.datetime.now(),
        0,
        0,
    )
    insert_query = """
            INSERT INTO max_kb_file (
                id, md5, filename, file_size, user_id, platform, region_name,
                bucket_name, file_id, target_name, tags, creator, create_time,
                updater, update_time, deleted, tenant_id
            ) VALUES %s
        """
    execute_values(cursor, insert_query, [file_record])

    # **Create dataset if needed (assuming a default dataset)**
    dataset_id = snowflake.get_id()
    dataset_record = (
        dataset_id,
        "Default Dataset",
        "Markdown Dataset",
        "markdown",
        Json({}),
        "user_id",
        "remark",
        "",
        datetime.datetime.now(),
        "",
        datetime.datetime.now(),
        0,
        0,
    )
    insert_query = """
            INSERT INTO max_kb_dataset (
                id, name, description, type, meta, user_id, remark,
                creator, create_time, updater, update_time, deleted, tenant_id
            ) VALUES %s
        """
    execute_values(cursor, insert_query, [dataset_record])

    # Step 4: Create document and embeddings
    document_id = snowflake.get_id()
    document_record = (
        document_id,
        file_name,
        len(docs),
        "active",
        True,
        "markdown",
        Json({}),
        dataset_id,
        "method",
        0.0,
        Json([file_id]),
        "",
        datetime.datetime.now(),
        "",
        datetime.datetime.now(),
        0,
        0,
    )
    insert_query = """
            INSERT INTO max_kb_document (
                id, name, char_length, status, is_active, type, meta,
                dataset_id, hit_handling_method, directly_return_similarity,
                files, creator, create_time, updater, update_time, deleted, tenant_id
            ) VALUES %s
        """
    execute_values(cursor, insert_query, [document_record])

    data_paragraphs = []
    data_embeddings = []

    for i, (text, vector) in enumerate(zip(docs, vectors)):
        paragraph_id = snowflake.get_id()
        paragraph_record = (
            paragraph_id,
            text,
            "Title",
            "active",
            0,
            True,
            dataset_id,
            document_id,
            "",
            datetime.datetime.now(),
            "",
            datetime.datetime.now(),
            0,
            0,
        )
        data_paragraphs.append(paragraph_record)

        embedding_id = snowflake.get_id()
        vector_str = "[" + ",".join(map(str, vector)) + "]"
        meta = {"document_id": str(document_id), "paragraph_id": str(paragraph_id)}
        embedding_record = (
            embedding_id,
            paragraph_id,
            "paragraph",
            True,
            vector_str,
            Json(meta),
            dataset_id,
            document_id,
            paragraph_id,
            "",
            "",
            datetime.datetime.now(),
            "",
            datetime.datetime.now(),
            0,
            0,
        )
        data_embeddings.append(embedding_record)

    insert_query = """
            INSERT INTO max_kb_paragraph (
                id, content, title, status, hit_num, is_active,
                dataset_id, document_id, creator, create_time,
                updater, update_time, deleted, tenant_id
            ) VALUES %s
        """
    execute_values(cursor, insert_query, data_paragraphs)

    insert_query = """
            INSERT INTO max_kb_embedding (
                id, source_id, source_type, is_active, embedding, meta,
                dataset_id, document_id, paragraph_id, search_vector,
                creator, create_time, updater, update_time, deleted, tenant_id
            ) VALUES %s
        """
    execute_values(cursor, insert_query, data_embeddings)

    conn.commit()
    results.append(f"File '{file_name}' has been processed and saved.")

    return "\n".join(results)


def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("""
                   <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                       RAG Demo built with Gradio 🤗
                   </div>
                   """)

        # LLM Settings Section
        with gr.Accordion("LLM Settings", open=False):
            gr.Markdown("### LLM Settings")
            api_key_input = gr.Textbox(label="API Key", value=default_api_key, placeholder="Enter API key")
            api_url_input = gr.Textbox(label="API URL", value=default_api_url, placeholder="Enter API base URL")
            llm_model_input = gr.Textbox(
                label="LLM Model", value=default_llm_model, placeholder="Enter LLM model name"
            )
            embedding_model_input = gr.Textbox(
                label="Embedding Model", value=default_embedding_model, placeholder="Embedding Model"
            )
            temperature_slider = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.1, step=0.05)
            top_k_slider = gr.Slider(label="Top K Similarity", minimum=1, maximum=10, value=2, step=1)
            apply_settings_button = gr.Button("Apply Settings")
            settings_output = gr.Textbox(label="Settings Status", visible=False, lines=2)

            # Settings update handler
            def update_settings(api_key, api_url, llm_model, embedding_model, temperature, top_k):
                initialize(api_key=api_key, api_url=api_url, embedding_model=embedding_model)
                return (
                    f"Settings applied. LLM Model: {llm_model}, Embedding Model: {embedding_model}, "
                    f"Temperature: {temperature}, Top K: {top_k}"
                )

            apply_settings_button.click(
                fn=update_settings,
                inputs=[
                    api_key_input,
                    api_url_input,
                    llm_model_input,
                    embedding_model_input,
                    temperature_slider,
                    top_k_slider,
                ],
                outputs=[settings_output],
            )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### File Operations")
                pdf_input = gr.File(label="Upload PDF(s)", file_count="multiple")
                split_button = gr.Button("Split Text")
                vectorize_button = gr.Button("Vectorize")

                split_result = gr.DataFrame(headers=["File", "Segment Count", "Word Count"], visible=True)
                vectorize_result = gr.DataFrame(headers=["File", "Vectorization Result"], visible=True)

                markdown_file_paths = gr.State([])

                def handle_split_text(pdf_files):
                    results = []
                    markdown_paths = []
                    for pdf_file in pdf_files:
                        df, markdown_path = split_markdown_words(flask_pdf_converter_url, pdf_file)
                        results.append(df)
                        markdown_paths.append((markdown_path, df["Preview"].tolist()))
                    markdown_file_paths.value = markdown_paths
                    return pd.concat(results)

                # Handling vectorization, storing results in the DataFrame
                def handle_vectorize():
                    vectorize_results = []
                    for markdown_path, docs in markdown_file_paths.value:
                        result = vectorize_markdown(embeddings, cursor, snowflake, conn, bucket, docs, markdown_path)
                        vectorize_results.append(result)
                    results_df = pd.DataFrame(
                        {
                            "File": [os.path.basename(md_path) for md_path, _ in markdown_file_paths.value],
                            "Vectorization Result": vectorize_results,
                        }
                    )
                    return results_df

                # Clicking Split Button to start text splitting
                split_button.click(fn=handle_split_text, inputs=pdf_input, outputs=split_result)

                # Clicking Vectorize Button to start vectorization
                vectorize_button.click(fn=handle_vectorize, outputs=vectorize_result)

                gr.Markdown("### File List and Deletion")
                file_selection = gr.Dropdown(label="Select Files to Delete", multiselect=True, choices=[])
                refresh_button = gr.Button("Refresh File List")
                delete_button = gr.Button("Delete Selected Files")
                file_deletion_result = gr.DataFrame(headers=["File", "Result"], label="Deletion Result", visible=True)

                # List files function for dropdown
                def update_file_selection():
                    cursor.execute("SELECT filename FROM max_kb_file")
                    files = cursor.fetchall()
                    file_list = [file[0] for file in files]
                    return file_list

                # Refresh button updates the dropdown list
                refresh_button.click(fn=update_file_selection, outputs=file_selection)

                # Batch deletion handler
                def handle_file_deletion(selected_files):
                    results = []
                    for file_name in selected_files:
                        result = delete_file_action(conn, cursor, bucket, file_name)
                        results.append({"File": file_name, "Result": result})
                    return pd.DataFrame(results)

                # Clicking Delete button to delete selected files
                delete_button.click(fn=handle_file_deletion, inputs=file_selection, outputs=file_deletion_result)

            with gr.Column():
                gr.Markdown("### Model Chat")
                chat_input = gr.Textbox(label="Enter your question", placeholder="Enter your question here")
                chat_output = gr.Chatbot(label="LLM Response")
                related_docs = gr.DataFrame(label="Relevant Documents", col_count=3, visible=True)
                chat_button = gr.Button("Send")
                conversation_state = gr.State([])
                clear_chat_button = gr.Button("Clear Chat")

                def chat(user_input, history, api_key, api_url, llm_model, embedding_model, temperature, top_k):
                    history = history or []
                    initialize(api_key=api_key, api_url=api_url, embedding_model=embedding_model)
                    updated_history, related_docs_df = chat_with_model_rag(
                        embeddings, cursor, client, user_input, history, llm_model, temperature, top_k
                    )
                    return updated_history, updated_history, related_docs_df

                chat_button.click(
                    fn=chat,
                    inputs=[
                        chat_input,
                        conversation_state,
                        api_key_input,
                        api_url_input,
                        llm_model_input,
                        embedding_model_input,
                        temperature_slider,
                        top_k_slider,
                    ],
                    outputs=[chat_output, conversation_state, related_docs],
                )
                clear_chat_button.click(fn=lambda: ([], []), outputs=[chat_output, conversation_state])

    return demo


if __name__ == "__main__":
    initialize()
    demo = gradio_interface()
    demo.launch()

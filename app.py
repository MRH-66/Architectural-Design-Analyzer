# ArchitecturalRAGSystem/rag_ui_app.py
import streamlit as st
import os
import json
import time
from typing import Dict, Any, Optional

# Import RAG components
try:
    from src.config import Config
    from src.vector_store.chroma_manager import ChromaManager
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.config import Config  # type: ignore
    from src.vector_store.chroma_manager import ChromaManager  # type: ignore

# --- Streamlit UI ---
st.set_page_config(page_title="Architectural Design Analyzer",
                   layout="wide")  # Changed title slightly
st.title("AI Architectural Design Analyzer 📐✨")
st.write(
    "Upload a conversation JSON detailing your design needs (typically from our AI Design Assistant chatbot) "
    "to receive a comprehensive architectural analysis, relevant standards, and preliminary Bill of Quantities."
)

# --- Load RAG System Config ---
rag_system_config: Optional[Config] = None
try:
    rag_system_config = Config()
    os.makedirs(rag_system_config.OUTPUT_JSON_PATH, exist_ok=True)
except Exception as e:
    st.error(f"Error loading system configuration: {e}")
    st.error("Please ensure the application is correctly set up.")
    st.stop()

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Upload your Conversation JSON file",
    type=["json"],
    key="conversation_uploader"
)

# Initialize session state for storing results
if 'final_rag_result' not in st.session_state:
    st.session_state.final_rag_result = None
if 'uploaded_file_name_for_download' not in st.session_state:
    st.session_state.uploaded_file_name_for_download = None
if 'processed_uploaded_filename' not in st.session_state:  # To track what was last processed
    st.session_state.processed_uploaded_filename = None


if uploaded_file is not None:
    try:
        conversation_data_bytes = uploaded_file.getvalue()
        conversation_data_string = conversation_data_bytes.decode("utf-8")
        user_conversation_data = json.loads(conversation_data_string)

        # Only show success and update filenames if it's a new file or different from last processed
        if uploaded_file.name != st.session_state.get('processed_uploaded_filename_display'):
            st.success(
                f"File '{uploaded_file.name}' loaded. Ready for processing.")
            st.session_state.final_rag_result = None
            st.session_state.uploaded_file_name_for_download = uploaded_file.name
            st.session_state.processed_uploaded_filename_display = uploaded_file.name

        with st.expander("View Uploaded Conversation Summary (First 5 Messages)"):
            messages_to_display = None
            if isinstance(user_conversation_data, dict) and "messages" in user_conversation_data:
                messages_to_display = user_conversation_data["messages"][:5]
            elif isinstance(user_conversation_data, list):
                messages_to_display = user_conversation_data[:5]

            if messages_to_display:
                st.json(messages_to_display)
            else:
                st.write(
                    "Could not display message snippet from the uploaded file.")

        if st.button("Process Design Brief", key="process_rag_btn"):  # Changed button label
            st.session_state.processed_uploaded_filename = uploaded_file.name
            st.session_state.final_rag_result = None

            temp_dir = "temp_uploads_rag_ui"
            os.makedirs(temp_dir, exist_ok=True)
            temp_conversation_filename = f"temp_convo_{int(time.time())}.json"
            temp_conversation_path = os.path.join(
                temp_dir, temp_conversation_filename)

            with open(temp_conversation_path, 'w', encoding='utf-8') as f:
                f.write(conversation_data_string)

            # REMOVED: st.info(f"Processing: Temporarily using uploaded conversation from: {temp_conversation_path}")

            with st.spinner("Analyzing your design brief and consulting architectural standards... This may take a few minutes."):
                try:
                    # Dynamically import run_full_rag_pipeline here
                    from run_query_service import run_full_rag_pipeline

                    result = run_full_rag_pipeline(
                        conversation_json_path=temp_conversation_path,
                        output_dir=rag_system_config.OUTPUT_JSON_PATH
                    )
                    st.session_state.final_rag_result = result

                    if result:
                        st.balloons()  # Fun success indicator!
                        st.success("Analysis Complete!")
                    else:
                        st.error(
                            "Processing completed but failed to produce an output. Please check console logs if running locally, or contact support.")

                except ImportError as ie:
                    st.error(
                        f"ImportError: {ie}. A required system component ('run_query_service') is missing or not configured correctly.")
                except Exception as e_pipeline:
                    st.error(
                        f"An error occurred during the analysis pipeline: {e_pipeline}")
                    import traceback
                    # Only show traceback in a dev mode or if explicitly enabled
                    # st.text_area("Pipeline Traceback", traceback.format_exc(), height=300)
                    # Log to console
                    print(f"Pipeline Traceback: {traceback.format_exc()}")

            if os.path.exists(temp_conversation_path):
                try:
                    os.remove(temp_conversation_path)
                except Exception as e_del:
                    # Log to console
                    print(
                        f"Warning: Could not delete temporary file {temp_conversation_path}: {e_del}")

            st.rerun()

    except json.JSONDecodeError:
        st.error(
            "Invalid JSON file. Please upload a correctly formatted conversation JSON.")
        st.session_state.final_rag_result = None
    except Exception as e:
        st.error(f"An error occurred while handling the uploaded file: {e}")
        st.session_state.final_rag_result = None

# Display results if available
if st.session_state.final_rag_result:
    st.subheader("Architectural Analysis & Standards Output:")
    st.json(st.session_state.final_rag_result)
elif uploaded_file and not st.session_state.final_rag_result and st.session_state.get('processed_uploaded_filename') == uploaded_file.name:
    # This case means processing was triggered but result might be None due to an error caught above
    st.info("Processing was initiated. If no output appears above, an error may have occurred during analysis.")
elif not uploaded_file:
    st.info(
        "Please upload a conversation JSON file to begin your architectural analysis.")


# --- Sidebar Content ---
st.sidebar.header("About This Tool")
st.sidebar.info(
    "This AI-powered tool analyzes your architectural design requirements, as captured in a "
    "conversation JSON file. It consults a knowledge base of established architectural standards "
    "to provide you with a detailed design brief, including relevant guidelines, dimensional "
    "considerations, and preliminary lists of items for key building systems (Civil, Electrical, Plumbing)."
)
st.sidebar.markdown("---")

st.sidebar.markdown(f"**Knowledge Base Status:**")
if rag_system_config:
    try:
        # Only create ChromaManager if not already created and stored in session_state for this check
        if 'sidebar_chroma_manager' not in st.session_state:
            st.session_state.sidebar_chroma_manager = ChromaManager(
                path=rag_system_config.CHROMA_DB_PATH,
                collection_name=rag_system_config.COLLECTION_NAME
            )

        item_count = st.session_state.sidebar_chroma_manager.count()
        if item_count > 0:
            # Kept the item count as it's informative
            st.sidebar.caption(
                "Derived from standard architectural references including:")
            # Clean up book names for display
            cleaned_book_names = [
                name.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
                for name in rag_system_config.BOOKS_TO_PROCESS
            ]
            for book_name in cleaned_book_names:
                st.sidebar.caption(f"- {book_name.title()}")
        else:
            st.sidebar.warning(
                f"Knowledge Base is empty. Please ensure data has been ingested by the system administrators.")
    except Exception as e_chroma_sidebar:
        st.sidebar.error(f"Could not check Knowledge Base status.")
        # Log to console
        print(f"Error checking ChromaDB for sidebar: {e_chroma_sidebar}")
else:
    st.sidebar.warning(
        "System configuration not loaded; cannot check Knowledge Base.")

# --- Download Button for RAG Output in Sidebar ---
if st.session_state.final_rag_result and st.session_state.uploaded_file_name_for_download:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Download Output")
    final_json_string_to_download = json.dumps(
        st.session_state.final_rag_result, indent=2, ensure_ascii=False)
    download_filename = f"architectural_analysis_{os.path.splitext(st.session_state.uploaded_file_name_for_download)[0]}_{time.strftime('%Y%m%d-%H%M%S')}.json"

    st.sidebar.download_button(
        label="📥 Download Analysis (JSON)",
        data=final_json_string_to_download,
        file_name=download_filename,
        mime="application/json",
        key="download_rag_output_btn_sidebar"
    )

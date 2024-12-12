import streamlit as st
import os
from pathlib import Path
from agents.rag_agent import PharmaceuticalRAGAgent
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize session state
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_agent' not in st.session_state:
        try:
            # Initialize RAG agent with the correct data directory
            data_dir = str(Path(__file__).parent.parent / "datasets" / "processed")
            logger.info(f"Initializing RAG agent with data directory: {data_dir}")
            st.session_state.rag_agent = PharmaceuticalRAGAgent(
                data_dir=data_dir,
                model_name="llama3.2"
            )
            logger.info("RAG agent initialized successfully")
        except Exception as e:
            st.error(f"Error initializing RAG agent: {str(e)}")
            logger.error(f"RAG agent initialization error: {e}", exc_info=True)
            st.session_state.rag_agent = None

def main():
    st.set_page_config(
        page_title="Pharmaceutical Assistant",
        page_icon="💊",
        layout="wide"
    )

    # Initialize session state
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("💊 Pharma Assistant")
        st.markdown("""
        Welcome to the Pharmaceutical Assistant! I can help you with:
        - Drug information
        - Side effects
        - Dosage information
        - Drug interactions
        - General medical advice
        
        **Note:** Always consult healthcare professionals for medical decisions.
        """)
        
        # Add a clear conversation button
        if st.button("Clear Conversation", type="secondary"):
            st.session_state.messages = []
            st.session_state.rag_agent.reset_conversation()
            st.success("Conversation cleared!")

    # Main chat interface
    st.title("💬 Chat with Pharma Assistant")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about medications..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            if st.session_state.rag_agent is None:
                error_msg = "⚠️ The pharmaceutical assistant is currently unavailable. Please try again later."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                message_placeholder = st.empty()
                try:
                    with st.spinner("Searching medical databases..."):
                        response = st.session_state.rag_agent.query(prompt)
                        
                    # Format response for better readability
                    formatted_response = response.replace("Source:", "**Source:**")
                    formatted_response = formatted_response.replace("Note:", "**Note:**")
                    
                    message_placeholder.markdown(formatted_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                    
                except Exception as e:
                    error_message = f"❌ Error: {str(e)}\n\nPlease try again or rephrase your question."
                    message_placeholder.error(error_message)
                    logger.error(f"Error processing query: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p style='color: #666; font-size: 0.8em;'>
                This is an AI assistant for pharmaceutical information. 
                Always verify information with healthcare professionals.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

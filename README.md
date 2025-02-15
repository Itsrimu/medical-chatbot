# medical-chatbot
# Medibot Chat

Medibot Chat is an AI-powered chatbot that helps users with health-related queries. It utilizes a retrieval-based question-answering system powered by FAISS for vector search and Hugging Face models for text generation.

## Features
AI-powered chat for health-related queries
Retrieval-based question answering** using FAISS
Hugging Face modelsfor intelligent responses
Custom prompt template to maintain response accuracy
Chat history to maintain conversation context
Easy-to-use UI built with Streamlit

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.12
- pip
- Virtual environment ( recommended)

### Steps
1. Clone the repository:`

2. Create and activate a virtual environment:
   ```sh
   python -m venv medibot
   source medibot/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory and add:
     ```sh
     HF_TOKEN=your_huggingface_token
     ```

5. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage
- Open the Streamlit app in your browser.
- Enter your health-related query in the chat input.
- The chatbot retrieves the most relevant answer based on its database.
- If the answer is not available, it will inform you.




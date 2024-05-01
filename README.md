# PDF Conversational Interface via WhatsApp

This project is a Flask-based application designed to provide conversational access to PDF documents through WhatsApp. Using the Twilio API for WhatsApp messaging and Langchain for AI-driven text interactions, the application enables users to ask questions and receive answers derived directly from the content of PDF files.

## Technologies Used

- **Flask**: A micro web framework for Python, used to create web applications easily and with minimal code.
- **Twilio API**: Used for WhatsApp messaging, allowing the application to interact with users through WhatsApp messages. For more details on Twilio's API, visit the [Twilio Documentation](https://www.twilio.com/docs).
- **dotenv**: Manages environment variables from a `.env` file which stores confidential keys securely.
- **Langchain**: Provides tools for building language models and integrating conversational AI capabilities, including text splitting and conversational retrieval chains.
- **PyPDF2**: A Pure-Python library built as a PDF toolkit. It is capable of extracting document information, splitting documents page by page, merging documents page by page, cropping pages, and more.
- **FAISS (Facebook AI Similarity Search)**: Used for efficient similarity searching of high-dimensional vectors.
- **OpenAI's API**: Provides AI models for embedding generation, which are used to convert text into vector form that can be efficiently searched.
- **Ngrok**: A utility used to expose local servers behind NATs and firewalls to the public internet over secure tunnels.

Your Mission: Build a NASA Intelligence Chat System
For this project, you'll step into the shoes of a NASA mission operations specialist. Your task is to build a Q&A system that can answer questions about some of NASA's most historic space missions. You'll be working with actual mission transcripts and technical documents from Apollo 11, Apollo 13, and the Challenger missions.

The goal is to create a tool that allows astronauts, researchers, or even a curious historian to ask a question in plain English—like "What problems did Apollo 13 encounter?"—and get an accurate, detailed answer sourced directly from NASA's own archives.

To do this, you are going to build a complete Retrieval-Augmented Generation (RAG) system.

The Project Blueprint
You'll be working through a series of Python files, each with a specific job.

Here’s a high-level look at what you'll be building, piece by piece:

The Embedding Pipeline First, you'll take all those raw NASA text files and process them. You'll write code to break them into smaller, manageable chunks and then convert those chunks into numerical representations—or embeddings and store them in ChromaDB.
The RAG Client This is the core of your retrieval system. You'll build the logic that takes a user's question, searches the ChromaDB database to find the most relevant document chunks, and then formats that information neatly to be used as context.
The LLM Client Here, you'll connect to the OpenAI API. This component will take the user's question and the context from your RAG client and generate a helpful, human-readable answer.
The RAGAS Evaluator How do you know if your RAG system is any good? You'll implement a real-time evaluation system using RAGAS.
The Chat Application Finally, you'll bring everything together in an interactive chat interface using Streamlit.
What You'll Be Able to Do After This
By the time you finish, you will have built a complete, functioning AI application. You'll have demonstrated a whole set of valuable skills, including:

Building an end-to-end RAG system.
Using vector databases like ChromaDB for semantic search.
Integrating and prompting large language models for specific tasks.
Evaluating the performance of an AI system with modern tools.
This project is a fantastic piece to add to your portfolio. Ready to get started?

Project Assessment
Your project will be assessed by mentors using a detailed rubric. On the following pages, you'll find the rubric. Familiarize yourself with the rubric and make sure to check your project against it before you submit it.
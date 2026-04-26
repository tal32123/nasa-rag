Phase 1: Core Infrastructure
This first phase is all about building the foundational components of your RAG system.

1. Implement the LLM Client (llm_client.py)
Here, you'll connect to the OpenAI API. You'll learn how to write a good system prompt to give the AI its "persona" as a NASA expert. This component will take the user's question and the context from your RAG client and generate a helpful, human-readable answer.
What You'll Do: Your first step is to create a connection to the OpenAI API. You'll write the code to send a user's question, along with any relevant context and conversation history, to an LLM like GPT-3.5 or GPT-4.
Tasks:
Define a system prompt that tells the model to act as a NASA expert.
Manage conversation history so the model can remember previous turns.
Write the function that sends the request to OpenAI and returns the model's response.
2. Build the RAG Client (rag_client.py)
This is the core of your retrieval system. You'll build the logic that takes a user's question, searches the ChromaDB database to find the most relevant document chunks, and then formats that information neatly to be used as context.
What You'll Do: Next, you'll build the "retrieval" part of the RAG system. This component is responsible for searching the vector database to find the most relevant documents to answer a user's question.
Tasks:
Connect to the ChromaDB backend.
Implement the semantic search function that takes a question and finds the best matching document chunks.
Format the retrieved documents into a clean context string that can be passed to the LLM.
3. Create the Embedding Pipeline (embedding_pipeline.py)
You'll take all those raw NASA text files and process them. You'll write code to break them into smaller, manageable chunks and then convert those chunks into numerical representations—or embeddings—using an OpenAI model. You will then store all of this in a specialized vector database called ChromaDB. This is the foundation of your system's "memory."
What You'll Do: This is the most extensive part of the setup. You'll write a script that takes all the raw NASA .txt files, processes them, creates embeddings, and saves them into the ChromaDB database.
Tasks:
Implement a text chunking strategy to break large documents into smaller, more manageable pieces.
Use the OpenAI API to generate embeddings for each text chunk.
Manage the creation and population of collections within ChromaDB.
Build a command-line interface so you can easily run this pipeline from your terminal.
Phase 2: Evaluation and Interface
Once the core system is built, you'll focus on evaluating its performance and creating a user-friendly interface.

4. Develop the RAGAS Evaluator (ragas_evaluator.py)
How do you know if your RAG system is any good? You'll implement a real-time evaluation system using a framework called RAGAS. This will automatically score your system's answers on metrics like faithfulness (Is it sticking to the facts?) and relevancy (Is the answer actually helpful?).
What You'll Do: You'll build the system that scores how good your RAG system's answers are. This will give you real-time feedback on your system's quality.
Key Tasks:
Integrate the RAGAS framework.
Define the evaluation metrics you want to use, such as answer relevancy and faithfulness.
Write the function that takes a question, an answer, and the context and returns a set of quality scores.
5. Build the Chat Application (chat.py)
Finally, you'll bring everything together in an interactive chat interface using Streamlit. This is where your project comes to life! Users will be able to select different models, choose a mission to focus on, and see the evaluation scores for each answer in real-time.
What You'll Do: You'll bring everything together into a single, interactive web application using Streamlit.
Tasks:
Build the chat interface where a user can type in questions.
Integrate all the components you've built: the RAG client, the LLM client, and the RAGAS evaluator.
Display the AI's answer and its real-time quality scores in the interface.
Project Assessment
Your project will be assessed by mentors using a detailed rubric. On the following pages, you'll find the rubric. Familiarize yourself with the rubric and make sure to check your project against it before you submit it.

Submission Instructions
When you are ready to submit your project, please follow these steps to make sure everything is included.

Submission Checklist
Implement All TODO Items: Go through each of the Python files (llm_client.py, rag_client.py, embedding_pipeline.py, ragas_evaluator.py, and chat.py) and make sure you have completed all the TODO comments.

End-to-End Testing: Before submitting, run the entire workflow to confirm that everything works together.

First, run your embedding pipeline to process the documents.
Then, launch the chat application and test it with several questions to make sure it responds correctly and displays the evaluation scores.
Provide Sample Questions: In a text file called "evaluation_dataset.txt", Include a few sample questions that you used for testing and show the responses you expected the system to provide.

Prepare Your Files:

Make sure all your code is clean.
Zip up all your project files, including your report, into a single archive.
On the following pages, you'll find a submission button and instructions for zipping up your code and submitting it to mentors. We recommend you work in a Github repository, either in the Udacity workspace or on your local machine.
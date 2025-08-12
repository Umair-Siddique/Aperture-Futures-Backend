# import whisper
# import os

# def transcribe_audio(audio_file_path: str) -> str:
#     """
#     Transcribes an audio file (e.g., MP3) into text using the Whisper ASR model.

#     Args:
#         audio_file_path (str): The path to the audio file to be transcribed.

#     Returns:
#         str: The transcribed text from the audio file.
#     """
#     if not os.path.exists(audio_file_path):
#         print(f"Error: Audio file not found at '{audio_file_path}'")
#         return ""

#     print(f"Loading Whisper model (this may take a moment)...")
#     # You can choose different models based on your needs:
#     # 'tiny', 'base', 'small', 'medium', 'large'
#     # Larger models are more accurate but require more resources and time.
#     model = whisper.load_model("base")
#     print("Model loaded. Starting transcription...")

#     try:
#         # Transcribe the audio file
#         result = model.transcribe(audio_file_path,)
#         transcribed_text = result["text"]
#         print("Transcription complete!")
#         return transcribed_text
#     except Exception as e:
#         print(f"An error occurred during transcription: {e}")
#         return ""

# if __name__ == "__main__":
#     # --- IMPORTANT: Replace 'your_audio_file.mp3' with the actual path to your MP3 file ---
#     # Example:
#     # audio_path = "C:/Users/YourUser/Documents/my_meeting_audio.mp3"
#     # audio_path = "/home/youruser/recordings/podcast_episode.mp3"
#     audio_path = "3431341.mp3" # <--- CHANGE THIS TO YOUR MP3 FILE PATH

#     if audio_path == "your_audio_file.mp3":
#         print("\n--- ATTENTION ---")
#         print("Please change the 'audio_path' variable in the script to the actual path of your MP3 file.")
#         print("For example: audio_path = 'path/to/your/audio.mp3'")
#         print("-----------------\n")
#     else:
#         transcribed_text = transcribe_audio(audio_path)

#         if transcribed_text:
#             print("\n--- Transcribed Text ---")
#             print(transcribed_text)
#             print("------------------------")
#         else:
#             print("\nNo text was transcribed. Please check the file path and ensure the audio file is valid.")


# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# import os
# import time
# import whisper
# import uuid
# from supabase import create_client
# from extensions import init_supabase, init_openai_embeddings
# from config import Config
# from pinecone import Pinecone
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai import OpenAIEmbeddings
# import tempfile

# app = Flask(__name__)

# def init_supabase(app):
#     app.supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

# def init_openai_embeddings(app):
#     app.embeddings = OpenAIEmbeddings(
#         model="text-embedding-3-large",
#         openai_api_key=Config.OPENAI_API_KEY,
#     )

# init_supabase(app)
# init_openai_embeddings(app)

# # --------- Pinecone v3 Initialization ---------
# pc = Pinecone(api_key=Config.PINECONE_API_KEY)
# pinecone_index = pc.Index('meeting-transcripts')

# def preprocess_and_chunk(text, embeddings):
#     text_splitter = SemanticChunker(
#         embeddings,
#         breakpoint_threshold_type="gradient"
#     )
#     return text_splitter.split_text(text)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['mp3', 'wav', 'm4a']

# @app.route("/transcribe", methods=["POST"])
# def transcribe_and_store():
#     title = request.form.get("title")
#     audio_file = request.files.get("audio")
#     if not title or not audio_file:
#         return jsonify({"error": "title and audio file required"}), 400

#     response = app.supabase.table('audio_files').select('title').eq('title', title).execute()
#     if response.data:
#         return jsonify({"error": f"title '{title}' already exists"}), 409

#     filename = f"{uuid.uuid4().hex}_{secure_filename(audio_file.filename)}"
#     temp_dir = tempfile.gettempdir()
#     filepath = os.path.join(temp_dir, filename)
#     audio_file.save(filepath)

#     timestamp = int(time.time())
#     app.supabase.table('audio_files').insert({"title": title, "timestamp": timestamp}).execute()

#     model = whisper.load_model("base")
#     result = model.transcribe(filepath)
#     transcript = result.get("text", "")

#     if not transcript:
#         os.remove(filepath)
#         return jsonify({"error": "Transcription failed"}), 500

#     chunks = preprocess_and_chunk(transcript, app.embeddings)

#     def batch_embed_and_upsert(chunks, batch_size=16):
#         total_chunks = len(chunks)
#         vectors = []
#         for i in range(0, total_chunks, batch_size):
#             batch_chunks = chunks[i:i+batch_size]
#             batch_embeds = app.embeddings.embed_documents(batch_chunks)
#             vectors.extend([
#                 (
#                     f"{title}_{i+j}",
#                     vec,
#                     {"text": chunk}
#                 )
#                 for j, (vec, chunk) in enumerate(zip(batch_embeds, batch_chunks))
#             ])
#         # Pinecone v3 upsert
#         pinecone_index.upsert(vectors=vectors, namespace=title)
#         return len(vectors)

#     chunks_stored = batch_embed_and_upsert(chunks, batch_size=16)

#     os.remove(filepath)

#     return jsonify({
#         "title": title,
#         "timestamp": timestamp,
#         "chunks_stored": chunks_stored
#     })

# if __name__ == "__main__":
#     app.run(debug=True)


import whisper
import os


def transcribe_audio(file_path='3431631.mp3', model_size='base', output_file='transcript.txt'):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    model = whisper.load_model(model_size)
    result = model.transcribe(file_path)
    transcript = result.get("text", "")

    if not transcript:
        print("Transcription failed or returned empty result.")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcript)

    print(f"Transcript saved successfully to '{output_file}'.")


if __name__ == "__main__":
    transcribe_audio()
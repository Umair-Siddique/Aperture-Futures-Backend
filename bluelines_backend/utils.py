import os
import fitz  # PyMuPDF
import tempfile
from google.cloud import storage
import re

def get_blobs_for_year(bucket_name: str, year: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs()
    year_pattern = re.compile(r'_RES_\d+_(\d{4})_')

    matching_blobs = []

    for blob in blobs:
        match = year_pattern.search(blob.name)
        if match and match.group(1) == year:
            matching_blobs.append(blob)

    return matching_blobs

def extract_text_from_blob(blob) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file_path = tmp_file.name

    try:
        blob.download_to_filename(tmp_file_path)

        doc = fitz.open(tmp_file_path)
        extracted_text = ""
        for page_num in range(len(doc)):
            extracted_text += doc[page_num].get_text()
        doc.close()

        return extracted_text.replace("Decides to remain seized of the matter", "")
    finally:
        # Ensure temporary file is removed
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

from datetime import timedelta

def process_documents_for_year(bucket_name: str, year: str):
    blobs = get_blobs_for_year(bucket_name, year)
    print(f"Found {len(blobs)} documents for year {year}.")

    all_results = []

    for idx, blob in enumerate(blobs):
        try:
            print(f"\nProcessing {blob.name}...")

            # Construct public URL (only if bucket is public)
            url = f"https://storage.googleapis.com/{bucket_name}/{blob.name}"
            print(f"URL: {url}")

            text = extract_text_from_blob(blob)
            # if you want to use signed url, use generate_signed_url(blob)

            # Print the full text only for the first file, else print length summary
            if idx == 0:
                print("Extracted text:\n", text[:2000], "...")  # first 2000 chars only for readability
            else:
                print(f"Extracted text length: {len(text)} characters")

            all_results.append({
                "filename": blob.name,
                "url": url,
                "text": text,
            })

            print(f"✅ Done: {blob.name}")
        except Exception as e:
            print(f"❌ Failed: {blob.name} — {e}")

    return all_results


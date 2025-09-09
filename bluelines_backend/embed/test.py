from google.cloud import storage
import re

# Initialize GCS client
client = storage.Client()
bucket_name = 'bluelines-rag-bucket'  # Replace with your bucket
bucket = client.bucket(bucket_name)

# Regex to extract the year only (second 4-digit number after _RES_)
year_pattern = re.compile(r'_RES_\d+_(\d{4})_')

# List all files in the bucket
blobs = list(client.list_blobs(bucket))

for blob in blobs:
    filename = blob.name
    match = year_pattern.search(filename)

    if match:
        year = match.group(1)
        new_blob_name = f"{year}/{filename}"

        if new_blob_name != filename:
            # Copy to year folder
            new_blob = bucket.copy_blob(blob, bucket, new_blob_name)
            print(f"Copied {filename} to {new_blob_name}")

            # Optionally delete the original
            blob.delete()
            print(f"Deleted original blob: {filename}")
    else:
        print(f"Could not extract year from: {filename}")

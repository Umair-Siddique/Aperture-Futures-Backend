from textwrap import dedent
from typing import Dict, List
from flask import current_app
from config import Config
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from blueprints.system_prompt import (
    get_system_prompt, 
    PROMPT_KEY_REPORT_GENERATION,
    DEFAULT_REPORT_GENERATION_SYSTEM_PROMPT
)


def format_transcript_text(transcript: str) -> str:
    """
    Formats a raw transcript into properly structured markdown with headings.
    Uses RecursiveCharacterTextSplitter to split after paragraphs, max 10k tokens per chunk.
    AI can generate headings if not present, but must preserve ALL original content.
    """
    if not transcript or not transcript.strip():
        return transcript
    
    # Check if OpenAI client is properly initialized
    if not hasattr(current_app, 'openai_client') or current_app.openai_client is None:
        current_app.logger.warning("OpenAI client not initialized. Returning unformatted transcript.")
        return transcript
    
    system_prompt = """You are a professional transcript formatter. Your job is to format raw transcript text into well-structured, readable markdown.

CRITICAL RULES:
- PRESERVE ALL ORIGINAL CONTENT - do NOT remove, skip, or omit any text from the original transcript
- You may add markdown headings (##, ###) to organize content if they help structure the transcript
- If headings are not in the original text, you may infer logical headings based on topic changes or speaker transitions
- Format with proper paragraph breaks, punctuation, and capitalization
- Use markdown formatting (bold, italics, lists) to improve readability
- Maintain chronological order of the original content
- If speakers are identified, preserve their names/identifiers
- Ensure all sentences and paragraphs from the original are included

Your output should be the same content, but well-organized with proper markdown structure and headings."""

    try:
        # Use RecursiveCharacterTextSplitter to split after paragraphs
        # Target ~10k tokens (approximately 7500-8000 characters)
        # Priority: split after paragraphs, then sentences, then words
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=7500,  # ~10k tokens (roughly 4 chars per token)
            chunk_overlap=200,  # Small overlap to maintain context between chunks
            separators=["\n\n", "\n", ". ", " ", ""]  # Priority: paragraphs, then lines, then sentences, then words
        )
        
        # Split transcript into chunks
        chunks = text_splitter.split_text(transcript)
        current_app.logger.info(f"Split transcript into {len(chunks)} chunks for formatting (original: {len(transcript)} chars)")
        
        if len(chunks) == 1:
            # Process small transcript directly
            user_prompt = f"""Format this transcript into well-structured markdown with proper headings and formatting. 
Preserve ALL content from the original - do not remove or skip anything.

Original transcript:
{transcript}"""
            
            completion = current_app.openai_client.chat.completions.create(
                model="gpt-5-mini-2025-08-07",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            formatted_transcript = completion.choices[0].message.content
        else:
            # Process chunks sequentially
            formatted_chunks = []
            
            for i, chunk in enumerate(chunks):
                current_app.logger.info(f"Formatting transcript chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                
                # Add context about chunk position for better formatting
                chunk_context = ""
                if i > 0:
                    chunk_context = "This is a continuation of a longer transcript. "
                if i < len(chunks) - 1:
                    chunk_context += "This chunk will be followed by more content. "
                
                user_prompt = f"""{chunk_context}Format this portion of the transcript into well-structured markdown with proper headings and formatting.
Preserve ALL content from the original - do not remove or skip anything.

Transcript portion:
{chunk}"""
                
                completion = current_app.openai_client.chat.completions.create(
                    model="gpt-5-mini-2025-08-07",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                formatted_chunks.append(completion.choices[0].message.content)
                time.sleep(0.3)  # Small delay to avoid rate limiting
            
            # Combine chunks with proper spacing
            # Remove any duplicate headings at boundaries if they exist
            formatted_transcript = "\n\n".join(formatted_chunks)
        
        current_app.logger.info(f"Transcript formatting completed. Original: {len(transcript)} chars, Formatted: {len(formatted_transcript)} chars")
        return formatted_transcript
        
    except Exception as e:
        current_app.logger.error(f"Transcript formatting failed: {str(e)}. Returning original transcript.")
        import traceback
        current_app.logger.error(f"Formatting error traceback: {traceback.format_exc()}")
        # Return original transcript if formatting fails
        return transcript


def _build_system_prompt() -> str:
    """
    Return the instructions for formatting and structuring the report.
    Fetches from Supabase if available, otherwise uses the default prompt.
    """
    # Use the default prompt constant from system_prompt.py to avoid duplication
    return get_system_prompt(PROMPT_KEY_REPORT_GENERATION, DEFAULT_REPORT_GENERATION_SYSTEM_PROMPT)


def _chunk_transcript(transcript: str, chunk_size: int = 20000) -> List[str]:
    """Split transcript into chunks suitable for API processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=500,  # Overlap to maintain context
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_text(transcript)


def _generate_report_chunk(chunk: str, chunk_index: int, total_chunks: int) -> str:
    """Generate report for a single chunk of transcript."""
    if not hasattr(current_app, 'openai_client') or current_app.openai_client is None:
        raise ValueError("OpenAI client not properly initialized. Check OPENAI_API_KEY in .env file.")

    system_prompt = _build_system_prompt()
    
    # Add chunk context to the prompt
    chunk_prompt = f"""
This is chunk {chunk_index + 1} of {total_chunks} from a UN Security Council transcript. Explain and summarize Briefly.

{chunk}

Please generate a diplomatic report for this portion of the transcript. Focus on the key points, interventions, and diplomatic positions mentioned in this section.
"""
    
    try:
        completion = current_app.openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk_prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        if "api_key" in str(e).lower() or "authentication" in str(e).lower():
            raise ValueError(f"OpenAI API authentication failed. Check your OPENAI_API_KEY in .env file. Error: {str(e)}")
        else:
            raise ValueError(f"OpenAI API error: {str(e)}")


def _combine_chunk_reports(chunk_reports: List[str]) -> str:
    """Combine multiple chunk reports into a single comprehensive report."""
    if not chunk_reports:
        return ""
    
    if len(chunk_reports) == 1:
        return chunk_reports[0]
    
    # Create a prompt to combine the reports
    combined_prompt = f"""
You are a UN policy analyst. Below are {len(chunk_reports)} separate reports from different portions of the same UN Security Council meeting transcript. 

Please combine these into a single, comprehensive diplomatic report that:
1. Eliminates redundancy
2. Maintains chronological flow where relevant
3. Preserves all important diplomatic positions and interventions
4. Follows the same structure as the individual reports

Individual reports:
"""
    
    for i, report in enumerate(chunk_reports, 1):
        combined_prompt += f"\n\n--- Report {i} ---\n{report}"
    
    combined_prompt += """

Please provide the combined report in the same format:
- First the full report in English
- Then the same report in fluent Danish

Ensure the Danish translation is natural and fluent, not literal.
"""
    
    try:
        completion = current_app.openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a UN policy analyst expert at combining multiple reports into comprehensive diplomatic summaries."},
                {"role": "user", "content": combined_prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        # If combination fails, return the first report as fallback
        current_app.logger.warning(f"Failed to combine reports: {str(e)}. Returning first report.")
        return chunk_reports[0]


def generate_and_store_transcription_report(title: str, transcript: str) -> dict:
    """
    Generates a diplomatic report from the full transcript using OpenAI GPT-4o
    with chunking support for large transcripts and stores it into audio_files.transcription_report.
    """
    
    # Check if OpenAI client is properly initialized
    if not hasattr(current_app, 'openai_client') or current_app.openai_client is None:
        raise ValueError("OpenAI client not properly initialized. Check OPENAI_API_KEY in .env file.")

    current_app.logger.info(f"Starting report generation for: {title}")
    current_app.logger.info(f"Transcript length: {len(transcript)} characters")
    
    try:
        # Check if transcript is too large and needs chunking
        if len(transcript) > 20000:  # Rough estimate for token count
            current_app.logger.info("Transcript is large, using chunking approach")
            
            # Split transcript into chunks
            chunks = _chunk_transcript(transcript, chunk_size=20000)
            current_app.logger.info(f"Split transcript into {len(chunks)} chunks")
            
            # Generate reports for each chunk
            chunk_reports = []
            for i, chunk in enumerate(chunks):
                current_app.logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
                try:
                    chunk_report = _generate_report_chunk(chunk, i, len(chunks))
                    if chunk_report:
                        chunk_reports.append(chunk_report)
                    else:
                        current_app.logger.warning(f"Empty report for chunk {i + 1}")
                except Exception as e:
                    current_app.logger.error(f"Error processing chunk {i + 1}: {str(e)}")
                    continue
                
                # Add small delay between API calls to avoid rate limiting
                time.sleep(1)
            
            if not chunk_reports:
                raise ValueError("Failed to generate any chunk reports")
            
            # Combine chunk reports into final report
            if len(chunk_reports) > 1:
                current_app.logger.info("Combining chunk reports into final report")
                final_report = _combine_chunk_reports(chunk_reports)
            else:
                final_report = chunk_reports[0]
                
        else:
            # Process small transcript directly
            current_app.logger.info("Transcript is small, processing directly")
            final_report = _generate_report_chunk(transcript, 0, 1)
        
        if not final_report:
            raise ValueError("Failed to generate report")
        
        current_app.logger.info(f"Report generation completed. Report length: {len(final_report)} characters")
        
        # Store into Supabase
        current_app.supabase.table("audio_files") \
            .update({"transcription_report": final_report}) \
            .eq("title", title) \
            .execute()

        return {"ok": True, "chars_used": len(transcript), "report_length": len(final_report)}

    except Exception as e:
        current_app.logger.error(f"Report generation failed: {str(e)}")
        raise ValueError(f"Report generation failed: {str(e)}")
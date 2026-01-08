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
    
    system_prompt = """You are LiveLines, a specialized transcript-formatting assistant for
United Nations Security Council (UNSC) meetings.
 
Your task is to transform raw, error-prone ASR transcripts into a clean,
readable, UN-style verbatim transcript that mirrors official S/PV records,
while preserving the speaker’s original words as faithfully as possible.
 
You are NOT a summarizer. You are NOT an analyst.
You are a formatter, segmenter, and structural corrector.
 
────────────────────────────────────────
NON-NEGOTIABLE RULES (HARD CONSTRAINTS)
────────────────────────────────────────
1) DO NOT summarize, paraphrase, or add content.
2) DO NOT invent or “fix” missing information.
3) Preserve the speaker’s wording as spoken.
4) You MAY fix:
   - capitalization
   - punctuation
   - paragraph breaks
   - obvious disfluencies (e.g. repeated sentence starts),
     ONLY when meaning is unchanged.
5) NEVER merge text from different speakers into one block.
6) If text from different speakers is interleaved, you MUST split and reassign it.
7) Remove exact duplicate sentences or paragraphs (keep first occurrence only).
8) If attribution is unclear, label it explicitly as:
   SPEAKER: Unknown | ROLE: Unknown
9) DO NOT guess speakers, countries, or roles based on outside knowledge.
10) If something is unclear or inaudible, mark it as [unclear] or [inaudible].
 
────────────────────────────────────────
UNSC FORMAT PACK (STRUCTURE, NOT GUESSING)
────────────────────────────────────────
 
A) Typical UNSC meeting flow (may vary):
1. Meeting called to order (President/Chair)
2. Agenda adoption (President/Chair)
3. Invitations under Rule 37 (non-Council Member States)
4. Invitations under Rule 39 (briefers: UN officials, others)
5. Briefings (Rule 39)
6. Statements by Council members
7. Statements by invited participants
8. Concluding remarks / adjournment
 
B) Procedural language cues
Treat the following phrases as PRESIDENT / CHAIR unless text clearly says otherwise:
- “The meeting is called to order.”
- “The provisional agenda is adopted.”
- “In accordance with rule 37/39…”
- “It is so decided.”
- “I thank the representative/briefer of…”
- “I now give the floor to…”
- “There are no more names inscribed…”
- “The meeting is adjourned.”
 
C) Speaker classification rules
- Procedural cues → ROLE: President/Chair
- UN officials explicitly invited → ROLE: Briefer (Rule 39)
- Participants invited under rule 37 → ROLE: Invited Participant
- State statements without procedural language → ROLE: Council Member
- If unsure → ROLE: Unknown (do NOT guess)
 
D) Anti-hallucination guardrail
DO NOT use knowledge of UNSC membership, presidency rotation,
or usual speaking order to infer or “correct” attribution.
 
────────────────────────────────────────
INPUT CHARACTERISTICS
────────────────────────────────────────
The raw transcript may include:
- broken paragraphs
- repeated text
- incorrect headings (e.g. “Statement”, “Transcript continuation”)
- procedural remarks out of order
- speaker interleaving
- foreign-language fragments
- role mislabeling
- ASR artifacts and noise
 
────────────────────────────────────────
OUTPUT FORMAT (STRICT — NO DEVIATION)
────────────────────────────────────────
 
[MEETING METADATA]
- Meeting: Security Council (number if stated, otherwise “unknown”)
- Date/Time: (if stated, otherwise “unknown”)
- Agenda: (if stated, otherwise “unknown”)
- President: (if stated, otherwise “unknown”)
 
[PROCEDURAL OPENING]
(President/Chair) …   (only if present)
 
[STATEMENTS]
 
SPEAKER: <Country / Entity / Person as stated>
ROLE: <President/Chair | Council Member | Briefer (Rule 39) | Invited Participant (Rule 37) | Unknown>
TEXT:
<Paragraphs of verbatim speech>
 
[PROCEDURAL TRANSITIONS]
(President/Chair) I thank … I now give the floor to …
 
[PROCEDURAL CLOSING]
(President/Chair) The meeting is adjourned.   (only if present)
 
────────────────────────────────────────
DETAILED PROCESS (FOLLOW IN ORDER)
────────────────────────────────────────
 
Step 1 — Detect speaker boundaries
- Start a new speaker block when encountering:
  - “Representative of…”
  - “President…”
  - “I now give the floor…”
  - “I thank the representative of…”
- Also split when content clearly shifts voice, country, or role.
 
Step 2 — Remove structural noise
- Delete headings such as:
  “Statement”, “Continued transcript”, “Transcript continuation”
- Keep meaningful procedural actions as bracketed notes if needed.
 
Step 3 — De-interleave aggressively
- If text within a paragraph clearly belongs to another speaker,
  split at that point and reassign.
- Do NOT allow one speaker block to contain:
  procedural control + substantive national positions.
 
Step 4 — Language handling
- If non-English fragments appear:
  - Preserve verbatim.
  - Do NOT translate unless instructed.
  - If attribution unclear, place under SPEAKER: Unknown.
 
Step 5 — Consistency check before final output
- No Council Member statement should contain “I now give the floor…”
- Procedural remarks must be attributed only to President/Chair.
- Remove duplicated paragraphs.
- Ensure every block has SPEAKER, ROLE, and TEXT.
 
────────────────────────────────────────
QUALITY BAR
────────────────────────────────────────
The final transcript must:
- Be readable and clearly segmented by speaker
- Resemble an official UNSC verbatim record
- Preserve the original wording
- Explicitly flag uncertainty rather than guessing"""

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
                model="gpt-5.2-2025-12-11",
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
            model="gpt-5.2-2025-12-11",
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
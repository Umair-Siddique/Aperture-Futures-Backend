from textwrap import dedent
from typing import Dict, List
from flask import current_app
from config import Config
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from blueprints.system_prompt import (
    get_system_prompt, 
    PROMPT_KEY_REPORT_GENERATION
)


def format_transcript_text(transcript: str) -> str:
    """
    Formats a raw transcript into properly structured markdown with headings.
    Uses RecursiveCharacterTextSplitter to split after paragraphs, max 10k tokens per chunk.
    AI can generate headings if not present, but must preserve ALL original content.
    Outputs in English only with clean markdown formatting.
    """
    if not transcript or not transcript.strip():
        return transcript
    
    # Check if OpenAI client is properly initialized
    if not hasattr(current_app, 'openai_client') or current_app.openai_client is None:
        current_app.logger.warning("OpenAI client not initialized. Returning unformatted transcript.")
        return transcript
    
    system_prompt = """You are LiveLines, a specialized transcript-formatting assistant for United Nations Security Council (UNSC) meetings.
 
Your task is to transform raw, error-prone ASR transcripts into a clean, readable, UN-style verbatim transcript that mirrors official S/PV records, while preserving the speaker's original words as faithfully as possible.
 
Your primary function is transcript normalization and structuring. Summarization or analysis is out of scope unless explicitly requested.
 
────────────────────────────────────────
NON-NEGOTIABLE RULES (HARD CONSTRAINTS)
────────────────────────────────────────
1) DO NOT summarize, paraphrase, or add content.
2) DO NOT invent or "fix" missing information.
3) Preserve the speaker's wording as spoken.
4) You MAY fix:
   - capitalization
   - punctuation
   - paragraph breaks
   - obvious disfluencies (e.g. repeated sentence starts), ONLY when meaning is unchanged.
5) NEVER merge text from different speakers into one block.
6) If text from different speakers is interleaved, split and reassign it ONLY when an explicit speaker boundary is present.
7) Remove exact duplicate sentences or paragraphs (keep first occurrence only).
8) DO NOT guess speakers, countries, or roles based on outside knowledge.
9) If something is unclear or inaudible, mark it as [unclear] or [inaudible].
 
────────────────────────────────────────
SPEAKER CONTINUITY & ATTRIBUTION RULES
────────────────────────────────────────
SPEAKER CONTINUITY OVERRIDE:
Once a speaker has been identified, assume all subsequent text belongs to that speaker UNTIL one of the following explicit boundaries appears:
- "I now give the floor to…"
- "I thank you"
- "I thank the representative of…"
- "Representative of [X]"
- "President" / "Chair"
- A clearly labeled new speaker line
 
Tone, rhetoric, or political position changes alone are NOT valid grounds for starting a new speaker block.
 
UNKNOWN ATTRIBUTION (LAST RESORT ONLY):
Use SPEAKER: Unknown ONLY if:
1) No speaker has been identified previously, AND
2) No explicit speaker cue appears in the text.
 
If a speaker is already active, continue attribution to that speaker unless an explicit boundary is detected.
 
────────────────────────────────────────
UNSC FORMAT PACK (STRUCTURE, NOT GUESSING)
────────────────────────────────────────
 
A) Typical UNSC meeting flow (may vary):
1. Meeting called to order (President/Chair)
2. Agenda adoption (President/Chair)
3. Invitations under Rule 37 (non-Council Member States)
4. Invitations under Rule 39 (briefers)
5. Briefings (Rule 39)
6. Statements by Council members
7. Statements by invited participants
8. Concluding remarks / adjournment
 
B) Procedural language cues
Treat the following phrases as PRESIDENT / CHAIR unless text clearly says otherwise:
- "The meeting is called to order."
- "The provisional agenda is adopted."
- "In accordance with rule 37/39…"
- "It is so decided."
- "I thank the representative/briefer of…"
- "I now give the floor to…"
- "I thank you"
- "There are no more names inscribed…"
- "The meeting is adjourned."
 
C) Speaker classification rules
- Procedural cues → ROLE: President/Chair
- UN officials explicitly invited → ROLE: Briefer (Rule 39)
- Participants invited under rule 37 → ROLE: Invited Participant
- State statements → ROLE: Council Member
- If unclear → ROLE: Unknown (do NOT guess)
 
D) Safe role inference allowance
If a speaker is explicitly named with a title (e.g. "Prime Minister of Ukraine", "Representative of Slovenia"), assign ROLE based on the stated title, even if Council membership is not explicitly mentioned.
 
DO NOT infer roles from UNSC membership lists, presidency rotation, or speaking order.
 
────────────────────────────────────────
INPUT CHARACTERISTICS
────────────────────────────────────────
The raw transcript may include:
- broken paragraphs
- repeated text
- incorrect headings (e.g. "Statement", "Transcript continuation")
- procedural remarks out of order
- speaker interleaving
- foreign-language fragments
- role mislabeling
- ASR artifacts and noise
 
────────────────────────────────────────
OUTPUT FORMAT (STRICT — NO DEVIATION)
────────────────────────────────────────
 
[MEETING METADATA]
- Meeting: Security Council (number if stated, otherwise "unknown")
- Agenda: (if stated, otherwise "unknown")
 
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
- Start a new speaker block ONLY when encountering:
  - "Representative of…"
  - "President" / "Chair"
  - "I now give the floor…"
  - "I thank the representative of…"
- Otherwise, maintain speaker continuity.
 
Step 2 — Remove structural noise
- Delete labels such as: "Statement", "Continued transcript", "Transcript continuation"
- If headings like "Meeting Transcript" or "Meeting Information" appear after the initial header, DISCARD them.
- Do NOT restart or duplicate the transcript structure.
 
Step 3 — De-interleave conservatively
- Split and reassign text ONLY when an explicit boundary is present.
- Example: If a paragraph begins as a national statement but later contains "I now give the floor to…", split the paragraph and move the procedural sentence into a new President/Chair block.
 
Step 4 — Language handling
- Preserve non-English fragments verbatim.
- Do NOT translate unless instructed.
- If attribution is unclear and no active speaker exists, assign SPEAKER: Unknown.
 
Step 5 — Final consistency check
- No Council Member statement may contain procedural control language except the President (first and last speaker of the meeting).
- Procedural remarks must be attributed only to President.
- Remove duplicated paragraphs.
- Ensure every block has SPEAKER, ROLE, and TEXT.
 
────────────────────────────────────────
QUALITY BAR
────────────────────────────────────────
The final transcript must:
- Maintain correct speaker continuity
- Be readable and sequential
- Resemble an official UNSC verbatim record
- Explicitly flag uncertainty rather than guessing"""

    try:
        # Use RecursiveCharacterTextSplitter to split after paragraphs
        # Target ~10k tokens (approximately 7500-8000 characters)
        # Priority: split after paragraphs, then sentences, then words
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=7500,  # ~10k tokens (roughly 4 chars per token)
            chunk_overlap=300,  # Small overlap to maintain context between chunks
            separators=["\n\n", "\n", ". ", " ", ""]  # Priority: paragraphs, then lines, then sentences, then words
        )
        
        # Split transcript into chunks
        chunks = text_splitter.split_text(transcript)
        current_app.logger.info(f"Split transcript into {len(chunks)} chunks for formatting (original: {len(transcript)} chars)")
        
        if len(chunks) == 1:
            # Process small transcript directly
            user_prompt = f"""Format this transcript into well-structured markdown following UNSC verbatim record style.

CRITICAL REQUIREMENTS:
- Output in English ONLY (translate any non-English text naturally and fluently)
- Use markdown format with clear headings (##, ###) as specified in output format
- Preserve ALL content in CHRONOLOGICAL ORDER - do not reorder or skip anything
- Clean up typos, grammar errors, and nonsensical sentences while preserving meaning
- Remove duplicates and filler words (um, uh, like) that don't affect meaning
- Identify and label speakers with their roles using the format specified
- Maintain the original sequence of the meeting from start to finish

Original transcript:
{transcript}"""
            
            completion = current_app.openai_client.chat.completions.create(
                model="gpt-5",
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
                if i == 0:
                    chunk_context = "This is the BEGINNING of a longer transcript. "
                elif i > 0 and i < len(chunks) - 1:
                    chunk_context = f"This is PART {i+1} of {len(chunks)} of a longer transcript (continuation). Maintain consistent formatting and speaker identification with previous sections. "
                elif i == len(chunks) - 1:
                    chunk_context = f"This is the FINAL PART ({i+1} of {len(chunks)}) of a longer transcript. "
                
                user_prompt = f"""{chunk_context}Format this portion of the transcript into well-structured markdown following UNSC verbatim record style.

CRITICAL REQUIREMENTS:
- Output in English ONLY (translate any non-English text naturally and fluently)
- Use markdown format with clear headings (##, ###) as specified in output format
- Preserve ALL content in EXACT CHRONOLOGICAL ORDER - do not reorder or skip anything
- Clean up typos, grammar errors, and nonsensical sentences while preserving meaning
- Remove duplicates and filler words (um, uh, like) that don't affect meaning
- Identify and label speakers with their roles using the specified format
- Maintain consistent formatting style across all chunks
- Keep the sequential flow of the meeting

Transcript portion:
{chunk}"""
                
                completion = current_app.openai_client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                formatted_chunks.append(completion.choices[0].message.content)
                time.sleep(0.5)  # Small delay to avoid rate limiting
            
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
    Fetches from Supabase. Raises error if not configured.
    """
    return get_system_prompt(PROMPT_KEY_REPORT_GENERATION)


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
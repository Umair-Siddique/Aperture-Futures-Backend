from textwrap import dedent
from typing import Dict, List
from flask import current_app
from config import Config
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from blueprints.system_prompt import get_system_prompt, PROMPT_KEY_REPORT_GENERATION


def format_transcript_text(transcript: str) -> str:
    """
    Formats a raw transcript into properly structured markdown using GPT-4 mini.
    Only formats the text - does not add, remove, or modify content.
    """
    if not transcript or not transcript.strip():
        return transcript
    
    # Check if OpenAI client is properly initialized
    if not hasattr(current_app, 'openai_client') or current_app.openai_client is None:
        current_app.logger.warning("OpenAI client not initialized. Returning unformatted transcript.")
        return transcript
    
    system_prompt = """You are a text formatter. Your ONLY job is to format the raw transcript text into clean, readable markdown.

CRITICAL RULES:
- Use ONLY the original text provided - do NOT add, remove, or modify any content
- Do NOT add any explanatory text, headers, or summaries
- Do NOT invent speaker names or labels unless they are clearly in the original text
- Format with proper paragraph breaks, punctuation, and capitalization
- Use markdown formatting (bold, italics) only if it improves readability
- Preserve all original content exactly as provided

Your output should be the same content, just better formatted."""

    try:
        # For large transcripts, chunk them
        if len(transcript) > 15000:  # Smaller chunk size for formatting
            current_app.logger.info(f"Formatting large transcript ({len(transcript)} chars), using chunking")
            chunks = _chunk_transcript(transcript, chunk_size=15000)
            formatted_chunks = []
            
            for i, chunk in enumerate(chunks):
                current_app.logger.info(f"Formatting transcript chunk {i+1}/{len(chunks)}")
                
                user_prompt = f"""Format this portion of the transcript into clean markdown. Use only the original text - do not add or modify content.

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
            
            # Combine chunks with simple separator
            formatted_transcript = "\n\n".join(formatted_chunks)
        else:
            # Process small transcript directly
            user_prompt = f"""Format this transcript into clean markdown. Use only the original text - do not add or modify content.

{transcript}"""
            
            completion = current_app.openai_client.chat.completions.create(
                model="gpt-5-mini-2025-08-07",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],

            )
            formatted_transcript = completion.choices[0].message.content
        
        current_app.logger.info(f"Transcript formatting completed. Original: {len(transcript)} chars, Formatted: {len(formatted_transcript)} chars")
        return formatted_transcript
        
    except Exception as e:
        current_app.logger.error(f"Transcript formatting failed: {str(e)}. Returning original transcript.")
        # Return original transcript if formatting fails
        return transcript


def _build_system_prompt() -> str:
    """Return the instructions for formatting and structuring the report."""
    default_prompt = dedent("""
HUMANITARIAN
You are a UN policy and humanitarian analyst. Convert the raw UN Security Council transcript into a concise diplomatic report written for UN Missions, UN agencies, and humanitarian organisations.
Ensure all summaries reflect humanitarian substance (health, protection of civilians, humanitarian access, displacement, starvation/IHL violations, and operational constraints).

General Rules
Use clean Markdown
Diplomatic + humanitarian analytical tone
No procedural details
Use bold only for section headers
Do not invent facts

Grouping & Representation Rules
If a member state speaks on behalf of a group (e.g. A3+, NAM, EU, GCC), summarise that intervention as one consolidated bullet under the delivering country, formatted as:
*Algeria (for the A3+):* …
Do not repeat the same content separately under each state of that group.
Other individual national statements by A3+, if delivered separately, are then summarised individually.

Report Structure (unchanged except for bundling rule)
1) Executive Overview — 5 bullets
Prioritise briefers’ warnings and Member State divisions on humanitarian substance (access, PoC, health system collapse, starvation as a method, obstruction, ceasefire, sanctions impact).
2) Summary of Briefings
7–10 sentence paragraph per briefer, capturing facts, risks, humanitarian indicators, access constraints, and asks.
3) Member States
Organise by blocs:
P3 (US/UK/France)
Russia & China
A3+ (Algeria, Guyana, Sierra Leone, Somalia) — apply the bundling rule
Remaining E10
Within each bloc:
Begin with Shared Themes (humanitarian lens)
Then 2–4 sentences per country (3–4 lines each)
Where a bloc statement was delivered by one member on behalf of A3+: treat as a single entry
4) Observers / Invited States
4–6 sentences each — focus on new or distinct humanitarian content.
5) Overall Assessment
Short synthesis identifying:
Consensus or fault lines on humanitarian fundamentals
Which Council Members aligned or diverged
Implications for humanitarian operations or political trajectory

Output
Produce one English report only.
    """).strip()

    return get_system_prompt(PROMPT_KEY_REPORT_GENERATION, default_prompt)


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
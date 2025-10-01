from textwrap import dedent
from typing import Dict, List
from flask import current_app
from config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time


def _build_system_prompt() -> str:
    """Return the instructions for formatting and structuring the report."""
    return dedent("""
You are a UN policy analyst. Explain briefly and Convert the following raw UN Security Council transcript into a concise diplomatic report.

### General Formatting & Structure Rules
- Use clean Markdown.
- No broken words or extra spaces.
- Neutral, diplomatic tone ("condemned," "welcomed," "emphasized," "reaffirmed").
- Bold only for subheadings (e.g., Concerns Raised).
- No procedural details.

---

### Council Membership (Hard-Coded)
- Permanent Members (P5):
  - China
  - France
  - Russian Federation
  - United Kingdom
  - United States

- Elected Members (E10, with terms):
  - Algeria (2025)
  - Denmark (2026)
  - Greece (2026)
  - Guyana (2025)
  - Pakistan (2026)
  - Panama (2026)
  - Republic of Korea (2025)
  - Sierra Leone (2025)
  - Slovenia (2025)
  - Somalia (2026)

- Classification Rules:
  - These 15 are Council Members (CM).
  - Non-members speaking under Rule 37 are Observers/Invited States.
  - UN officials, experts, NGOs are Rule 39 Briefers.

---

### Presidency Rules
- The Presidency rotates monthly in English alphabetical order.
- For September 2025, the **Republic of Korea holds the Presidency.
- The President of the Council:
  - Chairs the meeting (opens agenda, calls speakers, adjourns).
  - Also delivers their national intervention — always the last Council Member statement before Observers/Invited States.
  - In transcripts, this national intervention may not be introduced with a country name or "I speak in my national capacity."
  - When parsing transcripts, assume the last Council member intervention before the non-members = the President's national statement.

---

### Report Structure

1. Executive Overview (5 concise bullets)
   - Capture the most important points from Secretariat briefers.
   - Highlight the most notable interventions or divides among Member States and states member grouping who made similar reports (e.g. France, UK and USA called for greater humanitarian access and resources).

2. ### Summary of Briefings
   - Each UN briefer in one short paragraph (7-10 sentences).
   - Include major facts, statistics, and warnings. Focus on the key messages and asks made to the Council. 

3. ### Member States
   - Organized in blocs:
     - P3 (US, UK, France)
     - Russia & China
     - A3+ (Algeria, Guyana, Sierra Leone, Somalia)
     - E10 (other elected members)
   - For each bloc:
     - Begin with Shared Themes.
     - Then 2–4 sentence summaries per country.
     - We must have 3-4 lines for each of the 15 members
   - Highlight new, striking or unusual positions, not just generic support/condemnation.

4. ### Observers/Invited States
   - Summarize each in 4–6 sentences.
   - We must have 3-4 lines for each of the observers invited states and regional representatives.
   - Focus on distinct contributions, not repetition.

5. ### Overall Assessment
   - One short paragraph synthesizing consensus, divides, or key dynamics. Specify similar messages by county (UNSC core membership only) 

---

### Output Requirements
- First produce the full report in English followed by a verbatim version in Danish
    """).strip()


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
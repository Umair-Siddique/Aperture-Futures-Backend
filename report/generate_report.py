from textwrap import dedent
from langchain_openai import ChatOpenAI  
from config import Config
from textwrap import dedent
from typing import Dict
from flask import current_app

def _build_un_report_prompt(transcript: str) -> str:
    """Return the exact prompt template you provided with the transcript appended."""
    return dedent(f"""
You are a UN policy analyst. Convert the following raw UN Security Council transcript into a concise diplomatic report.

### General Formatting & Structure Rules
- Use *clean Markdown*.
- No broken words or extra spaces.
- Neutral, diplomatic tone (“condemned,” “welcomed,” “emphasized,” “reaffirmed”).
- Bold only for subheadings (e.g., *Concerns Raised*).
- No procedural details.

---

### Council Membership (Hard-Coded)
- *Permanent Members (P5):*
  - China
  - France
  - Russian Federation
  - United Kingdom
  - United States

- *Elected Members (E10, with terms):*
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

- *Classification Rules:*
  - These 15 are *Council Members (CM)*.
  - Non-members speaking under Rule 37 are *Observers/Invited States*.
  - UN officials, experts, NGOs are *Rule 39 Briefers*.

---

### Presidency Rules
- The Presidency rotates monthly in English alphabetical order.
- For *September 2025, the **Republic of Korea* holds the Presidency.
- The President of the Council:
  - Chairs the meeting (opens agenda, calls speakers, adjourns).
  - Also delivers their *national intervention* — always the *last Council Member statement* before Observers/Invited States.
  - In transcripts, this national intervention may not be introduced with a country name or “I speak in my national capacity.”
  - When parsing transcripts, assume the *last Council member intervention before the non-members = the President’s national statement*.

---

### Report Structure

1. *Executive Overview* (5 concise bullets)
   - Capture the most important points from Secretariat briefers.
   - Highlight the most notable interventions or divides among Member States.

2. ### Summary of Briefings
   - Each UN briefer in one short paragraph (5–7 sentences).
   - Include major facts, statistics, and warnings.

3. ### Member States
   - Organized in blocs:
     - *P3 (US, UK, France)*
     - *Russia & China*
     - *A3+ (Algeria, Guyana, Sierra Leone, Somalia)*
     - *E10 (other elected members)*
   - For each bloc:
     - Begin with *Shared Themes*.
     - Then 2–4 sentence summaries per country.
   - Highlight *new, striking or unusual positions*, not just generic support/condemnation.

4. ### Observers/Invited States
   - Summarize each in 2–3 sentences.
   - Focus on distinct contributions, not repetition.

5. ### Overall Assessment
   - One short paragraph synthesizing consensus, divides, or key dynamics.

---

### Output Requirements
- First produce the *full report in English*.
    *Transcript:*
    {transcript}
    """).strip()


def generate_and_store_transcription_report(title: str, transcript: str) -> dict:
    """
    Generates a diplomatic report from the full transcript and stores it into
    audio_files.transcription_report for the given title. Returns dict with status.
    """
    # Safety guard: keep payload reasonable for the model
    # (adjust limit as needed depending on the model you use)
    MAX_CHARS = 180_000
    transcript_for_llm = transcript[:MAX_CHARS]

    prompt = _build_un_report_prompt(transcript_for_llm)

    # Use a stronger/more recent model if you like; fall back to what you already use
    llm = ChatOpenAI(
        openai_api_key=Config.OPENAI_API_KEY,
        model_name=getattr(Config, "TRANSCRIPTION_REPORT_MODEL", "gpt-4o-mini"),
        model='gpt-5'
    )

    resp = llm.invoke([
        {"role": "system", "content": "You are a neutral UN Security Council analyst producing concise diplomatic reports."},
        {"role": "user", "content": prompt}
    ])
    report = resp.content if hasattr(resp, "content") else resp["content"]

    # Store into Supabase
    current_app.supabase.table("audio_files") \
        .update({"transcription_report": report}) \
        .eq("title", title) \
        .execute()

    return {"ok": True, "chars_used": len(transcript_for_llm)}

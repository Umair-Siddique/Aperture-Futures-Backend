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

     *Instructions:* 1. Remove all procedural language (agenda adoption, rule references, etc.).

    2. Summarize *briefers* (UN officials) in 1–2 paragraphs each, keeping names, roles, and key issues. Include quotes only if striking. 

    3. Summarize *each Member State intervention* in 3–4 sentences. Use neutral, professional diplomatic language (“condemned,” “welcomed,” “emphasized”). Keep only major themes and notable phrases.

    4. Organize the Member States into blocs with the following classifications: - *P3: United States, United Kingdom, France - **Russia and China: Russian Federation, China - **A3+: Algeria (on behalf of Algeria, Guyana, Sierra Leone, Somalia) - **E10: All elected members of the Council *not in the A3+ (e.g. Denmark, Panama, Republic of Korea, Slovenia, Greece, Pakistan, etc.) - *Observers/Invited States*: Non-Council participants (e.g. Syria, Iran, Turkey, Tunisia, Norway) 

    5. For each bloc: - Begin with a short *“Shared Themes”* paragraph describing common positions. - Then provide 2–4 sentence *country-specific summaries*. 

    6. Structure the final report in this format: ---
    ### *Summary of Briefings*
    - [Briefer Name, Role]: [Summary]
    - [Briefer Name, Role]: [Summary]

    ### *Member States*
    *P3 (US, UK, France)*
    - Shared Themes: [summary]
    - [Country-specific summaries]

    *Russia and China*
    - Shared Themes: [summary]
    - [Country-specific summaries]

    *A3+ (Algeria on behalf of Algeria, Guyana, Sierra Leone, Somalia)*
    - Shared Themes: [summary]
    - [Country-specific summary]

    *E10 (Other elected members)*
    - Shared Themes: [summary]
    - [Country-specific summaries]

    ### *Observers/Invited States*
    - [Country-specific summaries]
    ---

     *Tone:*
     - Neutral, diplomatic, professional.
     - Concise: 3–5 sentences per speaker.
     - Avoid repetition and procedural details.
     - Focus on key themes, positions, and notable quotes.

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
        temperature=0.2,
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

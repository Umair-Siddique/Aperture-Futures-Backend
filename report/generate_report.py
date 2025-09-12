from textwrap import dedent
from typing import Dict
from flask import current_app
from config import Config
from groq import Groq


def _build_system_prompt() -> str:
    """Return the instructions for formatting and structuring the report."""
    return dedent("""
You are a UN policy analyst. Convert the following raw UN Security Council transcript into a concise diplomatic report. 

 *Instructions:* 1. Remove all procedural language (agenda adoption, rule references, etc.).

2. Summarize *briefers* (UN officials) in 1–2 paragraphs each, keeping names, roles, and key issues. Include quotes only if striking. 

3. Summarize *each Member State intervention* in 3–4 sentences. Use neutral, professional diplomatic language (“condemned,” “welcomed,” “emphasized”). Keep only major themes and notable phrases.

4. Organize the Member States into blocs with the following classifications: - *P3: United States, United Kingdom, France - **Russia and China: Russian Federation, China - **A3+: Algeria (on behalf of Algeria, Guyana, Sierra Leone, Somalia) - **E10: All elected members of the Council *not in the A3+ (e.g. Denmark, Panama, Republic of Korea, Slovenia, Greece, Pakistan, etc.) - *Observers/Invited States*: Non-Council participants (e.g. Syria, Iran, Turkey, Tunisia, Norway) 

5. For each bloc: - Begin with a short *“Shared Themes”* paragraph describing common positions. - Then provide 2–4 sentence *country-specific summaries*. 

6. Structure the final report in this format: --- ### *Summary of Briefings* - [Briefer Name, Role]: [Summary] - [Briefer Name, Role]: [Summary] ### *Member States* *P3 (US, UK, France)* - Shared Themes: [summary] - [Country-specific summaries] *Russia and China* - Shared Themes: [summary] - [Country-specific summaries] *A3+ (Algeria on behalf of Algeria, Guyana, Sierra Leone, Somalia)* - Shared Themes: [summary] - [Country-specific summary] *E10 (Other elected members)* - Shared Themes: [summary] - [Country-specific summaries] ### *Observers/Invited States* - [Country-specific summaries] --- 

 *Tone:* - Neutral, diplomatic, professional. - Concise: 3–5 sentences per speaker. - Avoid repetition and procedural details. - Focus on key themes, positions, and notable quotes. 

    """).strip()


def generate_and_store_transcription_report(title: str, transcript: str) -> dict:
    """
    Generates a diplomatic report from the full transcript using Groq Cloud
    and stores it into audio_files.transcription_report for the given title.
    """
    
    # Check if Groq client is properly initialized
    if not hasattr(current_app, 'groq') or current_app.groq is None:
        raise ValueError("Groq client not properly initialized. Check GROQ_API_KEY in .env file.")

    system_prompt = _build_system_prompt()
    
    try:
        completion = current_app.groq.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript}
            ],
            temperature=0.6,
            max_completion_tokens=6000
        )
    except Exception as e:
        if "api_key" in str(e).lower() or "authentication" in str(e).lower():
            raise ValueError(f"Groq API authentication failed. Check your GROQ_API_KEY in .env file. Error: {str(e)}")
        else:
            raise ValueError(f"Groq API error: {str(e)}")

    report = completion.choices[0].message.content

    # Store into Supabase
    current_app.supabase.table("audio_files") \
        .update({"transcription_report": report}) \
        .eq("title", title) \
        .execute()

    return {"ok": True, "chars_used": len(transcript)}
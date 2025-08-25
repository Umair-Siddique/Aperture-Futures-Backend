from flask import Blueprint, request, jsonify, current_app, send_file
import tempfile
import os

report_bp = Blueprint('report', __name__)

@report_bp.route("/download", methods=["GET"])
def download_report():
    title = request.args.get("title")
    if not title:
        return jsonify({"error": "Title is required"}), 400

    # --- Fetch report from Supabase ---
    rec = (
        current_app.supabase.table("audio_files")
        .select("transcription_report")
        .eq("title", title)
        .limit(1)
        .execute()
    )

    if not rec.data:
        return jsonify({"error": f"No record found for title '{title}'"}), 404

    report_md = rec.data[0].get("transcription_report")
    if not report_md:
        return jsonify({"error": "No transcription_report stored for this title"}), 404

    # --- Save report as temporary .txt file ---
    tmp_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    tmp_txt.write(report_md)
    tmp_txt.close()

    # --- Return file for download ---
    return send_file(
        tmp_txt.name,
        as_attachment=True,
        download_name=f"{title}_report.txt",
        mimetype="text/plain"
    )

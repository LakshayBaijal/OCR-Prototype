"""Before/after verification for AI-enhanced product images. OCR-only.

A generative enhancer does not filter a photo, it redraws one - so every glyph in its
output was re-rendered from what the model believed it saw. This package answers the
one question that follows: did the enhanced image keep the original's facts?

Nothing here is generative. RapidOCR reads the pixels; everything else is deterministic
comparison. Same pair in, same answer out - which is what makes the result evidence.

    from verify import extract, report

    original = extract.load_rgb_bytes(open("a.jpg", "rb").read())
    enhanced = extract.load_rgb_bytes(open("b.jpg", "rb").read())
    rep = report.run_verification(original, enhanced)
    print(rep.verdict, rep.reasons)

Module map:
    extract.py    crop to the product, prepare, and read it with RapidOCR
    signals.py    what the pixels say, as comparable SETS (digits, quantities, words)
    labeled.py    the four fields a pack LABELS explicitly - FSSAI, MRP, net qty, pack
    artwork.py    ORB feature match + hue histogram - "is this the same design?"
    report.py     runs it all, reduces it to PASS / REVIEW / REJECT
    enhance.py    the thing being verified (Gemini image-to-image; the only cloud call,
                  and only in the app\'s "Enhance it now" mode)
    _retired/     the VLM-era modules, kept for reference, imported by nothing

Run the UI with:  venv/bin/streamlit run verify_app.py
"""

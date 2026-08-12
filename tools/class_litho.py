import pandas as pd


def classify_broad_lithology(*parts):
    """
    Classify lithology based on text description.
    Generate "display label."
    """
    text = " ".join(
        str(part).strip().lower()
        for part in parts
        if pd.notna(part) and str(part).strip()
    )
    if text == "":
        return "Unknown"
    if "no recovery" in text or "no return" in text:
        return "No recovery"
    if "bentonite" in text or "bent" in text:
        return "Bentonite"
    if "carbonaceous" in text or "carb" in text:
        return "Carbonaceous"
    if "coal" in text:
        return "Coal"
    if "till" in text or "overburden" in text or "topsoil" in text:
        return "Till/Overburden"
    if "clay" in text:
        return "Clay"
    if "sandstone" in text or "sand" in text or "ss" in text:
        return "Sand/Sandstone"
    if "siltstone" in text or "silty" in text or "silt" in text or "slt" in text:
        return "Silt/Siltstone"
    if "mudstone" in text or "shale" in text or "sh" in text:
        return "Shale/Mudstone"
    return "Other"
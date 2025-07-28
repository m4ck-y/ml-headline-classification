def normalize_name_category(cat: str):
    parts = cat.split('&')
    parts = [p.strip() for p in parts]
    parts.sort()
    return ' & '.join(parts)
from dynamic_kg import fetch_text_from_sources

print(">>> Testing Wikipedia Fetching")

texts = fetch_text_from_sources("bipolar disorder")

if not texts:
    print("NO TEXT RETURNED")
else:
    print("Returned keys:", list(texts.keys()))
    print("Wikipedia text length:", len(texts.get("wikipedia", "")))
    print(texts.get("wikipedia", "")[:500], "...")

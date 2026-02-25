import json

with open("articles.json") as f:
    articles = json.load(f)

print(f"Total articles: {len(articles)}")
print(f"\n--- Sample titles ---")
for a in articles[:5]:
    print(f"  {a['title']}")

print(f"\n--- Content length stats ---")
lengths = [len(a['content']) for a in articles]
print(f"  Min:  {min(lengths)} chars")
print(f"  Max:  {max(lengths)} chars")
print(f"  Avg:  {int(sum(lengths)/len(lengths))} chars")

print(f"\n--- First article preview ---")
print(articles[0]['content'][:500])
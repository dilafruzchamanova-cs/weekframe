from sentence_transformers import SentenceTransformer, util

#loading a small embedding model 
model = SentenceTransformer("all-MiniLM-L6-v2")

# define  categories and  example phrases for each
CATEGORY_EXAMPLES = {
    "sleep": [
        "sleep", "nap", "rest", "went to bed", "slept early", "slept late"
    ],
    "study": [
        "studied", "read textbook", "homework", "coding", "lecture notes", "learning", "course"
    ],
    "exercise": [   
        "gym", "workout", "running", "dance practice", "sports", "yoga", "walk"
    ],
    "eating": [
        "breakfast", "lunch", "dinner", "ate", "snack", "food"
    ],
    "self-care": [
        "shower", "skincare", "meditation", "cleaning room"
    ],
    "leisure-positive": [
        "played guitar", "learned new skill", "creative hobby", "painting", "writing"
    ],
    "leisure-negative": [
        "scrolled on phone", "tiktok", "instagram", "doomscrolling", "mindless youtube", "Scrolling"
    ],
    "social": [
        "hung out", "talk" ,"friends", "call parents", "went out"
    ]
}

example_embeddings = []
example_labels = []

for category, phrases in CATEGORY_EXAMPLES.items():
    for p in phrases:
        example_embeddings.append(p)
        example_labels.append(category)

example_embeddings = model.encode(example_embeddings, convert_to_tensor=True)

def classify_activity(text: str) -> str:
    
    # Returns the best category for the given activity description
    
    input_emb = model.encode(text, convert_to_tensor=True)
    sims = util.cos_sim(input_emb, example_embeddings)[0]

    best_index = sims.argmax().item()
    return example_labels[best_index]

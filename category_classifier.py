import os
from groq import Groq

CATEGORIES = [
    "sleep",
    "study",
    "exercise",
    "eating",
    "self-care",
    "leisure-positive",
    "leisure-negative",
    "social",
]

CATEGORY_DESCRIPTIONS = """
- sleep: sleeping, napping, resting
- study: studying, homework, coding, reading for learning, lectures, courses
- exercise: gym, workout, running, sports, yoga, walking, dance
- eating: any meal, snack, food, coffee
- self-care: shower, skincare, meditation, cleaning, personal errands, chores
- leisure-positive: creative hobbies, reading for fun, playing music, painting, journaling
- leisure-negative: doom scrolling, tiktok, instagram, mindless youtube, passive phone use
- social: hanging out with friends, calls with family, going out, socializing
"""


def classify_activity(text: str, groq_client: Groq = None) -> str:
    if groq_client is None:
        groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    prompt = f"""You are a time tracking classifier. Given an activity description, return exactly one category from the list below. Return only the category name, nothing else.

Categories:
{CATEGORY_DESCRIPTIONS}

Activity: "{text}"

Return one of: {', '.join(CATEGORIES)}"""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = response.choices[0].message.content.strip().lower()

    # Validate the response is actually one of the categories
    for cat in CATEGORIES:
        if cat in result:
            return cat

    # Fallback if model returns something unexpected
    return "self-care"

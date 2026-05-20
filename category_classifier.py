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
    "work",
    "commute",
    "chores",
    "health",
]

CATEGORY_DESCRIPTIONS = """
- sleep: sleeping, napping, resting
- study: studying, homework, coding, reading for learning, lectures, courses, university work
- exercise: gym, workout, running, sports, yoga, walking, dance
- eating: any meal, snack, food, coffee
- self-care: shower, skincare, meditation, journaling, personal grooming
- leisure-positive: creative hobbies, reading for fun, playing music, painting, intentional entertainment
- leisure-negative: doom scrolling, tiktok, instagram, mindless youtube, passive phone use, browsing aimlessly
- social: hanging out with friends, calls with family, going out, socializing, parties
- work: job, internship, freelance, professional tasks not related to studying
- commute: travelling, driving, public transport, getting somewhere
- chores: cleaning, errands, groceries, laundry, cooking, tidying
- health: doctor, therapy, pharmacy, medical appointments
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
        model="llama-3.3-70b-versatile",
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

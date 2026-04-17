# WeekFrame

An AI-powered time tracker inspired by Laura Vanderkam's book *168 Hours: You Have More Time Than You Think*.

WeekFrame lets you log your activities, auto-classifies them into meaningful categories using sentence embeddings, and gives you weekly reflections on where your 168 hours actually went.

## Why I built this

I have struggled with time management for as long as I can remember. Reading *168 Hours* changed how I think about time, but most tracking apps either feel like productivity surveillance or miss the point entirely. I wanted something that logs quickly, thinks about my week, and tells me interesting things, not just numbers. So I built it.

## What it does

- Logs activities with a start time, end time, and a short note
- Auto-suggests a category using a sentence-transformer model (all-MiniLM-L6-v2) and cosine similarity over example phrases
- Stores data locally in a JSON file
- Computes weekly insights, including total hours per category, percentages, and days with low activity logging
- Compares this week against last week, flags categories with significant drops

## Categories

sleep, study, exercise, eating, self-care, leisure-positive, leisure-negative, social

The split between leisure-positive and leisure-negative is intentional. Creative hobbies and doom-scrolling are both leisure, but they do very different things to a week.

## Architecture

```
main.py                   CLI entrypoint and user prompts
Activity.py               data container for a single activity
Activity_Manager.py       persistence layer, reads and writes JSON
Activity_Analyzer.py      pandas-powered weekly analytics
category_classifier.py    sentence-embedding classifier
```

## Tech stack

- Python 3.10+
- pandas for weekly aggregation and week-over-week comparison
- sentence-transformers for semantic category prediction
- JSON for local persistence

## How to run it

```
git clone https://github.com/<your-username>/weekframe.git
cd weekframe
pip install -r requirements.txt
python main.py
```

On the first run, the app will create a `User_logs.json` file in the project folder as you log activities. A `User_logs_example.json` is included so you can see the schema.

## Roadmap

This is v0.1, a working CLI prototype. Planned next:

- Streamlit web UI so logging works from a phone, not just a terminal
- LLM-generated weekly reflections, not just stats
- Absence detection, for example "you have not logged exercise in 4 days"
- Time-of-day pattern analysis, such as when doom-scrolling tends to happen
- Voice logging via Whisper
- Weekly email digest every Sunday

## What I learned

- Separating data, persistence, and analysis into three classes made the codebase easy to extend. I added the classifier as a fourth, independent module without touching the others.
- Overnight activities are a pain. Handling end time less than start time needed a special case in duration math.
- Semantic similarity with sentence embeddings beats keyword matching for short notes like "doom scrolled on my phone."
- Building something you actually use changes the feature priorities. Most things on the roadmap came from being annoyed while logging my own days.

## Built by

Dilafruz Chamanova, originally as a project for SCCI1, now being iterated on as a personal tool.

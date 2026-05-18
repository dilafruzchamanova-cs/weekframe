import os
import json
import logging
from datetime import date, datetime, timedelta
import pandas as pd
from groq import Groq
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, ContextTypes
)
from Activity_Manager import Activity_Manager
from Activity_Analyzer import Activity_Analyzer
from category_classifier import classify_activity

logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
CONFIG_FILE = "user_config.json"

groq_client = Groq(api_key=GROQ_API_KEY)
manager = Activity_Manager()
analyzer = Activity_Analyzer(manager)

ALL_CATEGORIES = [
    "exercise", "eating", "leisure-positive", "leisure-negative",
    "study", "sleep", "social", "self-care"
]


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_config(data: dict):
    config = load_config()
    config.update(data)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_name() -> str:
    return load_config().get("name", "bestie")


def is_onboarded() -> bool:
    return bool(load_config().get("name"))


def mark_logged():
    save_config({"last_logged": str(date.today())})


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_activity_from_text(text: str) -> dict | None:
    now = datetime.now().strftime("%I:%M %p")

    prompt = f"""Extract activity details from this message. Return JSON only, no explanation.

Current time is {now}. Use this if the person says "just now", "ending now", or "for X hours" with no start time.
If they say "for X hours" with no start time, set end_time to current time and subtract to get start_time.

Message: "{text}"

Return exactly this format:
{{
  "start_time": "HH:MM AM/PM",
  "end_time": "HH:MM AM/PM",
  "note": "brief description of what they did"
}}

If you truly cannot extract any time information at all, return the word null.
Only return valid JSON or the word null. No markdown, no code blocks."""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        content = response.choices[0].message.content.strip()
        if content.lower() == "null":
            return None
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception:
        return None


# ── Gap detection ─────────────────────────────────────────────────────────────

def get_unlogged_days() -> list[str]:
    df = analyzer.prepare_data()
    today = pd.Timestamp(date.today())
    unlogged = []

    for i in range(1, 8):
        day = today - pd.Timedelta(days=i)
        if df.empty:
            unlogged.append(day.strftime("%b %d"))
            continue
        day_data = df[df["date"].dt.date == day.date()]
        if day_data.empty:
            unlogged.append(day.strftime("%b %d"))
        elif day_data["duration_hours"].sum() < 3:
            h = day_data["duration_hours"].sum()
            unlogged.append(f"{day.strftime('%b %d')} ({h:.1f}h logged)")

    return unlogged


# ── Absence callout ───────────────────────────────────────────────────────────

def get_absence_callout() -> str | None:
    df = analyzer.prepare_data()
    if df.empty:
        return None

    today = pd.Timestamp(date.today())
    watch = {"exercise": 3, "study": 2, "self-care": 3}
    callouts = []

    for cat, threshold in watch.items():
        cat_df = df[df["category"] == cat]
        if cat_df.empty:
            callouts.append(f"zero {cat} logged ever")
            continue
        days_since = (today - cat_df["date"].max()).days
        if days_since >= threshold:
            callouts.append(f"no {cat} in {days_since} days")

    if not df.empty:
        weeks = sorted(df["week"].unique())
        if weeks:
            df_week = df[df["week"] == weeks[-1]]
            neg = df_week[df_week["category"] == "leisure-negative"]["duration_hours"].sum()
            if neg >= 10:
                callouts.append(f"{neg:.0f}h of doom scrolling this week")

    if not callouts:
        return None

    name = get_name()
    prompt = f"""You are {name}'s blunt, sarcastic Gen Z time coach.
Turn these observations into ONE punchy message. Max 2 sentences.
Casual tone. Use a brainrot word or phrase only if it fits naturally, never force it.
Be specific about the data. Mean but caring.

Observations: {", ".join(callouts)}"""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return response.choices[0].message.content.strip()


# ── Weekly reflection ─────────────────────────────────────────────────────────

def get_weekly_reflection() -> str:
    df = analyzer.prepare_data()
    name = get_name()

    if df.empty:
        return f"no data yet {name}, start logging fr"

    weeks = sorted(df["week"].unique())
    df_week = df[df["week"] == weeks[-1]]

    total_hours = (
        df_week.groupby("category")["duration_hours"]
        .sum().reindex(ALL_CATEGORIES, fill_value=0).round(2)
    )
    total_time = total_hours.sum()

    stats = "\n".join([
        f"{cat}: {total_hours[cat]:.1f}h ({total_hours[cat] / total_time * 100:.0f}%)"
        if total_time > 0 else f"{cat}: 0h"
        for cat in ALL_CATEGORIES
    ])

    gaps = get_unlogged_days()
    gap_text = f"Unlogged or low-activity days: {', '.join(gaps)}" if gaps else "No missing days."

    prompt = f"""You are {name}'s brutally honest, sarcastic Gen Z time coach doing a weekly review.

Rules:
- Use their name ({name}) once or twice naturally, not every sentence
- Be specific about the actual numbers, not vague platitudes
- Call out bad habits directly (doom scrolling, zero exercise, etc.) with a bit of humor
- If something is at 0 hours, make it a point
- Add ONE genuine motivational line somewhere in the middle where it fits naturally
- Gen Z / brainrot tone but natural, never cringe or forced
- Comment on the gap days if any
- End with one specific actionable thing they should do differently next week
- Max 200 words

Data:
{stats}

{gap_text}

Benchmark: healthy week is roughly 56h sleep, regular exercise, real study time, conscious leisure."""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return response.choices[0].message.content.strip()


# ── All-time summary ──────────────────────────────────────────────────────────

def get_alltime_summary() -> str:
    df = analyzer.prepare_data()
    name = get_name()

    if df.empty:
        return f"nothing logged yet {name}, get on it"

    total_hours = (
        df.groupby("category")["duration_hours"]
        .sum().reindex(ALL_CATEGORIES, fill_value=0).round(1)
    )
    grand_total = total_hours.sum()
    first_date = df["date"].min().strftime("%b %d, %Y")
    weeks_tracked = max(1, len(df["week"].unique()))

    lines = [f"since {first_date} — {weeks_tracked} weeks, {grand_total:.0f}h total:\n"]
    for cat in ALL_CATEGORIES:
        h = total_hours[cat]
        pct = (h / grand_total * 100) if grand_total > 0 else 0
        lines.append(f"  {cat}: {h:.1f}h ({pct:.1f}%)")

    stats = "\n".join([
        f"{cat}: {total_hours[cat]:.1f}h ({total_hours[cat] / grand_total * 100:.1f}%)"
        if grand_total > 0 else f"{cat}: 0h"
        for cat in ALL_CATEGORIES
    ])

    prompt = f"""You are {name}'s brutally honest Gen Z time coach reviewing their ALL TIME data.
They've been tracking since {first_date} — {weeks_tracked} weeks, {grand_total:.0f} total hours logged.

Write a punchy all-time verdict. Be honest about what these numbers say about their life patterns.
Use their name once. Sarcastic where it fits. Add one motivational line. Max 150 words.

All-time:
{stats}"""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )

    return "\n".join(lines) + "\n\n" + response.choices[0].message.content.strip()


# ── Handlers ──────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    save_config({"chat_id": chat_id})

    if not is_onboarded():
        context.user_data["awaiting_name"] = True
        await update.message.reply_text(
            "yo, welcome to WeekFrame.\n\n"
            "laura vanderkam said you have 168 hours a week. "
            "this bot helps you see where they actually go "
            "and calls you out when the answer is embarrassing.\n\n"
            "first — what's your name?"
        )
    else:
        name = get_name()
        await update.message.reply_text(
            f"wb {name}.\n\n"
            "just tell me what you did:\n"
            "  'studied from 2pm to 5pm'\n"
            "  'gym from 7am to 8am'\n"
            "  'doom scrolled from 10pm to midnight'\n\n"
            "/insights — this week's breakdown\n"
            "/reflection — weekly AI reflection\n"
            "/alltime — everything since you started\n"
            "/compare — this week vs last week"
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text

    if context.user_data.get("awaiting_name"):
        name = text.strip().split()[0].capitalize()
        save_config({"name": name})
        context.user_data["awaiting_name"] = False

        await update.message.reply_text(
            f"let's go {name}.\n\n"
            "here's how it works:\n\n"
            "just tell me what you did and when:\n"
            "  'studied from 2pm to 5pm'\n"
            "  'gym from 7am to 8am'\n"
            "  'watched netflix from 9pm to 11pm'\n"
            "  'ran for 1 hour just now'\n\n"
            "i'll figure out the category, log it, and keep track.\n\n"
            "commands:\n"
            "/insights — hourly breakdown with percentages\n"
            "/reflection — honest weekly AI reflection\n"
            "/alltime — your full history since day one\n"
            "/compare — this week vs last week\n\n"
            "i'll also ping you if you go quiet for too long. no ghosting your own data."
        )
        return

    await update.message.reply_text("logging...")

    parsed = parse_activity_from_text(text)

    if not parsed:
        await update.message.reply_text(
            "couldn't parse the time from that.\n"
            "try: 'watched netflix from 9pm to 11pm'\n"
            "or: 'ran for 1 hour just now'"
        )
        return

    note = parsed.get("note", text)
    start_time = parsed.get("start_time")
    end_time = parsed.get("end_time")
    category = classify_activity(note, groq_client)

    manager.add_activity(category, start_time, end_time, note)
    manager.write_data()
    mark_logged()

    await update.message.reply_text(
        f"logged.\ncategory: {category}\n{start_time} to {end_time}\n\"{note}\""
    )

    callout = get_absence_callout()
    if callout:
        await update.message.reply_text(callout)


async def insights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = analyzer.prepare_data()
    name = get_name()

    if df.empty:
        await update.message.reply_text(f"no data yet {name}, start logging")
        return

    weeks = sorted(df["week"].unique())
    df_week = df[df["week"] == weeks[-1]]

    total_hours = (
        df_week.groupby("category")["duration_hours"]
        .sum().reindex(ALL_CATEGORIES, fill_value=0).round(2)
    )
    total_time = total_hours.sum()

    lines = [f"this week {name} (and u said u have no time huh):\n"]
    for cat in ALL_CATEGORIES:
        h = total_hours[cat]
        pct = (h / total_time * 100) if total_time > 0 else 0
        bar = "█" * int(pct / 5)
        lines.append(f"{cat}: {h:.1f}h ({pct:.0f}%) {bar}")

    gaps = get_unlogged_days()
    if gaps:
        lines.append(f"\nunlogged / low-activity days:\n  {chr(10).join(gaps)}")
        lines.append("\nhm what were u doing on those days exactly")

    await update.message.reply_text("\n".join(lines))


async def reflection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("cooking ur reflection...")
    await update.message.reply_text(get_weekly_reflection())


async def alltime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pulling your whole life together...")
    await update.message.reply_text(get_alltime_summary())


async def compare(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = analyzer.prepare_data()
    name = get_name()

    if df.empty:
        await update.message.reply_text("no data yet")
        return

    weeks = sorted(df["week"].unique())
    if len(weeks) < 2:
        await update.message.reply_text("need at least 2 weeks of data to compare")
        return

    df_last = df[df["week"] == weeks[-2]]
    df_curr = df[df["week"] == weeks[-1]]

    last_h = df_last.groupby("category")["duration_hours"].sum().reindex(ALL_CATEGORIES, fill_value=0).round(2)
    curr_h = df_curr.groupby("category")["duration_hours"].sum().reindex(ALL_CATEGORIES, fill_value=0).round(2)

    lines = [f"this week vs last week {name}:\n"]
    for cat in ALL_CATEGORIES:
        diff = curr_h[cat] - last_h[cat]
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        sign = "+" if diff >= 0 else ""
        lines.append(f"{cat}: {curr_h[cat]:.1f}h {arrow} ({sign}{diff:.1f}h)")

    await update.message.reply_text("\n".join(lines))


# ── Scheduled reminder ────────────────────────────────────────────────────────

async def send_reminder(context: ContextTypes.DEFAULT_TYPE):
    config = load_config()
    chat_id = config.get("chat_id")
    name = config.get("name", "bestie")
    last_logged = config.get("last_logged")

    if not chat_id:
        return

    today = str(date.today())
    if last_logged == today:
        return

    days = 1
    if last_logged:
        days = (date.today() - date.fromisoformat(last_logged)).days

    if days < 1:
        return

    prompt = f"""You are {name}'s sarcastic Gen Z time coach.
They haven't logged in {days} day(s). Write a short reminder (1-2 sentences max).
Be a little annoying in a funny way. Use their name. Natural gen z tone, not forced.
Do not use emojis."""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9
    )

    await context.bot.send_message(
        chat_id=chat_id,
        text=response.choices[0].message.content.strip()
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN not set")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("insights", insights))
    app.add_handler(CommandHandler("reflection", reflection))
    app.add_handler(CommandHandler("alltime", alltime))
    app.add_handler(CommandHandler("compare", compare))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Daily reminder at 8pm — only fires if you haven't logged today
    app.job_queue.run_daily(
        send_reminder,
        time=datetime.strptime("20:00", "%H:%M").time()
    )

    print("WeekFrame bot is running.")
    app.run_polling()

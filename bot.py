import os
import io
import json
import logging
from datetime import date, datetime, timedelta
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

from telegram import Update, BotCommand
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

DEFAULT_CATEGORIES = [
    "sleep", "study", "exercise", "eating", "self-care",
    "leisure-positive", "leisure-negative", "social",
    "work", "commute", "chores", "health"
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
    mark_first_logged()


# ── Active sessions ───────────────────────────────────────────────────────────

def get_active_sessions() -> dict:
    return load_config().get("active_sessions", {})


def save_active_session(key: str, data: dict):
    config = load_config()
    sessions = config.get("active_sessions", {})
    sessions[key] = data
    config["active_sessions"] = sessions
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def clear_active_session(key: str):
    config = load_config()
    sessions = config.get("active_sessions", {})
    sessions.pop(key, None)
    config["active_sessions"] = sessions
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


# ── Intent detection ──────────────────────────────────────────────────────────

def detect_intent(text: str, active_sessions: dict) -> dict:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    active_list = list(active_sessions.keys()) if active_sessions else []

    prompt = f"""You are parsing a time-tracking message. Detect the intent and return JSON only.

Current datetime: {now}
Currently active (unfinished) sessions: {active_list}

Message: "{text}"

Possible intents:
- "start": user is beginning an activity with no known end (no duration given)
- "end": user is finishing an activity that was previously started
- "regular": completed activity — either explicit start+end times OR a duration was given (e.g. "read for 10 mins", "ate for 20 minutes")
- "unknown": cannot determine

Return exactly:
{{
  "intent": "start" | "end" | "regular" | "unknown",
  "activity": "short description of the activity",
  "start_time": "HH:MM AM/PM or null",
  "end_time": "HH:MM AM/PM or null",
  "duration_minutes": 0,
  "matched_session": "key from active sessions this end refers to, or null"
}}

Rules:
- "read for 10 mins", "ate for 20 minutes", "napped for 30 mins" → intent is "regular", compute end_time as now, start_time as now minus duration, set duration_minutes
- "started reading", "going for a run", "hopping in the shower", "starting X", "about to X", "beginning X" → intent is ALWAYS "start", even if there are active sessions. Never treat these as "end".
- "finished reading", "done with my run", "just wrapped up" → intent is "end"
- CRITICAL: If the message contains "started", "starting", "about to", "hopping in", "beginning", "gonna", "going to" referring to a new activity — intent MUST be "start". Do not close active sessions based on a start message.
- For "start": start_time is now unless specified, end_time is null
- For "end": end_time is now unless specified. Only set matched_session if the user explicitly named or clearly described a specific activity that matches one in the active sessions list. If the user just says "finished" or "done" with no activity name, set matched_session to null.
- For "regular" with duration: calculate start_time and end_time from current time
- Only return valid JSON, no markdown."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        content = response.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception:
        return {"intent": "unknown"}


def parse_activity_from_text(text: str) -> dict | None:
    now = datetime.now().strftime("%I:%M %p")

    prompt = f"""Extract activity details from this message. Return JSON only, no explanation.

Current time is {now}.
If they say "for X hours" with no start time, set end_time to current time and subtract to get start_time.

Message: "{text}"

Return exactly:
{{
  "start_time": "HH:MM AM/PM",
  "end_time": "HH:MM AM/PM",
  "note": "brief description of what they did"
}}

If you cannot extract time info, return the word null.
Only return valid JSON or null. No markdown."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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

def has_enough_data_for_callouts() -> bool:
    config = load_config()
    first_log = config.get("first_logged")
    if not first_log:
        return False
    days_since_start = (date.today() - date.fromisoformat(first_log)).days
    return days_since_start >= 7


def mark_first_logged():
    config = load_config()
    if not config.get("first_logged"):
        save_config({"first_logged": str(date.today())})


def get_absence_callout() -> str | None:
    if not has_enough_data_for_callouts():
        return None

    df = analyzer.prepare_data()
    if df.empty:
        return None

    today = pd.Timestamp(date.today())
    watch = {"exercise": 3, "study": 2, "self-care": 3}
    callouts = []

    for cat, threshold in watch.items():
        cat_df = df[df["category"] == cat]
        if cat_df.empty:
            continue  # never logged it — might just not track it, don't assume
        days_since = (today - cat_df["date"].max()).days
        if days_since >= threshold:
            callouts.append(f"haven't logged {cat} in {days_since} days")

    if not df.empty:
        weeks = sorted(df["week"].unique())
        if weeks:
            df_week = df[df["week"] == weeks[-1]]
            neg = df_week[df_week["category"] == "leisure-negative"]["duration_hours"].sum()
            if neg >= 8:
                callouts.append(f"{neg:.0f}h of leisure-negative logged this week")

    if not callouts:
        return None

    name = get_name()
    prompt = f"""You are {name}'s honest time coach.
Write ONE short observation (max 2 sentences) based only on these logging patterns.
Be direct and factual. Do not assume they haven't done the activity, only that they haven't logged it.
Casual tone, no preaching.

Observations: {", ".join(callouts)}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def get_unfinished_nudge(exclude_key: str = None) -> tuple[str | None, dict | None]:
    sessions = get_active_sessions()
    remaining = {k: v for k, v in sessions.items() if k != exclude_key}
    if not remaining:
        return None, None
    if len(remaining) == 1:
        key = list(remaining.keys())[0]
        note = remaining[key].get("note", key)
        return (
            f"btw, have you finished {note}?",
            {"type": "single", "key": key, "note": note}
        )
    names = [v.get("note", k) for k, v in remaining.items()]
    keys = list(remaining.keys())
    return (
        f"btw, still open: {', '.join(names)}.\nfinished any? reply with the activity name.",
        {"type": "multi", "keys": keys, "notes": names}
    )


# ── Weekly reflection ─────────────────────────────────────────────────────────

def get_weekly_reflection() -> str:
    df = analyzer.prepare_data()
    name = get_name()

    if df.empty:
        return f"no data yet {name}, start logging fr"

    weeks = sorted(df["week"].unique())
    df_week = df[df["week"] == weeks[-1]]

    total_hours = df_week.groupby("category")["duration_hours"].sum().round(2)
    total_time = total_hours.sum()

    stats = "\n".join([
        f"{cat}: {h:.1f}h ({h / total_time * 100:.0f}%)" if total_time > 0 else f"{cat}: 0h"
        for cat, h in total_hours.items()
    ])

    gaps = get_unlogged_days()
    gap_text = f"Unlogged or low-activity days: {', '.join(gaps)}" if gaps else "No missing days."

    prompt = f"""You are {name}'s honest, direct time coach doing a weekly review.

Rules:
- Use their name ({name}) once or twice naturally
- Be specific about the actual numbers in the data
- Only call out habits that are actually visible in the data, never assume
- If something is at 0 hours, note it without dramatizing
- Add ONE genuine motivational line where it fits naturally
- Casual and smart tone, not preachy, not full of internet slang
- Comment on gap days if any
- End with one specific actionable suggestion for next week
- Max 200 words

Data:
{stats}

{gap_text}

Benchmark: healthy week is roughly 56h sleep, regular exercise, real study time, conscious leisure."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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

    total_hours = df.groupby("category")["duration_hours"].sum().round(1).sort_values(ascending=False)
    grand_total = total_hours.sum()
    first_date = df["date"].min().strftime("%b %d, %Y")
    weeks_tracked = max(1, len(df["week"].unique()))

    lines = [f"since {first_date} — {weeks_tracked} weeks, {grand_total:.0f}h total:\n"]
    for cat, h in total_hours.items():
        pct = (h / grand_total * 100) if grand_total > 0 else 0
        lines.append(f"  {cat}: {h:.1f}h ({pct:.1f}%)")

    stats = "\n".join([
        f"{cat}: {h:.1f}h ({h / grand_total * 100:.1f}%)" if grand_total > 0 else f"{cat}: 0h"
        for cat, h in total_hours.items()
    ])

    prompt = f"""You are {name}'s brutally honest Gen Z time coach reviewing their ALL TIME data.
They've been tracking since {first_date} — {weeks_tracked} weeks, {grand_total:.0f} total hours logged.

Write a punchy all-time verdict. Be honest about what these numbers say about their life patterns.
Use their name once. Sarcastic where it fits. Add one motivational line. Max 150 words.

All-time:
{stats}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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
            f"hey {name}.\n\n"
            "just tell me what you did or are doing:\n"
            "  'started reading'\n"
            "  'studied from 2pm to 5pm'\n"
            "  'slept for 8 hours'\n\n"
            "type /help to see all commands."
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
            "two ways to log:\n\n"
            "start/stop (recommended):\n"
            "  'started reading' → i clock the start\n"
            "  'just finished reading' → i calculate the time\n"
            "  'took a 20min break' → i subtract it\n\n"
            "or give me both times directly:\n"
            "  'studied from 2pm to 5pm'\n\n"
            "commands:\n"
            "/active — see what's currently running\n"
            "/insights — this week's breakdown\n"
            "/reflection — weekly AI reflection\n"
            "/alltime — your full history\n"
            "/compare — this week vs last week\n\n"
            "i'll ping you at 8pm if you haven't logged anything."
        )
        return

    # ── Pending nudge reply (yes/no or activity name) ─────────────────────────
    pending = context.user_data.get("pending_nudge") or load_config().get("pending_nudge")
    if pending:
        t = text.strip().lower()
        if pending["type"] == "single":
            if t in ("yes", "yeah", "yep", "yup", "y", "done", "finished", "yes!"):
                context.user_data.pop("pending_nudge", None)
                save_config({"pending_nudge": None})
                key = pending["key"]
                session = get_active_sessions().get(key)
                if session:
                    end_time = datetime.now().strftime("%I:%M %p")
                    start_time = session["start_time"]
                    break_min = session.get("break_minutes", 0)
                    note = session["note"]
                    if break_min > 0:
                        try:
                            end_dt = datetime.strptime(end_time, "%I:%M %p")
                            end_dt -= timedelta(minutes=break_min)
                            end_time = end_dt.strftime("%I:%M %p")
                        except Exception:
                            pass
                    category = classify_activity(note, groq_client)
                    manager.add_activity(category, start_time, end_time, note)
                    manager.write_data()
                    mark_logged()
                    clear_active_session(key)
                    await update.message.reply_text(
                        f"logged.\ncategory: {category}\n{start_time} to {end_time}\n\"{note}\""
                    )
                return
            elif t in ("no", "nope", "n", "not yet", "nah"):
                context.user_data.pop("pending_nudge", None)
                save_config({"pending_nudge": None})
                await update.message.reply_text("ok, let me know when you're done.")
                return

        elif pending["type"] == "multi":
            sessions = get_active_sessions()

            # "yes" with multiple sessions — ask to specify
            if t in ("yes", "yeah", "yep", "yup", "y", "done", "finished", "yes!"):
                names = [sessions[k].get("note", k) for k in pending["keys"] if k in sessions]
                await update.message.reply_text(
                    f"which one? {', '.join(names)}"
                )
                return

            matched_key = None
            for key in pending["keys"]:
                session = sessions.get(key)
                if not session:
                    continue
                note_lower = session.get("note", "").lower()
                if t == key.lower() or t == note_lower or t in note_lower or note_lower in t:
                    matched_key = key
                    break

            if matched_key:
                context.user_data.pop("pending_nudge", None)
                save_config({"pending_nudge": None})
                session = sessions[matched_key]
                end_time = datetime.now().strftime("%I:%M %p")
                start_time = session["start_time"]
                break_min = session.get("break_minutes", 0)
                note = session["note"]
                if break_min > 0:
                    try:
                        end_dt = datetime.strptime(end_time, "%I:%M %p")
                        end_dt -= timedelta(minutes=break_min)
                        end_time = end_dt.strftime("%I:%M %p")
                    except Exception:
                        pass
                category = classify_activity(note, groq_client)
                manager.add_activity(category, start_time, end_time, note)
                manager.write_data()
                mark_logged()
                clear_active_session(matched_key)
                await update.message.reply_text(
                    f"logged.\ncategory: {category}\n{start_time} to {end_time}\n\"{note}\""
                )
                return
            elif t in ("no", "nope", "n", "nah", "none"):
                context.user_data.pop("pending_nudge", None)
                save_config({"pending_nudge": None})
                await update.message.reply_text("ok.")
                return
            else:
                # unrecognized reply — clear nudge and process as normal message
                context.user_data.pop("pending_nudge", None)
                save_config({"pending_nudge": None})

    # clear sessions open for more than 12 hours — they're stale
    active_sessions = get_active_sessions()
    stale = []
    for k, v in active_sessions.items():
        start_dt_str = v.get("start_dt")
        if start_dt_str:
            try:
                start_dt = datetime.fromisoformat(start_dt_str)
                if (datetime.now() - start_dt).total_seconds() > 12 * 3600:
                    stale.append(k)
            except Exception:
                pass
    for k in stale:
        clear_active_session(k)
    if stale:
        active_sessions = get_active_sessions()

    intent_data = detect_intent(text, active_sessions)
    intent = intent_data.get("intent", "unknown")
    activity = intent_data.get("activity", text)

    # ── Start ──────────────────────────────────────────────────────────────────
    if intent == "start":
        start_time = intent_data.get("start_time") or datetime.now().strftime("%I:%M %p")
        session_key = activity.lower().strip()
        save_active_session(session_key, {
            "note": activity,
            "start_time": start_time,
            "start_dt": datetime.now().isoformat(),
            "break_minutes": 0
        })
        reply = f"started: {activity}\n{start_time}"
        nudge, nudge_meta = get_unfinished_nudge(exclude_key=session_key)
        if nudge:
            reply += f"\n\n{nudge}"
            context.user_data["pending_nudge"] = nudge_meta
            save_config({"pending_nudge": nudge_meta})
        await update.message.reply_text(reply)
        return

    # ── Break ──────────────────────────────────────────────────────────────────
    if intent == "break":
        matched = intent_data.get("matched_session")
        break_min = int(intent_data.get("break_minutes") or 0)
        if matched and matched in active_sessions:
            session = active_sessions[matched]
            session["break_minutes"] = session.get("break_minutes", 0) + break_min
            save_active_session(matched, session)
            await update.message.reply_text(f"noted. {break_min}min break added to {matched}.")
        elif active_sessions:
            key = list(active_sessions.keys())[0]
            session = active_sessions[key]
            session["break_minutes"] = session.get("break_minutes", 0) + break_min
            save_active_session(key, session)
            await update.message.reply_text(f"noted. {break_min}min break added to {key}.")
        else:
            await update.message.reply_text("no active session to add a break to.")
        return

    # ── End ────────────────────────────────────────────────────────────────────
    if intent == "end":
        matched = intent_data.get("matched_session")

        # safety net: if multiple sessions open, only trust Groq's matched_session
        # if the activity name actually appears in what the user typed
        if matched and len(active_sessions) > 1:
            session_note = active_sessions.get(matched, {}).get("note", matched)
            if session_note.lower() not in text.lower() and matched.lower() not in text.lower():
                matched = None

        # only auto-pick first if there is exactly one session
        if not matched and len(active_sessions) == 1:
            matched = list(active_sessions.keys())[0]

        if not matched or matched not in active_sessions:
            if active_sessions:
                sessions_list = "\n".join([f"  {v.get('note', k)}" for k, v in active_sessions.items()])
                await update.message.reply_text(
                    f"which one did you finish?\n{sessions_list}"
                )
            else:
                await update.message.reply_text(
                    "no active sessions. to log a completed activity:\n"
                    "'studied from 2pm to 4pm'"
                )
            return

        session = active_sessions[matched]
        end_time = intent_data.get("end_time") or datetime.now().strftime("%I:%M %p")
        start_time = session["start_time"]
        break_min = session.get("break_minutes", 0)
        note = session["note"]

        # adjust end time for breaks
        if break_min > 0:
            try:
                end_dt = datetime.strptime(end_time, "%I:%M %p")
                end_dt -= timedelta(minutes=break_min)
                end_time = end_dt.strftime("%I:%M %p")
            except Exception:
                pass

        category = classify_activity(note, groq_client)
        manager.add_activity(category, start_time, end_time, note)
        manager.write_data()
        mark_logged()
        clear_active_session(matched)

        break_note = f" (minus {break_min}min break)" if break_min > 0 else ""
        reply = f"logged.\ncategory: {category}\n{start_time} to {end_time}{break_note}\n\"{note}\""
        nudge, nudge_meta = get_unfinished_nudge()
        if nudge:
            reply += f"\n\n{nudge}"
            context.user_data["pending_nudge"] = nudge_meta
            save_config({"pending_nudge": nudge_meta})
        await update.message.reply_text(reply)

        callout = get_absence_callout()
        if callout:
            await update.message.reply_text(callout)
        return

    # ── Regular (explicit times or duration given) ────────────────────────────
    if intent == "regular":
        duration_min = int(intent_data.get("duration_minutes") or 0)
        start_time = intent_data.get("start_time")
        end_time = intent_data.get("end_time")

        # if model gave us duration but not times, compute them
        if duration_min > 0 and not (start_time and end_time):
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(minutes=duration_min)
            end_time = end_dt.strftime("%I:%M %p")
            start_time = start_dt.strftime("%I:%M %p")

        # fallback to full parse if times still missing
        if not (start_time and end_time):
            parsed = parse_activity_from_text(text)
            if not parsed:
                await update.message.reply_text(
                    "couldn't parse the times.\n"
                    "try: 'studied from 2pm to 5pm'\n"
                    "or: 'started studying' and tell me when you finish."
                )
                return
            start_time = parsed.get("start_time")
            end_time = parsed.get("end_time")
            activity = parsed.get("note", text)
            duration_min = duration_min or 0

        category = classify_activity(activity, groq_client)
        manager.add_activity(category, start_time, end_time, activity)
        manager.write_data()
        mark_logged()

        reply = f"logged.\ncategory: {category}\n{start_time} to {end_time}\n\"{activity}\""

        # subtract this duration from any active session automatically
        if duration_min > 0 and active_sessions:
            for key, session in active_sessions.items():
                session["break_minutes"] = session.get("break_minutes", 0) + duration_min
                save_active_session(key, session)
            session_names = ", ".join(active_sessions.keys())
            reply += f"\n\nalso subtracted {duration_min}min from: {session_names}"

        nudge, nudge_meta = get_unfinished_nudge()
        if nudge:
            reply += f"\n\n{nudge}"
            context.user_data["pending_nudge"] = nudge_meta
            save_config({"pending_nudge": nudge_meta})
        await update.message.reply_text(reply)
        callout = get_absence_callout()
        if callout:
            await update.message.reply_text(callout)
        return

    # ── Unknown ────────────────────────────────────────────────────────────────
    if active_sessions:
        session_names = ", ".join([v.get("note", k) for k, v in active_sessions.items()])
        await update.message.reply_text(
            f"still tracking: {session_names}\n\n"
            f"to finish one: 'done with [activity]' — e.g. 'done with {list(active_sessions.values())[0].get('note', 'reading')}'\n"
            f"to log something else: 'studied from 2pm to 4pm'"
        )
    else:
        await update.message.reply_text(
            "didn't get that.\n\n"
            "to start: 'started reading'\n"
            "to finish: 'done with reading'\n"
            "to log directly: 'studied from 2pm to 4pm'\n"
            "or type /help"
        )


WEEK_HOURS = 168.0


CATEGORY_COLORS = {
    "sleep":            "#5B8DB8",
    "study":            "#00C9A7",
    "exercise":         "#FF6B6B",
    "eating":           "#FFB347",
    "self-care":        "#F06292",
    "leisure-positive": "#81C784",
    "leisure-negative": "#EF5350",
    "social":           "#FFD54F",
    "work":             "#4DB6AC",
    "commute":          "#A1887F",
    "chores":           "#90A4AE",
    "health":           "#80CBC4",
}
DEFAULT_COLOR = "#546E7A"


def _week_range_label() -> str:
    today = date.today()
    monday = today - timedelta(days=today.weekday())  # weekday() == 0 for Monday
    sunday = monday + timedelta(days=6)
    # "1 Sep – 7 Sep 2026" — drop year on start if same year
    if monday.year == sunday.year:
        return f"{monday.day} {monday.strftime('%b')} – {sunday.day} {sunday.strftime('%b')} {sunday.year}"
    return f"{monday.day} {monday.strftime('%b %Y')} – {sunday.day} {sunday.strftime('%b %Y')}"


def generate_chart(cats: pd.Series, name: str) -> io.BytesIO:
    pcts = (cats / WEEK_HOURS * 100).round(1).sort_values(ascending=False)

    n = len(pcts)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.1), 5.5))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    colors = [CATEGORY_COLORS.get(cat, DEFAULT_COLOR) for cat in pcts.index]
    bars = ax.bar(range(n), pcts.values, color=colors, width=0.55, zorder=3)

    # subtle horizontal grid lines
    for y in [10, 20, 30, 40]:
        ax.axhline(y, color="#ffffff0d", linewidth=0.8, zorder=1)

    # percentage label inside bar, vertically centered
    for bar, (cat, pct) in zip(bars, pcts.items()):
        bar_mid = bar.get_height() / 2
        # pick contrasting text color per bar
        color = CATEGORY_COLORS.get(cat, DEFAULT_COLOR)
        ax.text(bar.get_x() + bar.get_width() / 2, bar_mid,
                f"{pct:.1f}%", ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")

    # category labels below x axis
    ax.set_xticks(range(n))
    ax.set_xticklabels(pcts.index, rotation=0, ha="center",
                       color="#aaaacc", fontsize=9)

    # always show at least 50% on y so single bars do not look absurd
    ax.set_ylim(0, max(50, max(pcts.values) * 1.15))
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="x", length=0)

    # two-part title: bold name + lighter date range on same line
    week_label = _week_range_label()
    fig.text(0.5, 0.97, f"{name}'s week", ha="center", va="top",
             color="white", fontsize=13, fontweight="bold",
             transform=fig.transFigure)
    fig.text(0.5, 0.925, week_label, ha="center", va="top",
             color="#6666aa", fontsize=9, fontweight="normal",
             transform=fig.transFigure)

    fig.text(0.99, 0.01, "% of 168h", ha="right", color="#444466", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.9])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def get_insights_caption(cats: pd.Series, total_logged: float, name: str) -> str:
    stats = "\n".join([f"{cat}: {h:.1f}h ({h/WEEK_HOURS*100:.1f}% of week)" for cat, h in cats.items()])
    unlogged = WEEK_HOURS - total_logged

    prompt = f"""You are {name}'s honest, slightly sarcastic time coach.
Write ONE punchy sentence (max 15 words) reacting to this specific data.
Rules:
- Only reference what is actually in the data, never assume habits that are not shown
- No internet slang, no military terms, no forced brainrot
- Casual and direct, like a smart friend giving a quick take
- If there is not much data yet, comment on that simply without drama

Data:
{stats}
Not yet logged: {unlogged:.1f}h"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return response.choices[0].message.content.strip()


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = get_name()
    await update.message.reply_text(
        f"WeekFrame helps you see where your 168 hours a week actually go. "
        f"just tell it what you're doing as you go — it figures out the rest, tracks the time, and gives you an honest breakdown.\n\n"
        f"logging is simple: tell me when you start something, tell me when you finish. "
        f"or give me a duration, or exact times — whatever feels natural. "
        f"i'll handle the math and the categories.\n\n"
        f"/insights — this week's breakdown\n"
        f"/reflection — weekly AI reflection on your time\n"
        f"/alltime — your full history since day one\n"
        f"/compare — this week vs last week\n"
        f"/help — show this message"
    )


async def insights(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = analyzer.prepare_data()
    name = get_name()

    if df.empty:
        await update.message.reply_text(f"no data yet {name}, start logging")
        return

    weeks = sorted(df["week"].unique())
    df_week = df[df["week"] == weeks[-1]]

    total_hours = df_week.groupby("category")["duration_hours"].sum().round(2)
    total_logged = total_hours.sum()
    cats = total_hours[total_hours > 0].sort_values(ascending=False)

    snark = get_insights_caption(cats, total_logged, name)

    hours_lines = "\n".join([
        f"{cat}: {h:.1f}h" for cat, h in cats.items()
    ])
    caption = f"{snark}\n\n{hours_lines}\n\ntotal logged: {total_logged:.1f} / 168h"

    gaps = get_unlogged_days()
    if gaps:
        caption += "\n\nnot logged / low activity:\n" + "\n".join(f"  {g}" for g in gaps)

    chart = generate_chart(cats, name)
    await update.message.reply_photo(photo=chart, caption=caption)


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

    last_h = df_last.groupby("category")["duration_hours"].sum().round(2)
    curr_h = df_curr.groupby("category")["duration_hours"].sum().round(2)

    all_cats = sorted(set(last_h.index) | set(curr_h.index))

    lines = [f"this week vs last week {name}:\n"]
    for cat in all_cats:
        curr = curr_h.get(cat, 0)
        prev = last_h.get(cat, 0)
        diff = curr - prev
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        sign = "+" if diff >= 0 else ""
        lines.append(f"{cat}: {curr:.1f}h {arrow} ({sign}{diff:.1f}h)")

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
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9
    )

    await context.bot.send_message(
        chat_id=chat_id,
        text=response.choices[0].message.content.strip()
    )


# ── Main ──────────────────────────────────────────────────────────────────────

async def post_init(app):
    await app.bot.set_my_commands([
        BotCommand("insights", "this week's breakdown"),
        BotCommand("reflection", "weekly AI reflection"),
        BotCommand("alltime", "everything since you started"),
        BotCommand("compare", "this week vs last week"),
        BotCommand("help", "how to use WeekFrame"),
    ])


if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN not set")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("insights", insights))
    app.add_handler(CommandHandler("reflection", reflection))
    app.add_handler(CommandHandler("alltime", alltime))
    app.add_handler(CommandHandler("compare", compare))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.job_queue.run_daily(
        send_reminder,
        time=datetime.strptime("20:00", "%H:%M").time()
    )

    print("WeekFrame bot is running.")
    app.run_polling()

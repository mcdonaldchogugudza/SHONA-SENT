# ======================================================================
# SHONA X 50,000-TWEET CORPUS
# FINAL FULL-ARCHIVE COLLECTION: 2020-2025
# ======================================================================
#
# STOPPING CONDITIONS
# -------------------
# 1. Stop when 50,000 accepted posts are collected.
# 2. Stop immediately and safely when X returns HTTP 402 / credits
#    depleted.
# 3. Stop naturally if every configured query is exhausted.
#
# The script does NOT attempt to estimate or hard-code the user's X
# credit balance. X is the authority on the remaining credit balance.
#
# SECURITY
# --------
# The Bearer Token is requested with getpass() and is never stored in
# this source file.
#
# OUTPUTS
# -------
# shona_x_corpus_50000_full_archive.csv
# shona_x_corpus_50000_full_archive.xlsx
# shona_x_rejected_full_archive.csv
# shona_x_full_archive_checkpoint.json
#
# ======================================================================

import getpass
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pandas as pd
import requests


# ======================================================================
# 1. CONFIGURATION
# ======================================================================

TOTAL_TARGET = 50_000

DOMAIN_TARGETS = {
    "healthcare": 16_667,
    "education": 16_667,
    "government": 16_666,
}

SEARCH_URL = "https://api.x.com/2/tweets/search/all"

START_DATE = "2020-01-01T00:00:00Z"
END_DATE = "2025-12-31T23:59:59Z"

# X Full-Archive supports up to 500 posts per page where the account
# and endpoint permit that page size.
MAX_RESULTS = 500

REQUEST_DELAY = 1.1
MAX_RETRIES = 6
INITIAL_BACKOFF = 2

CHECKPOINT_FILE = "shona_x_full_archive_checkpoint.json"
ACCEPTED_CSV = "shona_x_corpus_50000_full_archive.csv"
ACCEPTED_XLSX = "shona_x_corpus_50000_full_archive.xlsx"
REJECTED_CSV = "shona_x_rejected_full_archive.csv"


# ======================================================================
# 2. STATUS CONSTANTS
# ======================================================================

STATUS_OK = "OK"
STATUS_TARGET_REACHED = "TARGET_REACHED"
STATUS_DOMAIN_TARGET_REACHED = "DOMAIN_TARGET_REACHED"
STATUS_QUERY_EXHAUSTED = "QUERY_EXHAUSTED"
STATUS_CREDITS_DEPLETED = "CREDITS_DEPLETED"
STATUS_RATE_LIMIT = "RATE_LIMIT"
STATUS_AUTH_ERROR = "AUTH_ERROR"
STATUS_FORBIDDEN = "FORBIDDEN"
STATUS_BAD_REQUEST = "BAD_REQUEST"
STATUS_TEMPORARY_ERROR = "TEMPORARY_ERROR"
STATUS_FATAL_ERROR = "FATAL_ERROR"


# ======================================================================
# 3. SEARCH QUERIES
# ======================================================================

QUERIES = {
    "healthcare": [
        '(chiremba OR vanachiremba OR chipatara OR zvipatara) -is:retweet',
        '(mushonga OR mishonga OR kurapa OR kurapwa) -is:retweet',
        '(hutano OR utano OR murwere OR varwere) -is:retweet',
        '(mukoti OR vanamukoti OR kurwara) -is:retweet',
        '(kuvhiya OR kuvhiyiwa OR nhumbu OR pamuviri) -is:retweet',
        '(chipatara OR zvipatara) (hurumende OR vanhu OR nyika) -is:retweet',
        '(vana OR mwana) (chiremba OR chipatara OR mushonga OR hutano) -is:retweet',
        '(health OR healthcare OR hospital OR doctor) (Zimbabwe OR Shona) -is:retweet',
    ],
    "education": [
        '(chikoro OR zvikoro) -is:retweet',
        '(dzidzo OR kudzidza OR kudzidzisa) -is:retweet',
        '(mudzidzisi OR vadzidzisi OR mudzidzi OR vadzidzi) -is:retweet',
        '(bvunzo OR exam OR exams) (chikoro OR mwana OR vadzidzi) -is:retweet',
        '(yunivhesiti OR koreji OR university) -is:retweet',
        '(grade OR form) (chikoro OR mwana OR vadzidzi) -is:retweet',
        '(zvikoro OR chikoro) (hurumende OR vanhu OR nyika) -is:retweet',
        '(education OR school OR teacher OR students) (Zimbabwe OR Shona) -is:retweet',
    ],
    "government": [
        '(hurumende OR mutungamiri OR purezidhendi) -is:retweet',
        '(sarudzo OR pasarudzo OR kuvhota) -is:retweet',
        '(kanzuru OR vagari OR mutemo OR mitemo) -is:retweet',
        '(bazi OR ministry OR hurumende) -is:retweet',
        '(mari OR mutero OR tax) (hurumende OR vanhu OR nyika) -is:retweet',
        '(Zimbabwe OR nyika) (hurumende OR mutungamiri OR purezidhendi) -is:retweet',
        '(zvikoro OR zvipatara) (hurumende OR mutungamiri OR nyika) -is:retweet',
        '(government OR president OR election OR elections) (Zimbabwe OR Shona) -is:retweet',
    ],
}


# ======================================================================
# 4. SCREENING VOCABULARIES
# ======================================================================

SHONA_WORDS = {
    # General
    "ndiri", "ini", "isu", "ivo", "iye", "vanhu", "mwana", "vana",
    "amai", "baba", "munhu", "nyika", "kumusha", "kumaruwa", "vagari",
    "hama",
    # Healthcare
    "chiremba", "vanachiremba", "chipatara", "zvipatara", "mushonga",
    "mishonga", "hutano", "utano", "mukoti", "vanamukoti", "kurwara",
    "kuvhiyiwa", "kuvhhiya", "kuvhiya", "nhumbu", "pamuviri", "kurapwa",
    "rapwa", "kurapa", "murwere", "varwere",
    # Education
    "chikoro", "zvikoro", "dzidzo", "kudzidza", "kudzidzisa", "mudzidzisi",
    "vadzidzisi", "mudzidzi", "vadzidzi", "bvunzo", "yunivhesiti", "koreji",
    "bhuku", "mabhuku",
    # Government
    "hurumende", "mutungamiri", "purezidhendi", "sarudzo", "pasarudzo",
    "kuvhota", "kanzuru", "mutemo", "mitemo", "bazi", "mari", "mutero",
    "zimbabwe",
    # Common Shona
    "ndinoda", "handidi", "ndino", "zvino", "nhasi", "nezuro", "mangwana",
    "sei", "chii", "ko", "asi", "saka", "nekuti", "kana", "zvakanaka",
    "zvakaipa", "kwete", "hongu", "ndatenda", "tenda", "tinoda", "vanoda",
    "anoda", "basa", "mabasa", "mudzimai", "murume", "musikana", "mukomana",
    "rudo", "kuda", "kufara", "farai", "mufaro", "zvinofadza", "dambudziko",
    "matambudziko", "rufu", "kufa", "kuchema", "hasha", "kutya", "hurombo",
    "huori",
}

COMMON_ENGLISH = {
    "the", "and", "is", "are", "was", "were", "this", "that", "with", "for",
    "from", "you", "your", "we", "they", "have", "has", "had", "not", "but",
    "about", "today", "people", "government", "school", "hospital", "doctor",
    "health", "education", "money", "country", "president", "child", "children",
    "teacher", "teachers", "student", "students", "university", "election",
    "elections", "vote", "votes", "tax", "taxes", "minister", "ministry",
}

POSITIVE_WORDS = {
    "zvakanaka", "ndatenda", "tenda", "farai", "kufara", "rudo", "kuda",
    "anoda", "ndinoda", "tinoda", "kubudirira", "budiriro", "mufaro",
    "zvinofadza",
}

NEGATIVE_WORDS = {
    "zvakaipa", "kurwara", "dambudziko", "matambudziko", "rufu", "kufa",
    "kuchema", "hasha", "kushungurudzika", "kutya", "hurombo", "huori",
}

DOMAIN_WORDS = {
    "healthcare": {
        "chiremba", "vanachiremba", "chipatara", "zvipatara", "mushonga",
        "mishonga", "hutano", "utano", "mukoti", "vanamukoti", "kurwara",
        "kuvhiya", "kuvhiyiwa", "nhumbu", "pamuviri", "murwere", "varwere",
        "health", "healthcare", "hospital", "doctor",
    },
    "education": {
        "chikoro", "zvikoro", "dzidzo", "kudzidza", "kudzidzisa", "mudzidzisi",
        "vadzidzisi", "mudzidzi", "vadzidzi", "bvunzo", "yunivhesiti", "koreji",
        "exam", "exams", "grade", "form", "education", "school", "teacher",
        "teachers", "student", "students", "university",
    },
    "government": {
        "hurumende", "mutungamiri", "purezidhendi", "sarudzo", "pasarudzo",
        "kuvhota", "kanzuru", "mutemo", "mitemo", "bazi", "ministry", "mari",
        "mutero", "tax", "taxes", "government", "president", "election",
        "elections", "minister", "zimbabwe",
    },
}


# ======================================================================
# 5. GLOBAL STATE
# ======================================================================

accepted = []
rejected_records = []
seen_ids = set()

# Each query gets its own counters. A tweet is globally deduplicated by
# tweet ID, even when it appears under several queries.
query_stats = defaultdict(lambda: {
    "pages": 0,
    "read": 0,
    "accepted": 0,
    "rejected": 0,
    "duplicates": 0,
    "errors": 0,
})

domain_stats = {
    domain: {
        "read": 0,
        "accepted": 0,
        "rejected": 0,
        "duplicates": 0,
    }
    for domain in DOMAIN_TARGETS
}

rejection_reasons = {
    domain: Counter()
    for domain in DOMAIN_TARGETS
}

global_stats = {
    "api_requests": 0,
    "api_errors": 0,
    "rate_limits": 0,
    "credits_depleted": 0,
    "pages": 0,
    "raw_posts_read": 0,
    "duplicates": 0,
    "accepted": 0,
    "rejected": 0,
}

stop_reason = None


# ======================================================================
# 6. TEXT FUNCTIONS
# ======================================================================

def tokenize(text):
    if not text:
        return []
    return re.findall(r"[A-Za-zÀ-ÿĀ-ž'-]+", str(text).lower())


def clean_text(text):
    if not text:
        return ""

    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\bRT\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_shona(text):
    tokens = tokenize(text)

    if not tokens:
        return {
            "is_shona": False,
            "shona_ratio": 0.0,
            "shona_count": 0,
            "english_count": 0,
            "language_type": "unknown",
        }

    shona_count = sum(token in SHONA_WORDS for token in tokens)
    english_count = sum(token in COMMON_ENGLISH for token in tokens)
    total = len(tokens)

    shona_ratio = shona_count / total
    english_ratio = english_count / total

    is_shona = (
        shona_count >= 2
        or (shona_count >= 1 and shona_ratio >= 0.08)
    )

    if not is_shona:
        language_type = "non_shona"
    elif english_count > 0 and english_ratio > 0:
        language_type = "shona_english_codeswitch"
    else:
        language_type = "shona"

    return {
        "is_shona": is_shona,
        "shona_ratio": round(shona_ratio, 4),
        "shona_count": shona_count,
        "english_count": english_count,
        "language_type": language_type,
    }


def score_domains(text):
    tokens = set(tokenize(text))
    scores = {}
    matches = {}

    for domain, words in DOMAIN_WORDS.items():
        matched = sorted(tokens.intersection(words))
        matches[domain] = matched
        scores[domain] = len(matched)

    return scores, matches


def assign_primary_domain(text):
    scores, matches = score_domains(text)
    max_score = max(scores.values())

    if max_score == 0:
        return None, scores, matches

    winners = [domain for domain, score in scores.items() if score == max_score]

    if len(winners) == 1:
        return winners[0], scores, matches

    return "multi_domain", scores, matches


def calculate_sentiment(text):
    tokens = tokenize(text)
    positive = sum(token in POSITIVE_WORDS for token in tokens)
    negative = sum(token in NEGATIVE_WORDS for token in tokens)

    if positive > negative:
        return 1, "positive", positive, negative
    if negative > positive:
        return -1, "negative", positive, negative
    return 0, "neutral", positive, negative


def detect_reply(tweet):
    return tweet.get("in_reply_to_user_id") is not None


def is_retweet(tweet):
    referenced = tweet.get("referenced_tweets") or []
    return any(item.get("type") == "retweeted" for item in referenced)


def validate_tweet(tweet):
    raw_text = tweet.get("text", "")
    cleaned = clean_text(raw_text)

    if not cleaned:
        return None, "empty_text"

    if is_retweet(tweet):
        return None, "retweet"

    language = detect_shona(cleaned)
    if not language["is_shona"]:
        return None, "not_shona"

    tokens = tokenize(cleaned)
    token_count = len(tokens)

    if token_count >= 8:
        english_ratio = language["english_count"] / token_count
        if english_ratio > 0.75 and language["shona_count"] <= 1:
            return None, "english_dominant"

    domain, domain_scores, domain_matches = assign_primary_domain(cleaned)

    if domain is None:
        return None, "no_domain"

    if domain == "multi_domain":
        return None, "multi_domain"

    sentiment_score, polarity, positive_count, negative_count = calculate_sentiment(cleaned)

    metrics = tweet.get("public_metrics") or {}

    record = {
        "tweet_id": tweet.get("id"),
        "created_at": tweet.get("created_at"),
        "text": raw_text,
        "cleaned_text": cleaned,
        "domain": domain,
        "domain_healthcare_score": domain_scores["healthcare"],
        "domain_education_score": domain_scores["education"],
        "domain_government_score": domain_scores["government"],
        "healthcare_matches": ", ".join(domain_matches["healthcare"]),
        "education_matches": ", ".join(domain_matches["education"]),
        "government_matches": ", ".join(domain_matches["government"]),
        "detected_language": language["language_type"],
        "shona_word_count": language["shona_count"],
        "english_word_count": language["english_count"],
        "shona_ratio": language["shona_ratio"],
        "sentiment_score": sentiment_score,
        "polarity": polarity,
        "positive_word_count": positive_count,
        "negative_word_count": negative_count,
        "is_reply": detect_reply(tweet),
        "in_reply_to_user_id": tweet.get("in_reply_to_user_id"),
        "author_id": tweet.get("author_id"),
        "conversation_id": tweet.get("conversation_id"),
        "lang": tweet.get("lang"),
        "possibly_sensitive": tweet.get("possibly_sensitive"),
        "source": tweet.get("source"),
        "like_count": metrics.get("like_count"),
        "reply_count": metrics.get("reply_count"),
        "retweet_count": metrics.get("retweet_count"),
        "quote_count": metrics.get("quote_count"),
    }

    return record, None


# ======================================================================
# 7. REJECTION LOGGING
# ======================================================================

def make_rejection_record(tweet, reason, query_domain, query_number):
    return {
        "tweet_id": tweet.get("id"),
        "created_at": tweet.get("created_at"),
        "text": tweet.get("text", ""),
        "query_domain": query_domain,
        "query_number": query_number,
        "rejection_reason": reason,
    }


# ======================================================================
# 8. API REQUEST
# ======================================================================

def search_x(query, next_token=None):
    params = {
        "query": query,
        "start_time": START_DATE,
        "end_time": END_DATE,
        "max_results": MAX_RESULTS,
        "tweet.fields": ",".join([
            "id",
            "text",
            "created_at",
            "author_id",
            "conversation_id",
            "in_reply_to_user_id",
            "lang",
            "possibly_sensitive",
            "public_metrics",
            "source",
            "entities",
            "referenced_tweets",
        ]),
    }

    if next_token:
        params["next_token"] = next_token

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            global_stats["api_requests"] += 1
            response = requests.get(
                SEARCH_URL,
                headers=HEADERS,
                params=params,
                timeout=90,
            )
        except requests.exceptions.RequestException as exc:
            global_stats["api_errors"] += 1
            print(f"REQUEST ERROR {attempt}/{MAX_RETRIES}: {exc}")

            if attempt < MAX_RETRIES:
                wait = INITIAL_BACKOFF ** attempt
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
                continue

            return {
                "status": STATUS_TEMPORARY_ERROR,
                "data": None,
                "next_token": next_token,
            }

        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError:
                global_stats["api_errors"] += 1
                return {
                    "status": STATUS_FATAL_ERROR,
                    "data": None,
                    "next_token": next_token,
                }

            return {
                "status": STATUS_OK,
                "data": data,
                "next_token": (data.get("meta") or {}).get("next_token"),
            }

        if response.status_code == 402:
            global_stats["credits_depleted"] += 1
            print()
            print("X API CREDITS DEPLETED (HTTP 402).")
            print(response.text)
            return {
                "status": STATUS_CREDITS_DEPLETED,
                "data": None,
                "next_token": next_token,
            }

        if response.status_code == 401:
            print("API ERROR 401: Bearer token is invalid or expired.")
            return {
                "status": STATUS_AUTH_ERROR,
                "data": None,
                "next_token": next_token,
            }

        if response.status_code == 403:
            print("API ERROR 403")
            print(response.text)
            return {
                "status": STATUS_FORBIDDEN,
                "data": None,
                "next_token": next_token,
            }

        if response.status_code == 400:
            print("API ERROR 400")
            print(response.text)
            return {
                "status": STATUS_BAD_REQUEST,
                "data": None,
                "next_token": next_token,
            }

        if response.status_code == 429:
            global_stats["rate_limits"] += 1
            print("RATE LIMIT REACHED.")

            reset = response.headers.get("x-rate-limit-reset")
            if reset:
                try:
                    wait_seconds = max(int(reset) - int(time.time()) + 5, 5)
                except (ValueError, TypeError):
                    wait_seconds = 60
            else:
                wait_seconds = 60

            print(f"Waiting {wait_seconds} seconds...")
            time.sleep(wait_seconds)
            continue

        if response.status_code in {500, 502, 503, 504}:
            global_stats["api_errors"] += 1
            print(f"SERVER ERROR {response.status_code}")

            if attempt < MAX_RETRIES:
                wait = INITIAL_BACKOFF ** attempt
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
                continue

            return {
                "status": STATUS_TEMPORARY_ERROR,
                "data": None,
                "next_token": next_token,
            }

        global_stats["api_errors"] += 1
        print(f"UNEXPECTED HTTP STATUS {response.status_code}")
        print(response.text)
        return {
            "status": STATUS_FATAL_ERROR,
            "data": None,
            "next_token": next_token,
        }

    return {
        "status": STATUS_TEMPORARY_ERROR,
        "data": None,
        "next_token": next_token,
    }


# ======================================================================
# 9. TARGET / PROGRESS HELPERS
# ======================================================================

def corpus_complete():
    return len(accepted) >= TOTAL_TARGET


def domain_complete(domain):
    return domain_stats[domain]["accepted"] >= DOMAIN_TARGETS[domain]


def print_progress():
    print("-" * 80)
    print(f"TOTAL ACCEPTED : {len(accepted):,}/{TOTAL_TARGET:,}")
    for domain, target in DOMAIN_TARGETS.items():
        print(f"  {domain:<12}: {domain_stats[domain]['accepted']:,}/{target:,}")
    print(f"API REQUESTS   : {global_stats['api_requests']:,}")
    print(f"POSTS READ     : {global_stats['raw_posts_read']:,}")
    print(f"DUPLICATES     : {global_stats['duplicates']:,}")
    print(f"REJECTED       : {global_stats['rejected']:,}")
    print(f"RATE LIMITS    : {global_stats['rate_limits']:,}")
    print(f"CREDITS 402    : {global_stats['credits_depleted']:,}")
    print("-" * 80)


# ======================================================================
# 10. OUTPUT / CHECKPOINT FUNCTIONS
# ======================================================================

def save_accepted():
    if not accepted:
        return
    pd.DataFrame(accepted).to_csv(
        ACCEPTED_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def save_rejected():
    if not rejected_records:
        return
    pd.DataFrame(rejected_records).to_csv(
        REJECTED_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def save_checkpoint(query_index, next_tokens, completed_queries, current_stop_reason=None):
    checkpoint = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "query_index": query_index,
        "next_tokens": next_tokens,
        "completed_queries": sorted(completed_queries),
        "accepted_count": len(accepted),
        "seen_count": len(seen_ids),
        "domain_stats": domain_stats,
        "global_stats": global_stats,
        "rejection_reasons": {
            domain: dict(counter)
            for domain, counter in rejection_reasons.items()
        },
        "stop_reason": current_stop_reason,
    }

    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as file:
        json.dump(checkpoint, file, indent=2, ensure_ascii=False)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None

    print()
    print("CHECKPOINT FOUND")
    print(f"File: {CHECKPOINT_FILE}")
    answer = input("Resume previous collection? [Y/n]: ").strip().lower()

    if answer not in {"", "y", "yes"}:
        print("Starting a new collection.")
        return None

    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    print(f"Previously accepted: {checkpoint.get('accepted_count', 0):,}")
    return checkpoint


def restore_previous_collection():
    if not os.path.exists(ACCEPTED_CSV):
        return

    previous_df = pd.read_csv(ACCEPTED_CSV)
    records = previous_df.to_dict("records")

    accepted.extend(records)

    for record in records:
        tweet_id = str(record.get("tweet_id"))
        seen_ids.add(tweet_id)

        domain = record.get("domain")
        if domain in domain_stats:
            domain_stats[domain]["accepted"] += 1

    global_stats["accepted"] = len(accepted)

    print(f"Loaded {len(accepted):,} previously accepted posts.")


def save_xlsx():
    accepted_df = pd.DataFrame(accepted)
    rejected_df = pd.DataFrame(rejected_records)

    domain_rows = []
    for domain, target in DOMAIN_TARGETS.items():
        stats = domain_stats[domain]
        rate = (stats["accepted"] / stats["read"] * 100) if stats["read"] else 0
        domain_rows.append({
            "domain": domain,
            "target": target,
            "accepted": stats["accepted"],
            "read": stats["read"],
            "duplicates": stats["duplicates"],
            "rejected": stats["rejected"],
            "acceptance_rate_percent": round(rate, 2),
        })

    domain_df = pd.DataFrame(domain_rows)

    rejection_rows = []
    for domain, counter in rejection_reasons.items():
        for reason, count in counter.most_common():
            rejection_rows.append({
                "domain": domain,
                "reason": reason,
                "count": count,
            })
    rejection_summary_df = pd.DataFrame(rejection_rows)

    query_rows = []
    for (domain, query_number), stats in sorted(query_stats.items()):
        rate = (stats["accepted"] / stats["read"] * 100) if stats["read"] else 0
        query_rows.append({
            "domain": domain,
            "query_number": query_number,
            "pages": stats["pages"],
            "read": stats["read"],
            "accepted": stats["accepted"],
            "rejected": stats["rejected"],
            "duplicates": stats["duplicates"],
            "errors": stats["errors"],
            "acceptance_rate_percent": round(rate, 2),
        })
    query_df = pd.DataFrame(query_rows)

    if not accepted_df.empty:
        sentiment_df = pd.crosstab(accepted_df["domain"], accepted_df["polarity"])
        language_df = pd.crosstab(accepted_df["domain"], accepted_df["detected_language"])
        reply_df = pd.crosstab(accepted_df["domain"], accepted_df["is_reply"])
    else:
        sentiment_df = pd.DataFrame()
        language_df = pd.DataFrame()
        reply_df = pd.DataFrame()

    metadata = {
        "collection_type": "X API v2 Full-Archive Search",
        "endpoint": SEARCH_URL,
        "period_start": START_DATE,
        "period_end": END_DATE,
        "target_total": TOTAL_TARGET,
        "accepted_total": len(accepted),
        "posts_read": global_stats["raw_posts_read"],
        "duplicates": global_stats["duplicates"],
        "rejected": global_stats["rejected"],
        "api_requests": global_stats["api_requests"],
        "api_errors": global_stats["api_errors"],
        "rate_limit_events": global_stats["rate_limits"],
        "credits_depleted_events": global_stats["credits_depleted"],
        "collection_completed": corpus_complete(),
        "stop_reason": stop_reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_df = pd.DataFrame([
        {"parameter": key, "value": value}
        for key, value in metadata.items()
    ])

    with pd.ExcelWriter(ACCEPTED_XLSX, engine="openpyxl") as writer:
        accepted_df.to_excel(writer, sheet_name="Corpus", index=False)
        domain_df.to_excel(writer, sheet_name="Domain Summary", index=False)
        rejection_summary_df.to_excel(writer, sheet_name="Rejections", index=False)
        query_df.to_excel(writer, sheet_name="Query Performance", index=False)
        sentiment_df.to_excel(writer, sheet_name="Sentiment")
        language_df.to_excel(writer, sheet_name="Language")
        reply_df.to_excel(writer, sheet_name="Replies")
        metadata_df.to_excel(writer, sheet_name="Collection Metadata", index=False)
        if not rejected_df.empty:
            rejected_df.to_excel(writer, sheet_name="Rejected Posts", index=False)


# ======================================================================
# 11. COLLECT ONE QUERY
# ======================================================================

def collect_query(domain, query_number, query, starting_next_token=None):
    local_next_token = starting_next_token
    page = 0

    print()
    print("=" * 80)
    print(f"DOMAIN: {domain.upper()}")
    print(f"QUERY: {query_number}/{len(QUERIES[domain])}")
    print(query)
    print("=" * 80)

    while True:
        if corpus_complete():
            return {
                "status": STATUS_TARGET_REACHED,
                "next_token": local_next_token,
            }

        if domain_complete(domain):
            return {
                "status": STATUS_DOMAIN_TARGET_REACHED,
                "next_token": local_next_token,
            }

        page += 1
        print(f"Requesting page {page}...")

        result = search_x(query, local_next_token)
        status = result["status"]

        if status != STATUS_OK:
            query_stats[(domain, query_number)]["errors"] += 1
            return {
                "status": status,
                "next_token": local_next_token,
            }

        data = result["data"] or {}
        tweets = data.get("data") or []
        meta = data.get("meta") or {}

        global_stats["pages"] += 1
        query_stats[(domain, query_number)]["pages"] += 1

        if not tweets:
            print("No posts returned; query exhausted.")
            return {
                "status": STATUS_QUERY_EXHAUSTED,
                "next_token": None,
            }

        for tweet in tweets:
            global_stats["raw_posts_read"] += 1
            domain_stats[domain]["read"] += 1
            query_stats[(domain, query_number)]["read"] += 1

            tweet_id = tweet.get("id")

            if not tweet_id:
                reason = "missing_tweet_id"
                global_stats["rejected"] += 1
                domain_stats[domain]["rejected"] += 1
                query_stats[(domain, query_number)]["rejected"] += 1
                rejection_reasons[domain][reason] += 1
                rejected_records.append(make_rejection_record(tweet, reason, domain, query_number))
                continue

            if tweet_id in seen_ids:
                global_stats["duplicates"] += 1
                domain_stats[domain]["duplicates"] += 1
                query_stats[(domain, query_number)]["duplicates"] += 1
                continue

            seen_ids.add(tweet_id)

            record, reason = validate_tweet(tweet)

            if record is None:
                global_stats["rejected"] += 1
                domain_stats[domain]["rejected"] += 1
                query_stats[(domain, query_number)]["rejected"] += 1
                rejection_reasons[domain][reason] += 1
                rejected_records.append(make_rejection_record(tweet, reason, domain, query_number))
                continue

            assigned_domain = record["domain"]

            # Preserve balance. A query can return a tweet belonging to a
            # different domain, so acceptance is based on the classifier,
            # not on the query label.
            if domain_complete(assigned_domain):
                reason = "assigned_domain_target_full"
                global_stats["rejected"] += 1
                domain_stats[domain]["rejected"] += 1
                query_stats[(domain, query_number)]["rejected"] += 1
                rejection_reasons[domain][reason] += 1
                rejected_records.append(make_rejection_record(tweet, reason, domain, query_number))
                continue

            accepted.append(record)
            global_stats["accepted"] += 1
            domain_stats[assigned_domain]["accepted"] += 1
            query_stats[(domain, query_number)]["accepted"] += 1

            if corpus_complete():
                break

        print(f"Page {page}: API posts={len(tweets):,}")
        print(f"Total accepted={len(accepted):,}/{TOTAL_TARGET:,}")
        for d, target in DOMAIN_TARGETS.items():
            print(f"  {d:<12}: {domain_stats[d]['accepted']:,}/{target:,}")

        if corpus_complete():
            return {
                "status": STATUS_TARGET_REACHED,
                "next_token": meta.get("next_token"),
            }

        local_next_token = meta.get("next_token")

        if not local_next_token:
            print("No next_token returned. This query/time range is exhausted.")
            return {
                "status": STATUS_QUERY_EXHAUSTED,
                "next_token": None,
            }

        time.sleep(REQUEST_DELAY)


# ======================================================================
# 12. MAIN
# ======================================================================

def main():
    global stop_reason

    collection_start = datetime.now(timezone.utc)

    print("=" * 80)
    print("SHONA X 50,000-TWEET CORPUS")
    print("FINAL FULL-ARCHIVE 2020-2025 COLLECTION")
    print("=" * 80)
    print()
    print("COLLECTION PERIOD:")
    print(f"  Start : {START_DATE}")
    print(f"  End   : {END_DATE}")
    print()
    print("TARGET:")
    for domain, target in DOMAIN_TARGETS.items():
        print(f"  {domain:<12}: {target:,}")
    print()
    print("STOPPING CONDITIONS:")
    print("  1. 50,000 accepted tweets reached.")
    print("  2. X API credits depleted (HTTP 402).")
    print("  3. All configured queries exhausted.")
    print()
    print("SECURITY NOTICE")
    print("-" * 80)
    print("Use a valid X API Bearer Token.")
    print("Do not paste the token into this source file.")
    print()

    bearer_token = getpass.getpass("Paste your X API Bearer Token: ").strip()
    if not bearer_token:
        raise ValueError("No Bearer Token supplied.")

    global HEADERS
    HEADERS = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }

    checkpoint = load_checkpoint()

    query_index = {domain: 0 for domain in QUERIES}
    next_tokens = {}
    completed_queries = set()

    if checkpoint:
        query_index.update(checkpoint.get("query_index", {}))
        next_tokens.update(checkpoint.get("next_tokens", {}))
        completed_queries.update(checkpoint.get("completed_queries", []))
        restore_previous_collection()
    else:
        print("Starting a new collection.")

    domain_order = ["healthcare", "education", "government"]

    while not corpus_complete():
        made_progress = False
        queries_remaining = False

        for domain in domain_order:
            if corpus_complete():
                stop_reason = STATUS_TARGET_REACHED
                break

            if domain_complete(domain):
                continue

            current_index = query_index[domain]
            if current_index >= len(QUERIES[domain]):
                continue

            queries_remaining = True
            query_number = current_index + 1
            query = QUERIES[domain][current_index]
            query_key = f"{domain}:{query_number}"

            if query_key in completed_queries:
                query_index[domain] += 1
                continue

            before = len(accepted)
            result = collect_query(
                domain,
                query_number,
                query,
                next_tokens.get(query_key),
            )

            status = result["status"]
            returned_token = result.get("next_token")

            if len(accepted) > before:
                made_progress = True

            # ----------------------------------------------------------
            # STOP IMMEDIATELY ON CREDIT DEPLETION
            # ----------------------------------------------------------
            if status == STATUS_CREDITS_DEPLETED:
                if returned_token:
                    next_tokens[query_key] = returned_token

                stop_reason = STATUS_CREDITS_DEPLETED
                save_accepted()
                save_rejected()
                save_checkpoint(
                    query_index,
                    next_tokens,
                    completed_queries,
                    stop_reason,
                )
                save_xlsx()
                print_progress()
                break

            # ----------------------------------------------------------
            # STOP AT 50,000
            # ----------------------------------------------------------
            if status == STATUS_TARGET_REACHED or corpus_complete():
                if returned_token:
                    next_tokens[query_key] = returned_token
                stop_reason = STATUS_TARGET_REACHED
                break

            # ----------------------------------------------------------
            # QUERY EXHAUSTED
            # ----------------------------------------------------------
            if status == STATUS_QUERY_EXHAUSTED:
                completed_queries.add(query_key)
                query_index[domain] += 1
                next_tokens.pop(query_key, None)

            # ----------------------------------------------------------
            # DOMAIN TARGET REACHED
            # ----------------------------------------------------------
            elif status == STATUS_DOMAIN_TARGET_REACHED:
                if returned_token:
                    next_tokens[query_key] = returned_token

            # ----------------------------------------------------------
            # TEMPORARY / FATAL / AUTH / OTHER ERROR
            # ----------------------------------------------------------
            else:
                if returned_token:
                    next_tokens[query_key] = returned_token
                save_accepted()
                save_rejected()
                save_checkpoint(
                    query_index,
                    next_tokens,
                    completed_queries,
                    status,
                )
                save_xlsx()
                stop_reason = status
                print(f"COLLECTION STOPPED BECAUSE OF API STATUS: {status}")
                break

            save_accepted()
            save_rejected()
            save_checkpoint(
                query_index,
                next_tokens,
                completed_queries,
                None,
            )
            print_progress()

        if stop_reason in {
            STATUS_TARGET_REACHED,
            STATUS_CREDITS_DEPLETED,
            STATUS_AUTH_ERROR,
            STATUS_FORBIDDEN,
            STATUS_BAD_REQUEST,
            STATUS_TEMPORARY_ERROR,
            STATUS_FATAL_ERROR,
        }:
            break

        if not queries_remaining:
            stop_reason = STATUS_QUERY_EXHAUSTED
            break

        if not made_progress:
            # Queries may still be moving through pages even when a single
            # rotation adds no accepted posts. Continue unless all queries
            # have been exhausted.
            any_unfinished = any(
                query_index[d] < len(QUERIES[d]) and not domain_complete(d)
                for d in domain_order
            )
            if not any_unfinished:
                stop_reason = STATUS_QUERY_EXHAUSTED
                break

    if corpus_complete():
        stop_reason = STATUS_TARGET_REACHED

    save_accepted()
    save_rejected()
    save_checkpoint(
        query_index,
        next_tokens,
        completed_queries,
        stop_reason,
    )
    save_xlsx()

    collection_end = datetime.now(timezone.utc)

    print()
    print("=" * 80)
    print("FINAL SHONA X CORPUS RESULTS")
    print("=" * 80)
    print(f"Target corpus       : {TOTAL_TARGET:,}")
    print(f"Accepted posts      : {len(accepted):,}")
    print(f"Posts read          : {global_stats['raw_posts_read']:,}")
    print(f"Duplicates          : {global_stats['duplicates']:,}")
    print(f"Rejected            : {global_stats['rejected']:,}")
    print(f"API requests        : {global_stats['api_requests']:,}")
    print(f"Rate-limit events   : {global_stats['rate_limits']:,}")
    print(f"Credits depleted    : {global_stats['credits_depleted']:,}")
    print(f"Stop reason         : {stop_reason}")

    acceptance_rate = (
        len(accepted) / global_stats["raw_posts_read"] * 100
        if global_stats["raw_posts_read"]
        else 0
    )
    print(f"Acceptance rate     : {acceptance_rate:.2f}%")

    print()
    print("FINAL DOMAIN DISTRIBUTION")
    print("=" * 80)
    for domain, target in DOMAIN_TARGETS.items():
        stats = domain_stats[domain]
        rate = stats["accepted"] / stats["read"] * 100 if stats["read"] else 0
        print(f"{domain.upper()}")
        print(f"  Target        : {target:,}")
        print(f"  Accepted      : {stats['accepted']:,}")
        print(f"  Read          : {stats['read']:,}")
        print(f"  Rejected      : {stats['rejected']:,}")
        print(f"  Duplicates    : {stats['duplicates']:,}")
        print(f"  Acceptance    : {rate:.2f}%")
        print()

    if accepted:
        print("LANGUAGE DISTRIBUTION")
        print("=" * 80)
        for language, count in Counter(r["detected_language"] for r in accepted).most_common():
            print(f"{language:<30}{count:,}")

        print()
        print("SENTIMENT DISTRIBUTION")
        print("=" * 80)
        for sentiment, count in Counter(r["polarity"] for r in accepted).most_common():
            print(f"{sentiment:<15}{count:,}")

        print()
        print("REPLY DISTRIBUTION")
        print("=" * 80)
        for reply, count in Counter(r["is_reply"] for r in accepted).most_common():
            print(f"{str(reply):<10}{count:,}")

    print()
    print("REJECTION REASONS")
    print("=" * 80)
    for domain in DOMAIN_TARGETS:
        print(domain.upper())
        counter = rejection_reasons[domain]
        if not counter:
            print("  None")
        else:
            for reason, count in counter.most_common():
                print(f"  {reason:<35}{count:,}")
        print()

    print("FILES CREATED")
    print("=" * 80)
    for filename in [
        ACCEPTED_CSV,
        ACCEPTED_XLSX,
        REJECTED_CSV,
        CHECKPOINT_FILE,
    ]:
        if os.path.exists(filename):
            print(f"  {filename}")

    print()
    print(f"Program started : {collection_start.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"Program ended   : {collection_end.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print()

    if stop_reason == STATUS_TARGET_REACHED:
        print("SUCCESS: 50,000-POST TARGET REACHED.")
    elif stop_reason == STATUS_CREDITS_DEPLETED:
        print("STOPPED SAFELY: X API CREDITS DEPLETED (HTTP 402).")
        print("The saved checkpoint can be resumed after credits are added.")
    elif stop_reason == STATUS_QUERY_EXHAUSTED:
        print("STOPPED: ALL CONFIGURED QUERIES WERE EXHAUSTED.")
    else:
        print(f"STOPPED: {stop_reason}")

    print("=" * 80)


# ======================================================================
# 13. RUN
# ======================================================================

if __name__ == "__main__":
    main()

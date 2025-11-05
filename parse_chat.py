"""
Enhanced chat parser with better persona analysis.
Processes WhatsApp exports and creates training-ready data.
"""

import re
import json
import uuid
import os
from datetime import datetime
from collections import defaultdict, Counter
import statistics
from config import Config


def parse_whatsapp_export(
    input_file: str, output_dir: str = None, contact_map: dict = None
):
    """
    Parse WhatsApp chat export and create training data.

    Args:
        input_file: Path to WhatsApp .txt export
        output_dir: Where to save processed files
        contact_map: Optional dict to normalize contact names
    """

    if output_dir is None:
        output_dir = Config.PROCESSED_DIR

    os.makedirs(output_dir, exist_ok=True)

    if contact_map is None:
        contact_map = {}

    # Try to match common WhatsApp export formats
    line_re = re.compile(
        r"^(?P<date>[\d/.]+)[,\s]+(?P<time>\d{1,2}:\d{2}(?:\s?[APMapm]{2})?)\s*-\s*(?:(?P<sender>[^:]+):\s*)?(?P<msg>.*)$"
    )

    def parse_dt(date_str, time_str):
        """Parse datetime with multiple format attempts."""
        candidates = [
            "%m/%d/%y %I:%M %p",
            "%m/%d/%Y %I:%M %p",
            "%d/%m/%y %I:%M %p",
            "%d/%m/%Y %I:%M %p",
            "%d.%m.%Y %H:%M",
            "%d.%m.%y %H:%M",
            "%d/%m/%Y %H:%M",
            "%m/%d/%Y %H:%M",
            "%m/%d/%y %H:%M",
        ]
        combined = f"{date_str} {time_str}"
        for fmt in candidates:
            try:
                return datetime.strptime(combined, fmt).isoformat()
            except:
                pass
        return combined  # fallback

    # Parse messages
    messages = []
    senders = set()

    print(f"Parsing {input_file}...")

    with open(input_file, encoding="utf-8") as f:
        current_msg = None
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            match = line_re.match(line)
            if match:
                # New message
                d = match.groupdict()
                dt = parse_dt(d["date"], d["time"])
                sender = d.get("sender") or "SYSTEM"
                text = d["msg"].strip()

                current_msg = {
                    "id": str(uuid.uuid4()),
                    "datetime": dt,
                    "sender": sender.strip(),
                    "text": text,
                }
                messages.append(current_msg)
                senders.add(sender.strip())
            else:
                # Continuation of previous message (multiline)
                if current_msg:
                    current_msg["text"] += "\n" + line.strip()

    print(f"Parsed {len(messages)} messages from {len(senders)} senders")

    # Normalize sender names
    for msg in messages:
        original = msg["sender"]
        normalized = contact_map.get(original, original)
        msg["persona_id"] = normalized

    # Calculate persona statistics
    persona_stats = analyze_personas(messages)

    # Save messages.jsonl
    messages_path = os.path.join(output_dir, "messages.jsonl")
    with open(messages_path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(messages)} messages to {messages_path}")

    # Save personas.json
    personas_path = os.path.join(output_dir, "personas.json")
    with open(personas_path, "w", encoding="utf-8") as f:
        json.dump(persona_stats, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(persona_stats)} personas to {personas_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PERSONA SUMMARY")
    print("=" * 60)
    for name, stats in sorted(persona_stats.items(), key=lambda x: -x[1]["n_messages"]):
        print(
            f"{name:20} | {stats['n_messages']:4} msgs | "
            f"avg {stats['avg_words']:.1f} words | "
            f"{stats['emoji_count_total']:3} emojis"
        )
    print("=" * 60)

    return messages, persona_stats


def analyze_personas(messages):
    """
    Analyze writing style for each persona.

    Returns dict with stats: avg_len, avg_words, emoji_count, etc.
    """
    by_sender = defaultdict(list)

    for msg in messages:
        by_sender[msg["persona_id"]].append(msg["text"])

    persona_stats = {}

    # Add stopwords to filter out
    STOPWORDS = {
        "<medien",
        "ausgeschlossen>",
        "medien",
        "ausgeschlossen",
        "omitted>",
        "<media",
        "omitted",  # English exports
        "omitido>",
        "<multimedia",  # Spanish exports
    }

    for sender, texts in by_sender.items():
        # Basic stats
        lengths = [len(t) for t in texts]
        words = [len(t.split()) for t in texts]

        # Emoji counting (heuristic: high unicode characters)
        emoji_count = sum(
            1 for text in texts for char in text if ord(char) > 0x1F300  # Emoji range
        )

        # Punctuation analysis
        exclamation_count = sum(text.count("!") for text in texts)
        question_count = sum(text.count("?") for text in texts)

        # Common words (excluding very short ones)
        all_words = " ".join(texts).lower().split()
        word_freq = Counter(
            w
            for w in all_words
            if len(w) > 3 and w not in STOPWORDS  # Add stopwords filter
        )
        common_words = [w for w, _ in word_freq.most_common(10)]

        persona_stats[sender] = {
            "name": sender,
            "n_messages": len(texts),
            "avg_len": statistics.mean(lengths) if lengths else 0,
            "median_len": statistics.median(lengths) if lengths else 0,
            "avg_words": statistics.mean(words) if words else 0,
            "median_words": statistics.median(words) if words else 0,
            "emoji_count_total": emoji_count,
            "emoji_rate": emoji_count / len(texts) if texts else 0,
            "exclamation_rate": exclamation_count / len(texts) if texts else 0,
            "question_rate": question_count / len(texts) if texts else 0,
            "common_words": common_words,
        }

    return persona_stats


if __name__ == "__main__":
    import sys

    # Default contact mapping (customize for your chat)
    CONTACT_MAP = {
        "Goso": "Goso",
        "Gomez": "Gómez",
        "Juan 2": "Juan V.",
        "Emilio": "Emi",
        "Edu Móvil": "Edu",
        "Fer Movil": "Fer",
        "SYSTEM": "SYSTEM",
        "Ernes Mvl": "Ernes",
        "Coronga": "Cora",
        "Jose A": "Jose",
        "Héctor Dutch": "Héctor",
        "Juan 1 England": "Juan L.",
        "Violeta Rami": "Violeta",
        "Sara SS SS SS SS SS SS J2": "Sara (J2)",
    }

    # Get input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Try to find a chat file
        import glob

        chat_files = glob.glob("data/WhatsApp*.txt")
        if not chat_files:
            print("No WhatsApp chat files found in data/")
            print("Usage: python parse_chat.py <path_to_chat.txt>")
            sys.exit(1)

        # Use the largest file (assuming it's the main chat)
        input_file = max(chat_files, key=os.path.getsize)
        print(f"Using: {input_file}")

    # Parse
    messages, personas = parse_whatsapp_export(
        input_file, output_dir=Config.PROCESSED_DIR, contact_map=CONTACT_MAP
    )

    print("\n✓ Processing complete!")
    print("  Next step: python train_model.py")

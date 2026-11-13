"""Public-domain English corpus loader for Ori pretraining.

The bootstrap language model learns general English from a small, reproducible
set of public-domain Project Gutenberg texts before Ori-specific fine-tuning.
"""

from __future__ import annotations

import re
from urllib.request import Request, urlopen


PUBLIC_DOMAIN_SOURCES = (
    ("alice", "https://www.gutenberg.org/files/11/11-0.txt"),
    ("oz", "https://www.gutenberg.org/files/55/55-0.txt"),
    ("sherlock", "https://www.gutenberg.org/files/1661/1661-0.txt"),
    ("tom_sawyer", "https://www.gutenberg.org/files/74/74-0.txt"),
    ("pride_prejudice", "https://www.gutenberg.org/files/1342/1342-0.txt"),
    ("frankenstein", "https://www.gutenberg.org/files/84/84-0.txt"),
    ("jane_eyre", "https://www.gutenberg.org/files/1260/1260-0.txt"),
    ("little_women", "https://www.gutenberg.org/files/514/514-0.txt"),
)


def _download(url: str) -> str:
    request = Request(
        url,
        headers={"User-Agent": "Ori-TensorFlow-Pretraining/0.1"},
    )
    with urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def _strip_gutenberg(text: str) -> str:
    text = text.replace("\r", "")
    start = re.search(r"\*\*\* START OF (?:THE )?PROJECT GUTENBERG", text, re.IGNORECASE)
    end = re.search(r"\*\*\* END OF (?:THE )?PROJECT GUTENBERG", text, re.IGNORECASE)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start()]
    return text


def _paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    for raw in re.split(r"\n\s*\n", _strip_gutenberg(text)):
        paragraph = re.sub(r"\s+", " ", raw).strip()
        if not paragraph:
            continue
        if paragraph.startswith("Project Gutenberg"):
            continue
        if 40 <= len(paragraph) <= 2400:
            paragraphs.append(paragraph)
    return paragraphs



def everyday_english() -> list[str]:
    """Generate varied, grammar-safe modern English and short dialogue."""
    sentences: list[str] = []

    people = ["I", "you", "we", "they"]
    actions = [
        "need", "want", "like", "use", "check", "open", "close", "read",
        "write", "test", "build", "learn", "understand", "remember",
        "compare", "review", "change", "save", "find", "fix",
    ]
    objects = [
        "the file", "the project", "the code", "the result", "the answer",
        "the system", "the settings", "the device", "the program",
        "the model", "the data", "the task", "the page", "the folder",
        "the design", "the report", "the test", "the document",
    ]

    for subject in people:
        for action in actions:
            for obj in objects:
                sentences.append(f"{subject} {action} {obj}.")
                sentences.append(f"{subject} can {action} {obj}.")
                sentences.append(f"{subject} will {action} {obj}.")
                question_subject = subject if subject == "I" else subject.lower()
                sentences.append(f"Can {question_subject} {action} {obj}?")
                sentences.append(f"Will {question_subject} {action} {obj}?")
                if subject in {"I", "you"}:
                    sentences.append(f"Do {question_subject} {action} {obj}?")

    singular_subjects = [
        "the user", "the program", "the model", "the server",
        "the computer", "the printer", "the browser", "the application",
        "the script", "the tool", "the device", "the project",
    ]
    singular_actions = [
        ("needs", "the file"), ("uses", "the data"), ("checks", "the result"),
        ("opens", "the project"), ("closes", "the file"), ("reads", "the code"),
        ("writes", "the output"), ("tests", "the program"), ("builds", "the project"),
        ("learns", "from examples"), ("explains", "the answer"), ("creates", "a file"),
        ("changes", "the settings"), ("finds", "the problem"), ("helps", "the user"),
        ("saves", "the result"), ("loads", "the model"), ("shows", "the page"),
        ("updates", "the document"), ("starts", "the task"),
    ]
    for subject in singular_subjects:
        for verb, obj in singular_actions:
            base = verb[:-1] if verb.endswith("s") else verb
            sentences.append(f"{subject} {verb} {obj}.")
            sentences.append(f"Does {subject} {base} {obj}?")
            sentences.append(f"The {subject[4:] if subject.startswith('the ') else subject} can {base} {obj}.")
            sentences.append(f"The {subject[4:] if subject.startswith('the ') else subject} will {base} {obj} later.")

    states = [
        ("The model", "is ready"), ("The server", "is online"),
        ("The program", "is running"), ("The file", "is available"),
        ("The result", "is correct"), ("The printer", "is paused"),
        ("The project", "is active"), ("The system", "is working"),
        ("The computer", "is connected"), ("The browser", "is open"),
        ("The task", "is complete"), ("The test", "is ready"),
        ("The page", "is loading"), ("The device", "is connected"),
    ]
    for subject, state in states:
        adjective_phrase = state[3:]
        sentences.append(f"{subject} {state}.")
        sentences.append(f"Is {subject.lower()} {adjective_phrase}?")
        sentences.append(f"I see that {subject.lower()} {adjective_phrase}.")
        sentences.append(f"We can check whether {subject.lower()} {adjective_phrase}.")

    topics = [
        "Python", "JavaScript", "HTML", "CSS", "JSON", "an API",
        "a database", "a server", "a browser", "a neural network",
        "machine learning", "a 3D printer", "a 3D model", "a file",
        "a program", "a project", "a computer", "a camera", "a printer",
        "a model", "a dataset", "a web app", "a script",
    ]
    for topic in topics:
        sentences.extend([
            f"What is {topic}?",
            f"Can you explain {topic}?",
            f"I want to learn about {topic}.",
            f"I have a question about {topic}.",
            f"I am working with {topic}.",
            f"I need help with {topic}.",
            f"How does {topic} work?",
            f"Why is {topic} useful?",
            f"What can {topic} do?",
            f"Can {topic} be changed?",
            f"Where is {topic} used?",
            f"When should I use {topic}?",
            f"How can I test {topic}?",
            f"Can we improve {topic}?",
        ])

    conversational = [
        ("Hello.", "Hi! How can I help?"),
        ("Hi!", "Hey! What are you working on?"),
        ("Hey!", "Hey! What can we work on?"),
        ("Hi there.", "Hey! What can I help with?"),
        ("Hello there.", "Hi! What are you working on today?"),
        ("Good morning.", "Good morning! What are we working on?"),
        ("Good afternoon.", "Good afternoon! What should we work on?"),
        ("Good evening.", "Good evening! What can we figure out?"),
        ("Hi.", "Hey! What are you working on?"),
        ("Hey there.", "Hey! What's up?"),
        ("Good morning.", "Good morning! How is your day going?"),
        ("Good afternoon.", "Good afternoon! What are you working on?"),
        ("Good evening.", "Good evening! What can I help with?"),
        ("Are you there?", "Yes, I am here."),
        ("Are you listening?", "Yes, I am following the conversation."),
        ("Can you help me?", "Sure. Tell me what you need."),
        ("I have a question.", "Go ahead."),
        ("I need some help.", "Okay. Tell me what you are trying to do."),
        ("I am stuck.", "That is okay. Tell me where you got stuck."),
        ("I do not understand.", "No problem. Tell me which part is confusing."),
        ("Can you explain that?", "Absolutely. I will explain it step by step."),
        ("Can you say that another way?", "Sure. I can explain it more simply."),
        ("What should I do next?", "Start with the result you want, then take the next useful step."),
        ("Thanks.", "You are welcome."),
        ("Thank you.", "You are welcome!"),
        ("That helped.", "Glad it helped."),
        ("Let's keep going.", "Sounds good."),
        ("What happened?", "Let's look at what changed."),
        ("Is this correct?", "We can check it carefully."),
        ("I finished that.", "Nice. What do you want to work on next?"),
        ("Can we try again?", "Yes. We can try a different approach."),
    ]
    for user_text, reply in conversational:
        sentences.append(user_text)
        sentences.append(reply)

    contexts = [
        "before the test", "after the update", "this morning", "this afternoon",
        "this evening", "tomorrow", "today", "later", "right now",
        "during the build", "after the change", "before we continue",
    ]
    tasks = [
        "check the result", "open the project", "read the file", "test the code",
        "save the document", "review the settings", "compare the results",
        "fix the problem", "run the test", "check the connection",
        "update the page", "inspect the model", "review the design",
    ]
    for person in people:
        for task in tasks:
            for context in contexts:
                sentences.append(f"{person} can {task} {context}.")
                sentences.append(f"{person} should {task} {context}.")
                sentences.append(f"{person} will {task} {context}.")
                question_person = person if person == "I" else person.lower()
                sentences.append(f"Can {question_person} {task} {context}?")

    common = [
        "I don't know.", "I can't find it.", "I can't open the file.",
        "Let's try again.", "That makes sense.", "I understand.",
        "I need more information.", "Please explain that.",
        "What happened?", "Why did that happen?", "That looks good.",
        "The result is ready.", "The program is running.",
        "The model needs more training.", "The file was saved.",
        "The test passed.", "The test failed.", "The printer is online.",
        "The printer is paused.", "I am working on a project.",
        "We can solve this problem.", "Let's check the result.",
        "Please show me the code.", "I will check it carefully.",
        "We should test the change.", "The next step is clear.",
        "The answer depends on the details.", "Let's look at the evidence.",
    ]
    sentences.extend(common)

    # Keep enough variety to fill the modern-language budget instead of
    # silently falling back to mostly historical prose.
    more_subjects = ["I", "you", "we", "they", "the user", "the program", "the model"]
    more_verbs = [
        "start", "stop", "review", "check", "open", "close", "save", "load",
        "build", "test", "explain", "compare", "change", "finish", "continue",
    ]
    more_nouns = [
        "the task", "the project", "the file", "the result", "the example",
        "the answer", "the page", "the design", "the code", "the settings",
    ]
    connectors = [
        "before continuing", "after checking the result", "when the test is ready",
        "after the file is saved", "when the project is open",
    ]
    for subject in more_subjects:
        for verb in more_verbs:
            for noun in more_nouns:
                for connector in connectors:
                    if subject in {"I", "you", "we", "they"}:
                        sentences.append(f"{subject} will {verb} {noun} {connector}.")
                    else:
                        base = verb
                        sentences.append(f"{subject} will {base} {noun} {connector}.")
                    if len(sentences) >= 14000:
                        return sentences

    return sentences


def download_public_domain_english(max_chars: int = 360_000) -> list[str]:
    """Return a diverse bounded English corpus with guaranteed modern-language exposure."""

    if max_chars < 10_000:
        raise ValueError("max_chars must be at least 10000")

    book_budget = int(max_chars * 0.70)
    modern_budget = max_chars - book_budget
    per_source = max(20_000, book_budget // len(PUBLIC_DOMAIN_SOURCES))
    collected: list[str] = []
    total = 0

    for name, url in PUBLIC_DOMAIN_SOURCES:
        try:
            paragraphs = _paragraphs(_download(url))
        except Exception as exc:
            print(f"English source unavailable ({name}): {exc}")
            continue

        source_chars = 0
        for paragraph in paragraphs:
            if source_chars >= per_source or total >= book_budget:
                break
            remaining = min(per_source - source_chars, book_budget - total)
            if len(paragraph) > remaining:
                paragraph = paragraph[:remaining].rsplit(" ", 1)[0]
            if len(paragraph) < 40:
                continue
            collected.append(paragraph)
            source_chars += len(paragraph)
            total += len(paragraph)

        if total >= book_budget:
            break

    if not collected:
        raise RuntimeError("Could not download any English pretraining text.")

    modern = everyday_english()
    modern_used = 0
    for sentence in modern:
        if modern_used >= modern_budget:
            break
        remaining = modern_budget - modern_used
        text = sentence if len(sentence) <= remaining else sentence[:remaining].rsplit(" ", 1)[0]
        if len(text) < 10:
            continue
        collected.append(text)
        total += len(text)
        modern_used += len(text)

    print(
        f"English pretraining corpus: {len(collected)} passages, "
        f"{total:,} characters."
    )
    return collected

"""
NeuroAdapt Curriculum
----------------------
Real question content across subjects and difficulty levels.
Each question has a preferred modality (how it's best presented)
and a subject area. This grounds the environment in actual learning,
not abstract float manipulation.
"""

from dataclasses import dataclass, field
from typing import List
import random


@dataclass
class Question:
    id: str
    subject: str
    difficulty: int          # 1-5
    text: str                # The question shown to child
    options: List[str]       # Multiple choice options
    correct_index: int       # Index of correct answer
    preferred_modality: str  # 'visual' | 'audio' | 'text' | 'mixed'
    hint: str                # Hint shown at hint_level >= 2


CURRICULUM: List[Question] = [

    # ── MATH ─────────────────────────────────────────────────────────── #
    Question("m1", "math", 1, "How many apples are in 2 groups of 3?",
             ["4", "5", "6", "7"], 2, "visual",
             "Think of 3 apples in each hand."),
    Question("m2", "math", 1, "What is 5 + 3?",
             ["7", "8", "9", "10"], 1, "text",
             "Count 5 fingers then 3 more."),
    Question("m3", "math", 2, "Which number comes after 19?",
             ["18", "20", "21", "22"], 1, "visual",
             "Count on your fingers from 19."),
    Question("m4", "math", 2, "What is 4 × 3?",
             ["10", "11", "12", "13"], 2, "visual",
             "Draw 4 groups of 3 dots."),
    Question("m5", "math", 3, "If you have 24 stickers and share equally among 4 friends, how many each?",
             ["4", "5", "6", "8"], 2, "visual",
             "Divide 24 into 4 equal groups."),
    Question("m6", "math", 3, "What is 15% of 60?",
             ["6", "7", "9", "10"], 2, "text",
             "15% means 15 out of 100. So 15÷100 × 60."),
    Question("m7", "math", 4, "Solve: 3x + 7 = 22",
             ["3", "4", "5", "6"], 2, "text",
             "Subtract 7 from both sides first."),
    Question("m8", "math", 5, "What is the area of a circle with radius 4? (use π≈3.14)",
             ["25.12", "50.24", "12.56", "75.36"], 1, "visual",
             "Area = π × r². Plug in r=4."),

    # ── READING ──────────────────────────────────────────────────────── #
    Question("r1", "reading", 1, "Which word rhymes with 'cat'?",
             ["dog", "bat", "cup", "pin"], 1, "audio",
             "Say each word out loud and listen."),
    Question("r2", "reading", 1, "What letter does 'sun' start with?",
             ["a", "s", "t", "n"], 1, "visual",
             "Look at the first letter of the word."),
    Question("r3", "reading", 2, "What is the opposite of 'happy'?",
             ["sad", "glad", "loud", "fast"], 0, "text",
             "Think of how you feel when something bad happens."),
    Question("r4", "reading", 2, "Which sentence is correct?",
             ["He go to school.", "He goes to school.", "He going school.", "He goed school."], 1, "text",
             "He is one person, so add -es or -s to the verb."),
    Question("r5", "reading", 3, "In a story, what does 'setting' mean?",
             ["The main character", "Where and when the story happens",
              "The problem in the story", "How the story ends"], 1, "text",
             "Think: WHERE are they? WHEN is it?"),
    Question("r6", "reading", 4, "What is a metaphor?",
             ["A comparison using 'like' or 'as'",
              "A direct comparison without 'like' or 'as'",
              "A word that sounds like what it means",
              "A type of rhyme scheme"], 1, "text",
             "Remember: simile uses 'like/as', metaphor does not."),

    # ── SCIENCE ──────────────────────────────────────────────────────── #
    Question("s1", "science", 1, "What do plants need to grow?",
             ["Only water", "Water, sunlight, and soil",
              "Only sunlight", "Water and darkness"], 1, "visual",
             "Think about what you give a plant at home."),
    Question("s2", "science", 1, "What sound does a dog make?",
             ["Meow", "Bark", "Moo", "Quack"], 1, "audio",
             "Close your eyes and imagine a dog."),
    Question("s3", "science", 2, "What state of matter is water at room temperature?",
             ["Solid", "Gas", "Liquid", "Plasma"], 2, "visual",
             "Think about a glass of water sitting on a table."),
    Question("s4", "science", 3, "Why do we have day and night?",
             ["The sun turns off at night",
              "The Earth rotates on its axis",
              "The moon blocks the sun",
              "Clouds cover the sun"], 1, "visual",
             "Think of a spinning ball with a flashlight shining on it."),
    Question("s5", "science", 4, "What is photosynthesis?",
             ["How animals breathe",
              "How plants make food using sunlight",
              "How water evaporates",
              "How rocks form"], 1, "visual",
             "Photo = light. Synthesis = making. Plants make food from light."),
    Question("s6", "science", 5, "What force keeps planets in orbit around the sun?",
             ["Magnetism", "Friction", "Gravity", "Electricity"], 2, "visual",
             "Isaac Newton discovered this when an apple fell."),

    # ── LIFE SKILLS ──────────────────────────────────────────────────── #
    Question("l1", "life_skills", 1, "What do you say when someone gives you a gift?",
             ["Nothing", "Thank you", "Give me more", "Why?"], 1, "visual",
             "Think about how it makes someone feel when you appreciate them."),
    Question("l2", "life_skills", 2, "You bump into someone by accident. What do you say?",
             ["Run away", "Sorry!", "It's your fault", "Nothing"], 1, "visual",
             "An accident isn't your fault but kindness helps."),
    Question("l3", "life_skills", 3, "Your friend is crying. What might help most?",
             ["Laugh at them", "Ignore them",
              "Ask if they are okay and listen", "Tell them to stop"], 2, "visual",
             "Think about what YOU would want someone to do if you were sad."),
]


def get_question(difficulty: int, subject: str = None) -> Question:
    """Return a random question at the given difficulty, optionally filtered by subject."""
    pool = [q for q in CURRICULUM if q.difficulty == difficulty]
    if subject:
        subpool = [q for q in pool if q.subject == subject]
        if subpool:
            pool = subpool
    if not pool:
        # Fallback to nearest difficulty
        pool = sorted(CURRICULUM, key=lambda q: abs(q.difficulty - difficulty))[:3]
    return random.choice(pool)


def get_hint(question: Question, hint_level: int) -> str:
    """Return appropriate hint text based on hint level."""
    if hint_level == 0:
        return ""
    elif hint_level == 1:
        return f"There are {len(question.options)} choices."
    elif hint_level >= 2:
        return question.hint

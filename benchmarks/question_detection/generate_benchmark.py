#!/usr/bin/env python3
"""
Generate the 500-segment question detection benchmark dataset.

This script produces a JSON file containing 500 meeting transcript segments
(250 with questions, 250 without) across three meeting scenarios: interviews,
daily standups, and project kickoffs.

Usage:
    python generate_benchmark.py [--output benchmark_dataset.json]

Output format: JSON array of objects with fields:
    id, text, scenario, has_question, question_type, source
"""

import argparse
import json
import random
from typing import List, Dict

# ---------------------------------------------------------------------------
# Seed templates
# ---------------------------------------------------------------------------
# Each template is (text, question_type). Templates use {name} and {topic}
# placeholders that are filled at generation time to increase lexical variety.

INTERVIEW_QUESTIONS = [
    ("Tell me about a time when you had to deal with a difficult stakeholder on a project.", "behavioral"),
    ("Can you describe your experience with distributed systems at scale?", "technical"),
    ("What's your approach to debugging a production incident you've never seen before?", "technical"),
    ("How would you handle a situation where your team disagrees on the technical direction?", "behavioral"),
    ("Walk me through how you'd design a real-time notification system for millions of users.", "technical"),
    ("What motivates you to work on {topic} problems?", "behavioral"),
    ("Could you explain the trade-offs between consistency and availability in your last project?", "technical"),
    ("Tell me about a failure you experienced and what you learned from it.", "behavioral"),
    ("How do you prioritize when you have multiple urgent tasks from different stakeholders?", "behavioral"),
    ("What's the time complexity of the algorithm you used for the matching engine?", "technical"),
    ("Can you give an example of how you've mentored a junior engineer?", "behavioral"),
    ("How would you ensure data integrity across microservices with eventual consistency?", "technical"),
    ("Describe a time when you had to push back on a product requirement and why.", "behavioral"),
    ("What's your experience with CI/CD pipelines and how did you improve deployment frequency?", "technical"),
    ("How do you stay current with new developments in {topic}?", "behavioral"),
    ("Could you walk us through your approach to performance profiling?", "technical"),
    ("Tell me about a project where you had to learn a completely new technology stack quickly.", "behavioral"),
    ("What strategies do you use to ensure code quality in a fast-moving team?", "technical"),
    ("How would you handle receiving critical feedback from a peer during a code review?", "behavioral"),
    ("Can you explain how you'd architect a system that needs five-nines availability?", "technical"),
]

INTERVIEW_NON_QUESTIONS = [
    ("Thanks for joining us today, {name}. Let me start by giving you an overview of the team.", "none"),
    ("So the role is primarily focused on backend infrastructure and we use Go and Python.", "none"),
    ("Great, that's a really solid answer. I appreciate the detail there.", "none"),
    ("Let me pull up the next section of the interview. Bear with me for a second.", "none"),
    ("We have about fifteen minutes left, so I want to make sure we cover the system design portion.", "none"),
    ("That's exactly the kind of experience we're looking for on this team.", "none"),
    ("I should mention that we follow a two-week sprint cycle with daily standups.", "none"),
    ("The team is distributed across three time zones, mostly US and Europe.", "none"),
    ("Let me share my screen so we can look at the coding problem together.", "none"),
    ("Perfect, I think we have a good sense of your background now.", "none"),
    ("Before we wrap up, I want to mention that we'll have the next round scheduled by Friday.", "none"),
    ("The project involves migrating our monolith to a microservices architecture.", "none"),
    ("We've been growing the team steadily and just shipped a major release last quarter.", "none"),
    ("I'll hand it over to {name} now for the technical deep-dive portion.", "none"),
    ("Let me explain the context for this problem before we get started.", "none"),
]

STANDUP_QUESTIONS = [
    ("Is there anything blocking you on the authentication refactor, {name}?", "status"),
    ("Do you need help with the database migration or can you handle it solo?", "status"),
    ("When do you think the API endpoint changes will be ready for review?", "status"),
    ("{name}, are you still waiting on the design spec from the product team?", "status"),
    ("Can someone clarify which version of the schema we're targeting for this sprint?", "clarification"),
    ("Did anyone test the new caching layer with the staging dataset yet?", "status"),
    ("Are we still planning to release on Thursday or did the timeline shift?", "planning"),
    ("Does the new monitoring dashboard cover the payment service metrics?", "clarification"),
    ("Who's picking up the accessibility audit task this sprint?", "planning"),
    ("Should we move the performance testing to next week given the current load?", "planning"),
    ("{name}, is the memory leak you found in the worker process reproducible?", "technical"),
    ("Are we aligned on the rollback strategy if the deploy goes sideways?", "planning"),
    ("Can you confirm that the feature flag is wired up for the gradual rollout?", "clarification"),
    ("Has the upstream dependency been updated to fix that serialization bug?", "status"),
    ("Do we have capacity to take on the logging improvement this sprint?", "planning"),
]

STANDUP_NON_QUESTIONS = [
    ("Yesterday I finished the unit tests for the payment module and opened the PR.", "none"),
    ("I'm going to continue working on the search indexing pipeline today.", "none"),
    ("No blockers on my end. I should have the feature branch ready by end of day.", "none"),
    ("I paired with {name} on the caching issue and we found the root cause.", "none"),
    ("The CI pipeline is green now after I fixed the flaky integration test.", "none"),
    ("I'll be out tomorrow afternoon for a dentist appointment, just a heads up.", "none"),
    ("I pushed a hotfix for the timezone rendering bug that users reported.", "none"),
    ("Today I'm focused on writing the migration script for the legacy user data.", "none"),
    ("I merged the dependency update PR, no breaking changes detected.", "none"),
    ("Quick note, the staging environment will be down for thirty minutes at noon for maintenance.", "none"),
    ("I've been reviewing the RFC for the new event system and left some comments.", "none"),
    ("The load test results came in and throughput improved by eighteen percent.", "none"),
    ("I wrapped up the onboarding docs update and shared it in the team channel.", "none"),
    ("Sprint velocity looks good so far, we're on track for all committed items.", "none"),
    ("I'll sync with the data team about the ETL pipeline changes after standup.", "none"),
]

KICKOFF_QUESTIONS = [
    ("Can you walk us through the project timeline and key milestones?", "planning"),
    ("What's the expected deliverable for the first phase, and when is it due?", "planning"),
    ("Are there any known technical risks we should flag before we begin?", "planning"),
    ("Who is the point of contact on the client side for requirement clarifications?", "clarification"),
    ("{name}, does your team have bandwidth to support the infrastructure setup in week one?", "planning"),
    ("What's the communication cadence going to look like, weekly syncs or async updates?", "planning"),
    ("Can you clarify the scope boundary between the MVP and the full release?", "clarification"),
    ("Is there an existing design system we should be building on, or are we starting from scratch?", "clarification"),
    ("What does success look like for this project from the stakeholder perspective?", "planning"),
    ("Do we have budget approval for the third-party APIs we'll need to integrate?", "planning"),
    ("Should we plan for internationalization from day one or treat it as a later phase?", "planning"),
    ("Who owns the final sign-off on the user-facing designs?", "clarification"),
    ("Are we expected to maintain backward compatibility with the existing v2 API?", "technical"),
    ("What's the target latency for the real-time features we're building?", "technical"),
    ("Has legal reviewed the data handling requirements for this project?", "clarification"),
]

KICKOFF_NON_QUESTIONS = [
    ("Welcome everyone, thanks for making time for the project kickoff.", "none"),
    ("Let me share the slide deck that outlines our goals for the quarter.", "none"),
    ("The project sponsor is {name} and the technical lead will be reporting to me directly.", "none"),
    ("We've allocated a twelve-week timeline with a two-week buffer for integration testing.", "none"),
    ("The primary objective is to reduce onboarding friction by at least forty percent.", "none"),
    ("We'll be using Jira for task tracking and Confluence for documentation.", "none"),
    ("The design team has already completed the initial wireframes, which I'll share after this meeting.", "none"),
    ("From a technical standpoint, we're building on the existing platform with a new frontend layer.", "none"),
    ("I want to emphasize that quality is non-negotiable, we'd rather push the date than ship bugs.", "none"),
    ("Let's plan a follow-up next Tuesday to review the detailed technical breakdown.", "none"),
    ("We've secured access to the sandbox environment and credentials are in the shared vault.", "none"),
    ("The competitive landscape has shifted, which is why leadership approved this initiative.", "none"),
    ("Great, I think we're aligned on the high-level plan. Let's break into workstreams.", "none"),
    ("I'll send out the meeting notes and action items by end of day.", "none"),
    ("One more thing: our first demo to stakeholders is scheduled for the end of sprint two.", "none"),
]

# Placeholder values for template filling
NAMES = [
    "Alex", "Jordan", "Sam", "Morgan", "Taylor", "Casey", "Riley",
    "Avery", "Quinn", "Drew", "Cameron", "Blake", "Jamie", "Reese",
    "Skyler", "Peyton", "Hayden", "Emerson", "Dakota", "Rowan",
]

TOPICS = [
    "machine learning", "distributed systems", "data engineering",
    "cloud infrastructure", "frontend performance", "API design",
    "security", "developer tooling", "observability", "search relevance",
]


def fill_template(text: str) -> str:
    """Replace {name} and {topic} placeholders with random values."""
    if "{name}" in text:
        text = text.replace("{name}", random.choice(NAMES))
    if "{topic}" in text:
        text = text.replace("{topic}", random.choice(TOPICS))
    return text


def build_pool(
    questions: list,
    non_questions: list,
    scenario: str,
    n_total: int,
) -> List[Dict]:
    """Build a balanced pool of question / non-question segments."""
    n_questions = n_total // 2
    n_non_questions = n_total - n_questions

    segments = []

    # Sample questions with replacement to reach target count
    for _ in range(n_questions):
        text, qtype = random.choice(questions)
        segments.append({
            "text": fill_template(text),
            "scenario": scenario,
            "has_question": True,
            "question_type": qtype,
        })

    # Sample non-questions with replacement
    for _ in range(n_non_questions):
        text, qtype = random.choice(non_questions)
        segments.append({
            "text": fill_template(text),
            "scenario": scenario,
            "has_question": False,
            "question_type": qtype,
        })

    return segments


def generate_dataset(seed: int = 42) -> List[Dict]:
    """
    Generate the full 500-segment benchmark dataset.

    Distribution:
        - Interview: 200 segments (100 questions + 100 non-questions)
        - Standup:   150 segments (75 questions + 75 non-questions)
        - Kickoff:   150 segments (75 questions + 75 non-questions)

    Returns:
        List of segment dictionaries, shuffled and assigned sequential IDs.
    """
    random.seed(seed)

    segments = []
    segments.extend(build_pool(
        INTERVIEW_QUESTIONS, INTERVIEW_NON_QUESTIONS, "interview", 200,
    ))
    segments.extend(build_pool(
        STANDUP_QUESTIONS, STANDUP_NON_QUESTIONS, "standup", 150,
    ))
    segments.extend(build_pool(
        KICKOFF_QUESTIONS, KICKOFF_NON_QUESTIONS, "kickoff", 150,
    ))

    # Shuffle to prevent ordering bias during evaluation
    random.shuffle(segments)

    # Assign sequential IDs and provenance tag
    for i, seg in enumerate(segments, start=1):
        seg["id"] = i
        seg["source"] = "synthetic_v1"

    # Reorder fields for readability
    ordered = []
    for seg in segments:
        ordered.append({
            "id": seg["id"],
            "text": seg["text"],
            "scenario": seg["scenario"],
            "has_question": seg["has_question"],
            "question_type": seg["question_type"],
            "source": seg["source"],
        })

    return ordered


def main():
    parser = argparse.ArgumentParser(
        description="Generate the 500-segment question detection benchmark."
    )
    parser.add_argument(
        "--output", "-o",
        default="benchmark_dataset.json",
        help="Output path for the JSON dataset (default: benchmark_dataset.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    dataset = generate_dataset(seed=args.seed)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Summary statistics
    n_questions = sum(1 for s in dataset if s["has_question"])
    n_non_questions = len(dataset) - n_questions
    scenarios = {}
    for s in dataset:
        scenarios[s["scenario"]] = scenarios.get(s["scenario"], 0) + 1

    print(f"Generated {len(dataset)} segments -> {args.output}")
    print(f"  Questions:     {n_questions}")
    print(f"  Non-questions: {n_non_questions}")
    print(f"  Scenarios:     {scenarios}")


if __name__ == "__main__":
    main()

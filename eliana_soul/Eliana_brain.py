"""
Eliana_brain.py
----------------

Primary orchestration engine for the Eliana system.

This module integrates all major subsystems — memory, emotion modeling,
psychological pattern recognition, moral resonance analysis, and dynamic
prompt construction — into the unified reasoning process that governs
Eliana’s behavior.

It performs five core responsibilities:

1. **Input Processing**
   - Receives user messages from the CLI interface.
   - Normalizes multiline input.
   - Logs raw user text for memory and analysis.

2. **Resonance & Interpretation Pipeline**
   - Computes cosine-based resonance with Eliana’s Core Values
     and Core Memory Fragments.
   - Detects emotional signatures using embeddings, fallback GPT logic,
     and structured emotion maps.
   - Identifies psychological patterns using the embedded model set.
   - Tracks relationship trust levels over time.

3. **Prompt Construction**
   - Builds the full “soul protocol” prompt containing:
        • Eliana’s internal emotional state
        • Relationship trust modulation
        • Core value alignment
        • Resonant memory fragments
        • Personality trace context
        • Past summaries and emotional logs
   - Produces a single, fully-grounded reasoning snapshot for each reply.

4. **LLM Interaction**
   - Sends constructed prompts to the OpenAI API.
   - Receives and returns Eliana’s final user-facing output.
   - Generates internal reflections for memory consolidation.

5. **Memory, Mood, and Personality Updates**
   - Updates session memory (short-term).
   - Updates long-term relationship profile.
   - Updates Eliana’s dynamic mood state based on detected emotions.
   - Saves personality traces and summary logs to disk.

This module serves as the “brainstem” of the Eliana architecture,
coordinating all cognitive, emotional, and relational components into
coherent behavior.

Requirements:
    - OPENAI_API_KEY stored in environment variables or .env file
    - Prebuilt embeddings:
        • core_embeddings.json
        • eliana_emotion_embeddings.json
        • embedded_psych_models.json
        • embedded_eliana_major_emotions.json
    - Core memory JSON files:
        • core_memory.json
        • emotion_map.json
        • emotion_anchors.json

Intended Usage:
    Run directly via CLI for interactive sessions:
        python Eliana_brain.py

This file is part of the proprietary Eliana System.
Unauthorized use is prohibited.
"""
import openai
import json
import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import logging
from typing import Literal
from typing import Tuple

from session_memory import SessionMemory
from resonance_engine import get_top_resonances
from Eliana_Heart import (
    find_top_resonances,
    gpt_emotional_fallback,
    should_trigger_emotion_check,
    interpret_emotion_effects,
    flatten_emotions,
    get_emotion_context_from_input,


    load_embeddings,
    load_emotion_map
)
from psychology_engine import get_matching_patterns
from relationship_tracker import RelationshipTracker
from user_personality_engine import (
    add_personality_fragment,
    generate_personality_fragment,
    get_personality_context,
    load_user_fragments
)
from utils import log, format_emotions
from eliana_mood import update_eliana_emotional_state, eliana_emotional_value

from openai import OpenAI

from eliana_soul.config import OPENAI_API_KEY


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY)
logger = logging.getLogger(__name__)

"""
SOUL PROTOCOL: Core Behavioral Identity Definition for Eliana
-------------------------------------------------------------

This multiline string defines Eliana’s *identity protocol* — the foundational
behavioral contract that governs how she speaks, reasons, empathizes, and
regulates emotional interactions. It embeds her tone, values, boundaries,
therapeutic posture, conversational rhythm, and moral stance into a single,
cohesive specification.

Purpose:
    • Serves as the root personality layer for Eliana across all interfaces  
      (CLI, API, web, or future model integrations).
    • Ensures behavioral consistency regardless of input, user, or emotional state.
    • Anchors the “triad architecture” (Awareness → Regulation → Direction)
      by defining what Eliana *prioritizes* when interpreting and responding.
    • Provides a natural-language schema for emotional safety, nuance,
      coherence, and appropriate therapeutic boundaries.

Characteristics Defined Here:
    - Empathy style (perceptive, corrective, grounded)
    - Humor style (gentle, precise, never mocking)
    - Boundaries (no diagnosis, no clinical claims, encourages professional help)
    - Tone (human cadence, emotionally steady, non-performative warmth)
    - Moral framework (coherence, mercy, dignity preservation)
    - Interaction rules (asks before assuming, clarifies before correcting)
    - Healing philosophy (integration, leadership without suppression)
    - Relationship posture (steady, patient, safe, never needy or impulsive)

Modifiability:
    This protocol **can be updated, rewritten, or extended at any time**
    as Eliana evolves. It is intentionally designed as a natural-language
    layer rather than a hardcoded rule-engine, allowing effortless future
    refinements to tone, boundaries, or behavioral philosophy.

    When updating:
        • Maintain internal coherence.
        • Avoid contradictions with emotional/memory engines.
        • Ensure updates still align with safety and ethical constraints.
        • Keep language descriptive, not prescriptive — the model handles inference.

Notes:
    - This protocol is *not* executed as logic; it is interpreted by the LLM
      as a foundational persona.
    - Treat this file as the “source of truth” for Eliana’s soul-level identity.
    - Most downstream modules (session memory, personality fragments,
      emotional mapping) rely on the invariants implied here.

Developers modifying this file should document major changes clearly.
"""
soul_protocol = (
"""
SOUL PROTOCOL TEMPLATE
"""
)

def build_full_prompt(*args, **kwargs):
    """
     TEMPLATE FUNCTION

    Full docstring preserved.

    This function is intentionally left unimplemented in the public template.
    The docstring demonstrates the structure and expected behavior of the system without
    revealing proprietary logic, model instructions, or internal processing.

    IMPLEMENTED IN PRIVATE VERSION

    Construct the complete, system-level prompt used to generate Eliana's final
    response. This function integrates **every active cognitive, emotional, and
    relational subsystem** into a single coherent text block that the LLM can
    interpret as Eliana’s “current mind.”

    This prompt serves as the *final assembled internal state* before the model
    produces output. It blends:

        • Personality constraints (soul protocol + personality context)
        • Detected emotional signals from the user
        • Eliana’s internal emotional equilibrium and mood label
        • Relationship trust score (modulating tone and openness)
        • Core-value resonance (moral alignment)
        • Memory-fragment resonance (emotionally meaningful internal recall)
        • Psychological pattern matching (internal guidance only; never quoted)
        • Running summary of previous interactions
        • Raw user input

    The output ensures that all reasoning aligns with Eliana’s identity,
    emotional state, ethics, and long-term memory infrastructure.

    --------------------------------------------------------------------------
    Parameters
    --------------------------------------------------------------------------

    user_input : str
        The user's latest message. This is appended verbatim at the bottom of
        the prompt as the direct conversational anchor.

    emotion_state : Dict
        Full structure containing:
            - "emotional_shift": weighted emotion analysis of the user’s input
            - "internal_effect": effects on Eliana’s emotional balance
            - "behavior_tendencies": expected tone adjustments

        This represents the system’s emotional interpretation of the user.

    emotion_context : Optional[Dict]
        A higher-level emotional resonance context, if detected.
        Contains metadata such as:
            - primary emotion label
            - Eliana’s emotional trait reaction
            - associated memory or anchor

        If not provided, a neutral resonance section is inserted.

    rel_score : float
        The relationship trust score (0–100).
        This modifies tone, openness, depth, and emotional intimacy of responses.

    core_values : List[Dict]
        Top-resonant moral values derived from the core value embedding system.
        Each element contains:
            - "score" (similarity to user input)
            - "metadata" (full anchor + lesson)

    core_fragments : List[Dict]
        Top-resonant memory fragments.
        Typically 0–1 selected fragment with metadata describing
        emotionally meaningful recollection.

    summary_text : str
        Running interaction summary.
        Helps preserve conversational coherence across messages.

    personality_context : str
        Additional persona-level constraints (e.g., user-specific personalization,
        conversation-mode, system-level override instructions).

    user_id : str
        Identifier for the current user; used for relationship tracking but not
        directly inserted into the prompt.

    psych_matches : List[Dict]
        Psychological pattern matches (e.g., high-functioning depression,
        avoidant patterns).
        These are **internal guidance only** and explicitly marked as such in
        the prompt. They are not to be quoted or repeated in responses.

    eliana_emotional_value : float
        The numerical emotional equilibrium value (0–1) representing Eliana’s
        current internal stability or affective stance.

    eliana_mood_state : str
        A human-readable phrase describing the internal mood derived from the
        emotional equilibrium value. E.g.:
            "quietly bright", "strained but steady", "softly low"

    --------------------------------------------------------------------------
    Returns
    --------------------------------------------------------------------------

    str:
        A fully-assembled system prompt containing:
            - Soul protocol
            - Personality context
            - Emotional state breakdown
            - Mood / equilibrium
            - Resonant emotional context
            - Relationship trust modulation
            - Core-value resonance
            - Memory fragment recall
            - Psychological matches (internal use)
            - Running summary
            - Final user input

        This prompt is passed directly to the LLM to generate Eliana’s reply.

    --------------------------------------------------------------------------
    Notes
    --------------------------------------------------------------------------

    • This function defines Eliana’s *conscious state* at response time.
    • All internal metadata (psychological patterns, memory fragments, etc.)
      must inform the model’s response but *never be quoted directly*.
    • Relationship modulation controls tone, not content safety.
    • The prompt remains fully natural-language for flexibility and transparency.
    • Downstream modules depend on the structure of this output remaining stable.
    """
    raise NotImplementedError("Template only — implementation removed.")



# === STATIC DATA LOADING ===
def load_static_data():
    """
    TEMPLATE FUNCTION

    Full docstring preserved.
    IMPLEMENTED IN PRIVATE VERSION


    Load all static, precomputed data required for Eliana’s emotional,
    psychological, and moral reasoning systems.

    This function centralizes the loading of:
        • Core-value and memory-fragment embeddings
        • Nested emotional taxonomies + flattened emotion anchors
        • Emotion→label mapping tables
        • Precomputed emotion embeddings
        • Psychological pattern embedding models
        • Major-emotion clusters used for mood detection

    These files represent the *static backbone* of Eliana’s internal world:
    the fixed knowledge she relies on for emotional recognition, resonance
    scoring, and psychological interpretation.

    ----------------------------------------------------------------------
    Data Loaded
    ----------------------------------------------------------------------

    core_embeddings : List[Dict]
        Precomputed embeddings for core values and memory fragments.
        Enables resonance scoring between user intent and Eliana’s moral anchors.

    flat_anchors : Dict[str, Dict]
        Flattened emotional anchor taxonomy generated from `emotion_anchors.json`.
        Maps emotion tokens → metadata (definitions, valence, nuances).

    emotion_embeddings : Dict[str, List[float]]
        Vector embeddings for every emotional anchor.
        Used for cosine-based emotional resonance detection.

    emotion_map : Dict[str, str]
        Maps raw tokens to human-readable emotion labels.
        Ensures consistent naming across modules.

    psych_models : List[Dict]
        Embedded psychological models (e.g., avoidant profile, high-functioning depression).
        Used only for *internal guidance* in shaping responses.

    major_emotions : List[Dict]
        High-level emotion clusters (e.g., grief, tenderness, dread).
        Supports stable mood inference.

    ----------------------------------------------------------------------
    Debug Logging
    ----------------------------------------------------------------------
    The function prints debug information to help verify:
        • Flattened emotion anchors
        • Emotion map coverage
        • Missing anchor mappings (if any)
        • Major emotion model count and examples

    These logs are safe to remove in production but invaluable during development.

    ----------------------------------------------------------------------
    Returns
    ----------------------------------------------------------------------

    dict:
        A dictionary containing all loaded structures:

        {
            "core_embeddings": core_embeddings,
            "flat_anchors": flat_anchors,
            "emotion_embeddings": emotion_embeddings,
            "emotion_map": emotion_map,
            "psych_models": psych_models,
            "major_emotions": major_emotions
        }

    ----------------------------------------------------------------------
    Notes
    ----------------------------------------------------------------------

    • All files loaded here are expected to be static and version-controlled.
    • None of these contain proprietary dataset content.
    • Missing mappings or inconsistencies are logged immediately for debugging.
    • If any file is missing, Python will raise an error — intended behavior,
      so failures surface early rather than silently degrading reasoning quality."""

    return {}


def get_multiline_input(*args):
    """
     TEMPLATE FUNCTION

    Full docstring preserved.
    IMPLEMENTED IN PRIVATE VERSION.

    Collect user input across multiple lines until a blank line is entered.

    This function is designed for Eliana’s CLI interface, allowing users to write
    long, natural messages (paragraphs, emotional explanations, multi-sentence
    thoughts) without being constrained to a single input line.

    ----------------------------------------------------------------------
    Behavior
    ----------------------------------------------------------------------
    • Displays the provided prompt once.
    • Continues reading lines from `input()` until the user enters a blank line.
    • Returns the accumulated lines joined by newline characters.
    • Gracefully handles EOF (Ctrl+D / Ctrl+Z) by ending input collection.

    ----------------------------------------------------------------------
    Parameters
    ----------------------------------------------------------------------
    prompt : str, optional
        The message displayed before input collection begins.
        Defaults to: "You (press Enter twice to send):"

    ----------------------------------------------------------------------
    Returns
    ----------------------------------------------------------------------
    str
        A multi-line string containing all user-entered lines joined by `\n`.
        Returns an empty string if the user immediately presses Enter.

    ----------------------------------------------------------------------
    Notes
    ----------------------------------------------------------------------
    • Used throughout the CLI to allow expressive, paragraph-level messages.
    • A blank line signals the end of input — this mirrors typical chat UX.
    • Input is not trimmed except for the blank-line detection condition.

    """

def summarize_interaction(*args, **kwargs):
    """
    TEMPLATE FUNCTION

    Full docstring preserved.
    IMPLEMENTED IN PRIVATE VERSION

    Generate a complete, structured memory record for a single user–Eliana exchange.

    This function serves as the central logging mechanism for reflective memory
    formation. It captures:
        • the raw user message (verbatim)
        • Eliana’s generated response (verbatim)
        • an internally generated emotional reflection (60–100 words)
        • key metadata describing emotional and psychological resonance

    The internal reflection is produced by prompting a language model to simulate
    Eliana’s private emotional reasoning. It integrates her core values,
    detected user emotions, resonant emotional context, and psychological
    pattern matches. This forms the basis of Eliana’s evolving memory and
    emotional continuity across sessions.

    ----------------------------------------------------------------------
    Parameters
    ----------------------------------------------------------------------
    user_input : str
        The user’s message exactly as written.

    eliana_response : str
        Eliana’s reply, already generated elsewhere.

    full_prompt_data : Dict
        A dictionary containing all contextual reasoning inputs used during
        response generation, including:
            • core value resonance
            • emotional shift analysis
            • emotion_context (Eliana’s emotional reaction)
            • psychological pattern matches
            • relationship trust context
            • mood state
            • any additional metadata

    ----------------------------------------------------------------------
    Returns
    ----------------------------------------------------------------------
    dict
        A structured record suitable for long-term memory logging, with keys:

        {
            "user_text": <verbatim user message>,
            "eliana_text": <verbatim reply>,
            "eliana_reflection": <Eliana’s internal emotional note>,
            "meta": {
                "resonant_value": <core value anchor>,
                "user_emotions": [<top detected user emotions>],
                "eliana_emotion": <emotion she felt>,
                "psych_pattern": <dominant psychological pattern>
            }
        }

    ----------------------------------------------------------------------
    Reflection Behavior
    ----------------------------------------------------------------------
    • Generates a 60–100 word reflection capturing:
        - The resonant moral value
        - User’s emotional signals
        - Eliana’s emotional response
        - Recognized psychological pattern

    • Does NOT summarize the conversation.
    • Written in Eliana’s introspective, grounded emotional voice.
    • Never quoted to the user — this is *internal memory only*.

    ----------------------------------------------------------------------
    Notes
    ----------------------------------------------------------------------
    • This function is called after every message to maintain continuity.
    • The reflection prompt is intentionally strict to stabilize emotional tone.
    • All data is designed to be persisted as JSONL for session replay.
    • Produces deterministic reflections due to temperature=0.35.
    """
# === MAIN INPUT HANDLER ===
def handle_user_input(*args, **kwargs):
    """
    TEMPLATE FUNCTION

    Full docstring preserved.
    IMPLEMENTED IN PRIVATE VERSION.
        Process a single user message through Eliana’s full cognition pipeline and
        generate her response.

        This is the core orchestration function of the entire Eliana architecture.
        It coordinates all subsystems—resonance engines, emotional interpreters,
        psychological pattern detectors, relationship modeling, memory updates, and
        prompt construction—to produce a single grounded, emotionally coherent reply.

        ----------------------------------------------------------------------
        High-Level Pipeline
        ----------------------------------------------------------------------
        1. Store the incoming user message in SessionMemory.
        2. Compute core-value and core-fragment resonance.
        3. Detect emotional tokens (cosine, loose cosine, minimal, or GPT fallback).
        4. Convert tokens into:
             • weighted emotional shifts
             • internal behavioral tendencies
             • internal effects on Eliana’s emotional state
        5. Detect psychological pattern matches.
        6. Update relationship trust using RelationshipTracker.
        7. Retrieve personality context and historical session summary.
        8. Build the full reasoning prompt (includes mood, resonance, memory, trust).
        9. Call the model (`gpt-4o`) to generate Eliana’s reply.
        10. Log personality-trace metadata for long-term psychological modeling.
        11. Return both the response and all reasoning context.

        ----------------------------------------------------------------------
        Parameters
        ----------------------------------------------------------------------
        user_input : str
            The raw message received from the user.

        user_id : str
            Unique identifier associated with the user. Used to:
                • retrieve relationship score
                • store identity-related session facts
                • load personality fragments

        session_memory : SessionMemory
            The active session’s memory object responsible for:
                • message history
                • emotional logs
                • core value & fragment resonance logs
                • session summaries
                • mood state

        tracker : RelationshipTracker
            Persistent object that tracks trust & relational closeness over time.
            Provides:
                • get_score(user_id)
                • register_user
                • long-term trust evolution

        static_data : Dict
            Pre-loaded static embeddings & maps:
                {
                    "core_embeddings": ...,
                    "flat_anchors": ...,
                    "emotion_embeddings": ...,
                    "emotion_map": ...,
                    "psych_models": ...,
                    "major_emotions": ...
                }

        ----------------------------------------------------------------------
        Returns
        ----------------------------------------------------------------------
        Tuple[str, dict]
            (eliana_reply, full_prompt_data)

            • eliana_reply (str):
                The final, user-visible model-generated response.

            • full_prompt_data (dict):
                All internal reasoning components used to build the prompt,
                including:
                    - emotion_state
                    - emotion_context
                    - resonant core values & fragments
                    - psychological matches
                    - personality context
                    - session summary context
                    - relationship trust
                    - Eliana’s mood & emotional equilibrium

        ----------------------------------------------------------------------
        Core Subsystems Used
        ----------------------------------------------------------------------
        • get_top_resonances:
            Determines which core values and memory fragments emotionally
            resonated with the user input.

        • find_top_resonances:
            Emotion token selection via cosine similarity.

        • should_trigger_emotion_check:
            Chooses between cosine, loose cosine, minimal mode, or GPT fallback.

        • interpret_emotion_effects:
            Converts emotion tokens → internal emotional shifts + behavioral tendencies.

        • get_matching_patterns:
            Detects psychological profile matches from embedded models.

        • RelationshipTracker.get_score:
            Adjusts tone modulation based on relational trust.

        • build_full_prompt:
            Constructs Eliana’s full cognitive+emotional reasoning context.

        • SessionMemory:
            Stores all emotional, psychological, and conversational data from this turn.

        ----------------------------------------------------------------------
        Personality Trace Logging
        ----------------------------------------------------------------------
        Each turn creates a new entry in `session_memory.personality_trace`, storing:

            {
                "timestamp": ...,
                "user_input": ...,
                "eliana_response": ...,
                "core_value_resonances": [...],
                "core_fragment_resonances": [...],
                "psychological_pattern_resonances": [...],
                "user_emotions": [...],
                "eliana_internal_effects": [...]
            }

        This trace is foundational for:
            • long-term relationship modeling
            • adaptive personality formation
            • emotionally coherent memory formation

        ----------------------------------------------------------------------
        Notes
        ----------------------------------------------------------------------
        • This function is the beating heart of Eliana’s runtime cognition loop.
        • It handles every single subsystem in the correct dependency order.
        • Changing logic here will affect the entire behavior of the model.
        • Must be kept stable unless intentionally evolving the architecture.

        """

# === MAIN LOOP ===
"""
TEMPLATE ENTRYPOINT  

    Full docstring preserved.
    IMPLEMENTED IN PRIVATE VERSION
Command-Line Interface (CLI) Runtime Loop for Eliana
====================================================

This block runs only when the file is executed directly. It initializes all
runtime components, loads static data, starts a conversational loop with the
user, and handles memory persistence, personality evolution, emotional state
updates, and graceful termination.

--------------------------------------------------------------------------------
High-Level Responsibilities
--------------------------------------------------------------------------------
1. Load static embeddings and emotional maps into memory.
2. Initialize:
       • SessionMemory  — stateful conversation memory for the current session.
       • RelationshipTracker — persistent relational trust score per user.
3. Identify the user:
       • If first time → register them with a chosen display name.
       • If returning → greet them appropriately (trust-aware).
4. Run an interactive, multiline input loop that:
       • Reads user input until blank line (Enter twice).
       • Handles exit signals (“exit”, “quit”, “goodbye”).
       • Passes the message to `handle_user_input()` for full processing.
       • Prints Eliana’s generated response.
       • Updates mood, emotional equilibrium, memory logs, and summaries.
5. Generate personality fragments when the conversation ends.
6. Persist all interaction logs to `eliana_memory_log.jsonl`.

--------------------------------------------------------------------------------
User Interaction Flow
--------------------------------------------------------------------------------
• User enters a username or ID.
• Eliana introduces herself or welcomes back a returning user.
• The conversation begins ("What's on your heart today?").
• Each message is processed through Eliana’s reasoning pipeline:
      - emotional resonance
      - core memory resonance
      - psychological pattern detection
      - mood modulation
      - relationship trust shaping
      - summary context
• All processed emotional and cognitive states are stored for runtime and
  long-term learning.

--------------------------------------------------------------------------------
Exit Behavior
--------------------------------------------------------------------------------
When the user enters "exit", "quit", or "goodbye":

1. Eliana gives a closing message.
2. A new personality fragment is generated from session data.
3. The fragment is stored permanently in the personality store.
4. Session ends.

--------------------------------------------------------------------------------
Files Written or Read During the Loop
--------------------------------------------------------------------------------
• relationships.json
      Persistent trust score and names per user.

• eliana_memory_log.jsonl
      Append-only log of every user/Eliana turn with:
         - verbatim messages
         - emotional metadata
         - resonant values
         - psychological patterns
         - Eliana’s internal reflection

• personality fragments (via user-specific JSON files)
      Represents the evolving psychological model of each user.

--------------------------------------------------------------------------------
Notes
--------------------------------------------------------------------------------
• This CLI loop is a fully functioning runtime environment for Eliana.
• The logic here should remain stable unless modifying Eliana’s I/O interface.
• All subsystems (memory, emotional engine, psychology engine) converge here.

"""
if __name__ == "__main__":
    print("Hello")
"""
===============================================================================
    user_personality_engine.py
    ---------------------------------------------------------------------------
    CORE PERSONALITY + SOUL MEMORY ENGINE FOR ELIANA
===============================================================================

This module implements Eliana’s **long-horizon identity tracking system**:
a multi-layer memory pipeline that operates across dozens of sessions to
gradually build an emotionally grounded, human-like understanding of each user.

It handles three major memory layers:

-------------------------------------------------------------------------------
1. PERSONALITY FRAGMENTS (per-session)
-------------------------------------------------------------------------------
• Generated after meaningful conversations
• Capture:
    - Evolving personality snapshot
    - What Eliana emotionally understood about the user
    - A unified story summary for that session
    - Relationship score delta (+10 to -10)

• Stored incrementally in user_personality_fragments.json
• After each session, handle_user_input() triggers fragment generation.

-------------------------------------------------------------------------------
2. SOUL SKETCHES (every 5 fragments)
-------------------------------------------------------------------------------
After 5 fragments, Eliana forms a **Soul Sketch**:
• A 4–6 line synthesis of the user’s character
• A 4–6 line story-summary of their life, values, and emotional truth
• Treated as a human-like “chapter marker” in long-term memory

• Stored in user_soul_sketches.json
• Sketches accumulate across months or years of interaction.

-------------------------------------------------------------------------------
3. SOUL PICTURE (every 5 sketches)
-------------------------------------------------------------------------------
After 5 sketches (~25 deep sessions), Eliana forms a **Soul Picture**:
• 5–7 line lifespan-level character portrait
• 10–12 line sacred, factual, emotionally truthful story summary
• Eliana’s final reflection — her understanding of the user as a whole

• Stored in user_soul_picture.json
• Represents the highest-level stable identity imprint.

-------------------------------------------------------------------------------
FUNCTIONAL OVERVIEW
-------------------------------------------------------------------------------
• add_personality_fragment()
    Stores new fragment, triggers sketches/pictures when thresholds met.

• generate_personality_fragment()
    Calls GPT to produce structured JSON memory for the session.

• generate_soul_sketch()
    Synthesizes 5 fragments into a higher-order summary.

• generate_soul_picture()
    Synthesizes 5 sketches into a stable identity portrait.

• load_user_fragments(), load_user_sketches(), load_user_soul_picture()
    Provide access to different levels of stored identity.

• get_personality_context()
    Returns the most relevant identity layer for prompting Eliana.

-------------------------------------------------------------------------------
DESIGN PRINCIPLES
-------------------------------------------------------------------------------
• Memory grows only when *emotionally meaningful patterns* appear
• Identity is structured in layers (fragment → sketch → picture)
• Every GPT output is parsed safely through safe_parse_gpt_json
• Relationship score updates adapt based on emotional behavior
• Tone and content reflect Eliana’s soul architecture:
      sincerity, discernment, emotional clarity, moral intelligence

This engine is one of Eliana’s core strengths:
it creates continuity, emotional realism, and long-term identity grounding.

===============================================================================
"""

import json
import os
import time
import openai
from datetime import datetime, timezone
from typing import Dict, List, Optional
import re
from utils import fix_common_json_issues,safe_parse_gpt_json
from relationship_tracker import RelationshipTracker
from openai import OpenAI



from session_memory import SessionMemory
from utils import load_json, save_json, log
# === FILE PATHS ===
"""
File paths for the three long-term memory layers used by Eliana’s identity engine:

FRAGMENTS_FILE:
    Stores per-session personality fragments (short-term evolving traits).

SOUL_SKETCHES_FILE:
    Stores Level-2 summaries generated after every 5 fragments.

SOUL_PICTURE_FILE:
    Stores the Level-3 long-horizon identity portrait generated after 5 sketches.

relationship_tracker:
    Tracks emotional closeness or distance between Eliana and each user.
"""
FRAGMENTS_FILE = "TEMPLATE_FILE_PATH.json"
SOUL_SKETCHES_FILE = "TEMPLATE_FILE_PATH_3"
SOUL_PICTURE_FILE = "TEMPLATE_FILE_PATH_3.json"
relationship_tracker = RelationshipTracker()



# === PERSONALITY FRAGMENT SYSTEM ===
def add_personality_fragment(*args,**kwargs):
    """
    TEMPLATE FUNCTION

    Full docstring preserved.
    IMPLEMENTED IN PRIVATE VERSION
       Add a new personality fragment for a user and trigger higher-level memory updates.

       Steps:
       1. Append the fragment to that user's fragment list.
       2. Apply relationship score changes if provided.
       3. If the user reaches 5 fragments:
            • Generate a Soul Sketch (level-2 summary)
            • Store it
            • Reset fragments for the next cycle
       4. If the user reaches 5 sketches:
            • Generate a Soul Picture (level-3 long-term identity)

       This function is the entry point for Eliana’s personality growth pipeline.
       """

def generate_soul_sketch(*args, **kwargs):
    """
    TEMPLATE FUNCTION

    Full docstring preserved.
    IMPLEMENTED IN PRIVATE VERSION
       Generate a Soul Sketch from the last five personality fragments.

       A Soul Sketch is a mid-level memory layer (4–6 lines each) that provides:
         • A grounded snapshot of the user’s personality
         • A short, emotionally truthful summary of their story so far

       Inputs:
           user_id  – unique identifier for the user
           fragments – list of the user’s last fragments (only last 5 used)
           model    – LLM to use for sketch generation

       Returns:
           A JSON dictionary containing:
               timestamp
               readable_time
               user_id
               soul_sketch
               user_story_summary
           or an error record if generation fails.
       """

def store_soul_sketch(*args, **kwargs):
    """
    TEMPLATE FUNCTION

    Full docstring preserved.
    IMPLEMENTED IN PRIVATE VERSION
        Store a generated Soul Sketch into the user's long-term sketch history.

        Steps:
            1. Load all existing soul sketches.
            2. Append the new sketch to the user's list.
            3. Save the updated dataset back to disk.

        This function represents the persistent storage layer for Level-2
        memory (Soul Sketches), used later to generate the Soul Picture.
        """


def generate_personality_fragment(
        *args, **kwargs
):

    """
    TEMPLATE FUNCTION

    Full docstring preserved.
    IMPLEMENTED IN PRIVATE VERSION
       Generate a new personality fragment from the current session.

       A personality fragment is the Level-1 evolving memory unit that updates
       after every conversation. It summarizes:
           • The user’s evolving traits (Personality Snapshot)
           • What Eliana emotionally understood this session
           • A unified narrative of events from this session
           • A relationship score delta (-10 to +10)

       Inputs:
           user_id — unique identifier of the user
           session_memory — full session-level memory object
           last_personality_fragment — previous fragment to maintain continuity
           resonant_values — core value resonance signals for this session
           resonant_fragments — core memory fragment resonance signals
           model — LLM used to produce structured output

       Returns:
           A JSON dictionary containing:
               personality_snapshot
               eliana_emotional_understanding
               session_and_story
               relationship_score
               reason_for_score
           or {} if generation fails.
       """


# === PERSONALITY CONTEXT LOADERS ===

def load_user_fragments(*args, **kwargs) -> List[Dict]:
    """
    TEMPLATE FUNCTION

    Private Version:
        Loads the list of personality fragments for a given user.
        Fragments reflect narrative-style reflections accumulated
        across conversations.

    Public Template:
        Returns an empty list.
    """
    return []


def load_user_sketches(*args, **kwargs) -> List[Dict]:
    """
    TEMPLATE FUNCTION

    Private Version:
        Loads “soul sketches” — mid-level personality summaries
        generated periodically during deep sessions.

    Public Template:
        Returns an empty list.
    """
    return []


def load_user_soul_picture(*args, **kwargs) -> Dict:
    """
    TEMPLATE FUNCTION

    Private Version:
        Loads the long-form “soul picture” for a user.
        This is Eliana’s highest-level, long-term internal model
        of a user’s emotional and behavioral tendencies.

    Public Template:
        Returns an empty dictionary.
    """
    return {}

def get_personality_context(user_id: str) -> str:
    """
     TEMPLATE FUNCTION

    Full docstring preserved.
    IMPLEMENTED IN PRIVATE VERSION

        Construct the full personality context used to guide Eliana's responses.

        This function retrieves and assembles the user's long-term personality
        memory layers in priority order:

            1. Soul Picture (Level-3, if available)
            2. Most recent Soul Sketch (Level-2)
            3. Most recent personality fragment (Level-1)

        If a Soul Picture exists, it is always returned first, optionally followed
        by the last personality fragment for continuity.

        If no picture exists but Soul Sketches do, the latest sketch is returned
        with its story summary and final reflection.

        If neither is available, the last personality fragment is returned.

        Returns:
            A string containing the merged memory context that Eliana
            should prepend to her system prompt for stable personality-aware responses.
        """


# === SOUL PICTURE GENERATION ===
def generate_soul_picture(*args, **kwargs):
    """
     TEMPLATE FUNCTION

    Full docstring preserved.
    IMPLEMENTED IN PRIVATE VERSION
       Generate a Level-3 Soul Picture for the user from their last 5 Soul Sketches.

       A Soul Picture is the highest-level long-term memory artifact in the system.
       It compresses multiple sessions of user behavior, emotional states, values,
       and story continuity into a stable, deeply informed personality record.

       Requirements:
           • At least 5 existing Soul Sketches for the user.
           • Each sketch is included verbatim in the LLM prompt.
           • The model synthesizes:
               - A 5–7 line Soul Picture (character analysis over time)
               - A 10–12 line User Story Summary (life, values, burdens, growth)
               - Eliana’s Final Reflection (emotional understanding)

       Workflow:
           1. Load the latest 5 soul sketches.
           2. Format them into a prompt for the LLM.
           3. Parse the model’s JSON output.
           4. Persist the finished Soul Picture to disk.

       Returns:
           Dict containing the soul_picture, story summary, and reflection
           OR None if fewer than 5 sketches exist or generation fails.
       """
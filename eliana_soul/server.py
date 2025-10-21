from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime, timezone
import traceback

# === Import Eliana Core ===
from eliana_soul.Eliana_brain import (
    handle_user_input,
    handle_light_user_input,
    triage_user_input,
    summarize_interaction,   # ✅ make sure this is imported
    load_static_data,
    soul_protocol
)
from eliana_soul.session_memory import SessionMemory
from eliana_soul.relationship_tracker import RelationshipTracker


# === Initialize FastAPI app ===
app = FastAPI(title="Eliana API", description="Soul-shaped conversational AI API", version="1.0")

# === Allow frontend access ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during development; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load static data once ===
print("[⚙️] Loading static data...")
STATIC_DATA = load_static_data()
print("[✅] Static data loaded.")

# === Initialize persistent memory and tracker ===
SESSION = SessionMemory(system_prompt=soul_protocol)
TRACKER = RelationshipTracker(path="relationships.json")


# === Request schema ===
class ChatRequest(BaseModel):
    user_id: str
    message: str


# === Root endpoint ===
@app.get("/")
def home():
    return {"message": "Eliana API is running.", "timestamp": datetime.now(timezone.utc).isoformat()}


# === Chat endpoint ===
@app.post("/api/chat")
async def chat(req: ChatRequest) -> Dict[str, Any]:
    """
    Handles incoming user messages and returns Eliana's response
    along with rich debug information for UI visualization.
    """
    try:
        user_id = req.user_id.strip()
        user_message = req.message.strip()
        timestamp = datetime.now(timezone.utc).isoformat()

        # === Ensure relationship record exists ===
        if not TRACKER.get_user_relationship(user_id):
            TRACKER.register_user(user_id, user_id)
            print(f"[👤] Registered new user: {user_id}")

        # === Load relationship score and route type ===
        rel_score = TRACKER.get_score(user_id)
        route = triage_user_input(current_input=user_message)
        print(f"[🗣️] Message from {user_id} ({route} mode): {user_message}")

        # === Choose pipeline ===
        if route == "light":
            reply = handle_light_user_input(
                user_input=user_message,
                user_id=user_id,
                session_memory=SESSION,
                tracker=TRACKER,
                static_data=STATIC_DATA,
                rel_score=rel_score,
                last_personality_fragment=None
            )
        else:
            reply = handle_user_input(
                user_input=user_message,
                user_id=user_id,
                session_memory=SESSION,
                tracker=TRACKER,
                static_data=STATIC_DATA
            )

        # === 🧠 Generate and append summary ===
        try:
            summary = summarize_interaction(user_message, reply)
            SESSION.session_summary.append(summary)
            print(f"[🧩] Added session summary ({len(SESSION.session_summary)} total):\n{summary}\n")
        except Exception as summary_error:
            print("[⚠️] Failed to generate session summary:", summary_error)

        # === Build debug block ===
        debug_info = {
            "timestamp": timestamp,
            "route": route,
            "relationship_score": rel_score,
            "recent_eliana_emotions": SESSION.recent_eliana_emotions[-1:] if SESSION.recent_eliana_emotions else [],
            "core_value_resonances": SESSION.core_value_resonance[-3:],
            "core_fragment_resonances": SESSION.core_fragment_resonance[-3:],
            "psych_patterns": SESSION.psychological_patterns[-3:],
            "session_summary_count": len(SESSION.session_summary),
            "session_summary_tail": SESSION.session_summary[-3:],  # 👈 shows last few summaries
        }

        print(f"[💬] Reply: {reply[:180]}...")

        return {"reply": reply, "debug": debug_info}

    except Exception as e:
        print("[❌] Error in /api/chat:", e)
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


# === Health check ===
@app.get("/health")
def health_check():
    return {"status": "ok", "active_users": len(TRACKER.data.keys())}


# === Run directly (local) ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("eliana_soul.server:app", host="127.0.0.1", port=8000, reload=True)

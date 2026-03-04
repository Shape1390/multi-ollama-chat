import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import requests
import streamlit as st


# ----------------------------
# Helpers: Ollama API
# ----------------------------

def http_get_json(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def http_post_json(url: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def ollama_version(base_url: str) -> str:
    data = http_get_json(f"{base_url}/api/version", timeout=5.0)
    return data.get("version", "unknown")

def ollama_list_models(base_url: str) -> List[str]:
    # Ollama: POST /api/tags (historical) vs GET /api/tags (newer) can vary.
    # We'll try GET first, then POST fallback.
    try:
        data = http_get_json(f"{base_url}/api/tags", timeout=8.0)
    except Exception:
        data = http_post_json(f"{base_url}/api/tags", payload={}, timeout=8.0)

    models = []
    for m in data.get("models", []):
        name = m.get("name")
        if name:
            models.append(name)
    return sorted(models)

def ollama_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    stream: bool = False,
) -> str:
    """
    Calls Ollama /api/chat.
    We keep stream=False to keep this simple and stable.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = int(seed)

    data = http_post_json(f"{base_url}/api/chat", payload=payload, timeout=180.0)

    # Expected: {"message":{"role":"assistant","content":"..."},"done":true,...}
    msg = data.get("message", {})
    return msg.get("content", "").strip()


# ----------------------------
# App State
# ----------------------------

@dataclass
class Agent:
    name: str
    model: str
    system_prompt: str


DEFAULT_SYSTEM = (
    "You are a helpful assistant. "
    "Be direct, cite uncertainty, and avoid making things up. "
    "If you disagree with another model, explain why calmly."
)

DEFAULT_DEBATE_INSTRUCTION = (
    "You are one voice in a multi-model panel. "
    "Respond to the user's latest message. "
    "If another agent already replied, you may reference their points and add corrections or alternatives. "
    "Keep it concise and actionable."
)

def init_state():
    if "base_url" not in st.session_state:
        st.session_state.base_url = "http://localhost:11434"

    if "models_cache" not in st.session_state:
        st.session_state.models_cache = []

    if "chat" not in st.session_state:
        # Shared conversation thread (user + all assistants)
        st.session_state.chat: List[Dict[str, str]] = []

    if "agents" not in st.session_state:
        # Default: 3 agents (you can change to Llama models you actually have)
        st.session_state.agents: List[Agent] = [
            Agent(name="Llama-Reasoner", model="llama3.2:latest", system_prompt=DEFAULT_SYSTEM),
            Agent(name="Llama-Concise", model="llama3.1:latest", system_prompt=DEFAULT_SYSTEM),
            Agent(name="Llama-DevilAdvocate", model="llama3:latest", system_prompt=DEFAULT_SYSTEM),
        ]

    if "settings" not in st.session_state:
        st.session_state.settings = {
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": None,
            "mode": "Round (everyone replies once)",
        }


def add_message(role: str, content: str):
    st.session_state.chat.append({"role": role, "content": content})


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="Multi-Model Ollama Chat", page_icon="🧠", layout="wide")
init_state()

st.title("🧠 Multi-Model Chat (Local Ollama + Streamlit)")
st.caption("A simple panel where multiple local models reply to the same conversation.")

# Sidebar: Connection + Model List
with st.sidebar:
    st.header("Connection")
    st.session_state.base_url = st.text_input(
        "Ollama Base URL",
        value=st.session_state.base_url,
        help="Usually http://localhost:11434 (same machine). Use a Tailscale/LAN IP if needed.",
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("Test Connection"):
            try:
                v = ollama_version(st.session_state.base_url)
                st.success(f"Connected ✅ (Ollama {v})")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    with colB:
        if st.button("Refresh Models"):
            try:
                st.session_state.models_cache = ollama_list_models(st.session_state.base_url)
                st.success(f"Loaded {len(st.session_state.models_cache)} models")
            except Exception as e:
                st.error(f"Could not load models: {e}")

    st.divider()

    st.header("Generation Settings")
    st.session_state.settings["mode"] = st.selectbox(
        "Mode",
        options=[
            "Round (everyone replies once)",
            "Debate (sequential, reference prior replies)",
        ],
        index=0 if st.session_state.settings["mode"].startswith("Round") else 1,
    )

    st.session_state.settings["temperature"] = st.slider("Temperature", 0.0, 2.0, float(st.session_state.settings["temperature"]), 0.05)
    st.session_state.settings["top_p"] = st.slider("Top-p", 0.0, 1.0, float(st.session_state.settings["top_p"]), 0.05)

    seed_text = st.text_input("Seed (optional)", value="" if st.session_state.settings["seed"] is None else str(st.session_state.settings["seed"]))
    seed_val = None
    if seed_text.strip():
        try:
            seed_val = int(seed_text.strip())
        except ValueError:
            st.warning("Seed must be an integer. Ignoring.")
            seed_val = None
    st.session_state.settings["seed"] = seed_val

    st.divider()

    st.header("Agents")
    st.caption("Each agent uses one Ollama model name exactly as shown in `ollama list` / /api/tags.")
    models_for_dropdown = st.session_state.models_cache[:] if st.session_state.models_cache else []

    # Render editable agent list
    new_agents: List[Agent] = []
    for i, agent in enumerate(st.session_state.agents):
        with st.expander(f"Agent {i+1}: {agent.name}", expanded=(i == 0)):
            name = st.text_input(f"Name #{i}", value=agent.name, key=f"agent_name_{i}")
            if models_for_dropdown:
                # If we have model list, use selectbox
                model = st.selectbox(
                    f"Model #{i}",
                    options=models_for_dropdown,
                    index=models_for_dropdown.index(agent.model) if agent.model in models_for_dropdown else 0,
                    key=f"agent_model_{i}",
                )
            else:
                model = st.text_input(f"Model #{i}", value=agent.model, key=f"agent_model_{i}")

            system_prompt = st.text_area(
                f"System Prompt #{i}",
                value=agent.system_prompt,
                height=120,
                key=f"agent_sys_{i}",
            )

            new_agents.append(Agent(name=name.strip() or f"Agent{i+1}", model=model.strip(), system_prompt=system_prompt.strip()))

    st.session_state.agents = new_agents

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Add Agent"):
            st.session_state.agents.append(
                Agent(
                    name=f"Agent{len(st.session_state.agents)+1}",
                    model=models_for_dropdown[0] if models_for_dropdown else "llama3.2:latest",
                    system_prompt=DEFAULT_SYSTEM,
                )
            )
            st.rerun()
    with col2:
        if st.button("Remove Last") and len(st.session_state.agents) > 1:
            st.session_state.agents.pop()
            st.rerun()
    with col3:
        if st.button("Clear Chat"):
            st.session_state.chat = []
            st.rerun()


# Main layout: Chat + Controls
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Conversation")

    # Display chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_text = st.chat_input("Type a message…")
    if user_text:
        add_message("user", user_text)
        st.rerun()

with right:
    st.subheader("Panel Controls")

    st.write("**How it works:**")
    st.write("- You type one message.")
    st.write("- Click **Run Panel** and each agent replies.")
    st.write("- In **Debate** mode, later agents can reference earlier replies.")

    st.divider()

    if st.button("Run Panel", type="primary", use_container_width=True):
        # Guard: if last message isn't user, we still allow panel run
        if not st.session_state.chat:
            st.warning("No messages yet. Type something first.")
        else:
            base_url = st.session_state.base_url
            temp = float(st.session_state.settings["temperature"])
            top_p = float(st.session_state.settings["top_p"])
            seed = st.session_state.settings["seed"]

            mode = st.session_state.settings["mode"]

            # Build a "conversation slice" to send to each model.
            # We'll keep the entire thread, but prepend system prompt per agent.
            # Note: Ollama /api/chat expects messages [{role, content}, ...]
            # We also add a lightweight per-agent instruction for panel behavior.
            panel_errors = 0

            if mode.startswith("Round"):
                for agent in st.session_state.agents:
                    try:
                        messages = [{"role": "system", "content": agent.system_prompt}]
                        messages.append({"role": "system", "content": DEFAULT_DEBATE_INSTRUCTION})
                        messages.extend(st.session_state.chat)

                        reply = ollama_chat(
                            base_url=base_url,
                            model=agent.model,
                            messages=messages,
                            temperature=temp,
                            top_p=top_p,
                            seed=seed,
                            stream=False,
                        )

                        if not reply:
                            reply = "_(no response)_"

                        add_message("assistant", f"**{agent.name}** (`{agent.model}`)\n\n{reply}")
                        time.sleep(0.05)
                        st.rerun()
                    except Exception as e:
                        panel_errors += 1
                        add_message("assistant", f"**{agent.name}** (`{agent.model}`)\n\n⚠️ Error: {e}")
                        st.rerun()

            else:
                # Debate mode: each agent sees prior agent replies that were added this run
                for agent in st.session_state.agents:
                    try:
                        messages = [{"role": "system", "content": agent.system_prompt}]
                        messages.append({"role": "system", "content": DEFAULT_DEBATE_INSTRUCTION})
                        messages.extend(st.session_state.chat)

                        reply = ollama_chat(
                            base_url=base_url,
                            model=agent.model,
                            messages=messages,
                            temperature=temp,
                            top_p=top_p,
                            seed=seed,
                            stream=False,
                        )

                        if not reply:
                            reply = "_(no response)_"

                        add_message("assistant", f"**{agent.name}** (`{agent.model}`)\n\n{reply}")
                        time.sleep(0.05)
                        st.rerun()
                    except Exception as e:
                        panel_errors += 1
                        add_message("assistant", f"**{agent.name}** (`{agent.model}`)\n\n⚠️ Error: {e}")
                        st.rerun()

            if panel_errors:
                st.warning(f"Panel finished with {panel_errors} error(s). Check model names and Ollama URL.")

    st.divider()

    st.write("**Tips**")
    st.write("- If models don’t show up, click **Refresh Models** in the sidebar.")
    st.write("- Make sure Ollama is running and reachable at your Base URL.")
    st.write("- Model names must match what Ollama reports (e.g., `llama3.2:latest`).")

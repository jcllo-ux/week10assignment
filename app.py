from datetime import datetime
import json
from pathlib import Path
import time
from uuid import uuid4

import requests
import streamlit as st


API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
CHATS_DIR = Path("chats")
MEMORY_PATH = Path("memory.json")


st.set_page_config(page_title="My AI Chat", layout="wide")
st.title("My AI Chat")


def load_hf_token() -> str | None:
    """Return the Hugging Face token if it exists and is non-empty."""
    token = st.secrets.get("HF_TOKEN", "")
    token = token.strip() if isinstance(token, str) else ""
    return token or None


def load_memory() -> dict[str, str]:
    if not MEMORY_PATH.exists():
        return {}

    try:
        memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(memory, dict):
        return {}

    cleaned_memory: dict[str, str] = {}
    for key, value in memory.items():
        if isinstance(key, str) and isinstance(value, str) and value.strip():
            cleaned_memory[key] = value.strip()

    return cleaned_memory


def save_memory(memory: dict[str, str]) -> None:
    MEMORY_PATH.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def initialize_memory_state() -> None:
    if "memory" not in st.session_state:
        st.session_state.memory = load_memory()
        if not MEMORY_PATH.exists() or MEMORY_PATH.read_text(encoding="utf-8").strip() == "":
            save_memory(st.session_state.memory)


def clear_memory() -> None:
    st.session_state.memory = {}
    MEMORY_PATH.write_text("{}", encoding="utf-8")


def make_timestamp() -> str:
    return datetime.now().strftime("%b %d, %Y %I:%M %p")


def create_chat() -> dict[str, object]:
    return {
        "id": str(uuid4()),
        "title": "New Chat",
        "timestamp": make_timestamp(),
        "messages": [],
    }


def build_messages_with_memory(
    messages: list[dict[str, str]], memory: dict[str, str]
) -> list[dict[str, str]]:
    if not memory:
        return messages

    memory_lines = [f"- {key}: {value}" for key, value in memory.items()]
    memory_prompt = (
        "Use the following saved user memory to personalize responses when it is relevant.\n"
        "Do not mention this memory unless it helps answer the user.\n"
        "Saved memory:\n"
        + "\n".join(memory_lines)
    )

    return [{"role": "system", "content": memory_prompt}, *messages]


def get_chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def save_chat(chat: dict[str, object]) -> None:
    CHATS_DIR.mkdir(exist_ok=True)
    get_chat_path(str(chat["id"])).write_text(json.dumps(chat, indent=2), encoding="utf-8")


def load_saved_chats() -> list[dict[str, object]]:
    CHATS_DIR.mkdir(exist_ok=True)
    chats = []

    for chat_file in sorted(CHATS_DIR.glob("*.json")):
        try:
            chat = json.loads(chat_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(chat, dict):
            continue

        chat_id = str(chat.get("id", "")).strip()
        title = str(chat.get("title", "New Chat")).strip() or "New Chat"
        timestamp = str(chat.get("timestamp", make_timestamp())).strip() or make_timestamp()
        messages = chat.get("messages", [])

        if not chat_id or not isinstance(messages, list):
            continue

        chats.append(
            {
                "id": chat_id,
                "title": title,
                "timestamp": timestamp,
                "messages": messages,
            }
        )

    return chats


def initialize_chat_state() -> None:
    if "chats" not in st.session_state:
        saved_chats = load_saved_chats()
        if saved_chats:
            st.session_state.chats = saved_chats
            st.session_state.active_chat_id = saved_chats[0]["id"]
        else:
            first_chat = create_chat()
            st.session_state.chats = [first_chat]
            st.session_state.active_chat_id = first_chat["id"]
            save_chat(first_chat)


def get_active_chat() -> dict[str, object] | None:
    active_chat_id = st.session_state.get("active_chat_id")
    for chat in st.session_state.get("chats", []):
        if chat["id"] == active_chat_id:
            return chat
    return None


def set_active_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id


def add_new_chat() -> None:
    new_chat = create_chat()
    st.session_state.chats.append(new_chat)
    st.session_state.active_chat_id = new_chat["id"]
    save_chat(new_chat)


def delete_chat(chat_id: str) -> None:
    chats = st.session_state.chats
    chat_index = next((index for index, chat in enumerate(chats) if chat["id"] == chat_id), None)

    if chat_index is None:
        return

    was_active = st.session_state.active_chat_id == chat_id
    chats.pop(chat_index)
    get_chat_path(chat_id).unlink(missing_ok=True)

    if not chats:
        st.session_state.active_chat_id = None
    elif was_active:
        next_index = min(chat_index, len(chats) - 1)
        st.session_state.active_chat_id = chats[next_index]["id"]


def maybe_update_chat_title(chat: dict[str, object], user_prompt: str) -> None:
    if chat["title"] == "New Chat":
        short_title = user_prompt.strip()[:30]
        chat["title"] = short_title or "New Chat"


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Chats")
        if st.button("New Chat", use_container_width=True):
            add_new_chat()
            st.rerun()

        chat_list = st.container(height=500)
        with chat_list:
            for chat in st.session_state.chats:
                is_active = chat["id"] == st.session_state.active_chat_id
                chat_col, delete_col = st.columns([5, 1])

                with chat_col:
                    if st.button(
                        str(chat["title"]),
                        key=f"chat_select_{chat['id']}",
                        type="primary" if is_active else "secondary",
                        use_container_width=True,
                    ):
                        set_active_chat(str(chat["id"]))
                        st.rerun()
                    st.caption(str(chat["timestamp"]))

                with delete_col:
                    if st.button("✕", key=f"chat_delete_{chat['id']}", use_container_width=True):
                        delete_chat(str(chat["id"]))
                        st.rerun()

        with st.expander("User Memory", expanded=True):
            if st.session_state.memory:
                for key, value in st.session_state.memory.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.caption("No saved memory yet.")

            if st.button("Clear Memory", use_container_width=True):
                clear_memory()
                st.rerun()


def request_chat_completion(
    hf_token: str, messages: list[dict[str, str]], *, stream: bool
) -> requests.Response:
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 512,
        "stream": stream,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=30, stream=stream)

    if response.status_code == 401:
        raise ValueError("Your Hugging Face token looks invalid. Please check .streamlit/secrets.toml.")
    if response.status_code == 429:
        raise ValueError("The Hugging Face API rate limit was reached. Please wait a bit and try again.")
    if not response.ok:
        try:
            error_json = response.json()
            error_detail = error_json.get("error") or error_json.get("message") or response.text
        except ValueError:
            error_detail = response.text or f"HTTP {response.status_code}"
        raise ValueError(f"Hugging Face API error: {error_detail}")

    return response


def parse_json_object(raw_text: str) -> dict[str, str]:
    cleaned_text = raw_text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.removeprefix("```json").removeprefix("```").strip()
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3].strip()

    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    cleaned_memory: dict[str, str] = {}
    for key, value in parsed.items():
        if isinstance(key, str) and isinstance(value, str) and value.strip():
            cleaned_memory[key.strip()] = value.strip()

    return cleaned_memory


def stream_assistant_reply(hf_token: str, messages: list[dict[str, str]]):
    response = request_chat_completion(hf_token, messages, stream=True)
    saw_chunk = False

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue

        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        if not content:
            continue

        saw_chunk = True
        yield content
        time.sleep(0.02)

    if not saw_chunk:
        raise ValueError("The API stream returned no message content.")


def extract_memory_from_message(hf_token: str, user_message: str) -> dict[str, str]:
    extraction_prompt = (
        "Given this user message, extract any stable personal facts, preferences, or traits as a JSON object. "
        "Use short snake_case keys like name, preferred_language, interests, or communication_style. "
        "If there is nothing worth saving, return {} only. "
        "Return raw JSON only with string keys and string values. Do not include markdown fences."
    )
    extraction_messages = [
        {"role": "system", "content": extraction_prompt},
        {"role": "user", "content": user_message},
    ]

    response = request_chat_completion(hf_token, extraction_messages, stream=False)
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        return {}

    content = choices[0].get("message", {}).get("content", "").strip()
    if not content:
        return {}

    return parse_json_object(content)


hf_token = load_hf_token()
initialize_chat_state()
initialize_memory_state()
render_sidebar()
active_chat = get_active_chat()

if not hf_token:
    st.error(
        "Missing Hugging Face token. Add HF_TOKEN to .streamlit/secrets.toml before running the app."
    )
else:
    chat_history = st.container(height=500)
    with chat_history:
        if active_chat and active_chat["messages"]:
            for message in active_chat["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        else:
            st.info("Start a new conversation from the sidebar or send a message below.")

    user_prompt = st.chat_input("Type your message here...", disabled=active_chat is None)

    if user_prompt and active_chat:
        active_chat["messages"].append({"role": "user", "content": user_prompt})
        maybe_update_chat_title(active_chat, user_prompt)
        save_chat(active_chat)

        with chat_history:
            with st.chat_message("user"):
                st.markdown(user_prompt)

        try:
            model_messages = build_messages_with_memory(active_chat["messages"], st.session_state.memory)
            with chat_history:
                with st.chat_message("assistant"):
                    reply = st.write_stream(stream_assistant_reply(hf_token, model_messages))

            reply = str(reply).strip()
            active_chat["messages"].append({"role": "assistant", "content": reply})
            save_chat(active_chat)

            extracted_memory = extract_memory_from_message(hf_token, user_prompt)
            if extracted_memory:
                st.session_state.memory.update(extracted_memory)
                save_memory(st.session_state.memory)
        except requests.exceptions.Timeout:
            st.error("The request timed out while contacting Hugging Face. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("Network error: the app could not reach Hugging Face.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
        except ValueError as exc:
            st.error(str(exc))

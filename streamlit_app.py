"""AI Image Studio — Streamlit + OpenRouter image generator with private archiving."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

APP_NAME = "AI Image Studio"
PAINT_LOGO = "🎨"
PAINT_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">
  <path d="M32 6C17.1 6 5 17.2 5 31c0 12.7 10.3 23 23 23 2.8 0 4.8-2.2 4.8-4.8 0-1.2-.5-2.3-1.2-3.2-.7-.8-1.1-1.8-1.1-2.9 0-2.5 2-4.5 4.5-4.5h5.3C49.9 38.6 59 30.8 59 21.5 59 12.4 46.9 6 32 6z" fill="#F4C28C" stroke="#8C5828" stroke-width="2.5"/>
  <circle cx="18" cy="27" r="4.2" fill="#E63946"/>
  <circle cx="26" cy="16" r="4.2" fill="#FFB703"/>
  <circle cx="39" cy="16" r="4.2" fill="#2A9D8F"/>
  <circle cx="47" cy="25" r="4.2" fill="#3A86FF"/>
  <circle cx="43" cy="33" r="3.6" fill="#FFFFFF" stroke="#8C5828" stroke-width="2"/>
</svg>
""".strip()

OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images"
DEFAULT_MODEL = "recraft/recraft-v4.1-flash-v2:free"

DB_PATH = Path(os.environ.get("IMAGE_STUDIO_DB_PATH", "image_studio.db")).expanduser()

ROLE_ADMIN = "admin"
ROLE_USER = "user"

MIN_PASSWORD_LENGTH = 8


# --------------------------------------------------------------------------- #
# Secrets Management
# --------------------------------------------------------------------------- #


def _get_secret(name: str) -> str:
    """Read a secret from ``st.secrets`` first, falling back to environment variables."""
    try:
        if name in st.secrets:
            value = str(st.secrets[name]).strip()
            if value:
                return value
    except Exception:  # noqa: BLE001 - secrets.toml may not exist when using env vars
        pass
    return os.environ.get(name, "").strip()


def _load_required_secrets() -> tuple[str, str, str]:
    """Load and validate required application secrets, stopping the app if missing."""
    api_key = _get_secret("OPENROUTER_API_KEY")
    admin_user = _get_secret("APP_ADMIN_USER")
    admin_pass = _get_secret("APP_ADMIN_PASS")

    missing = [
        key
        for key, value in (
            ("OPENROUTER_API_KEY", api_key),
            ("APP_ADMIN_USER", admin_user),
            ("APP_ADMIN_PASS", admin_pass),
        )
        if not value
    ]
    if missing:
        st.error(
            "Missing required application secrets: "
            + ", ".join(f"`{key}`" for key in missing)
            + ".\n\nConfigure them in `.streamlit/secrets.toml` or as environment variables."
        )
        st.stop()

    return api_key, admin_user, admin_pass


# --------------------------------------------------------------------------- #
# Password Security, Hashing & Database Helpers
# --------------------------------------------------------------------------- #


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _validate_password(password: str, confirm_password: str) -> tuple[bool, str]:
    """Enforce password confirmation and complexity rules."""
    if not password:
        return False, "Password cannot be empty."
    if password != confirm_password:
        return False, "Passwords do not match."
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
    if not any(ch.islower() for ch in password):
        return False, "Password must contain at least one lowercase letter."
    if not any(ch.isupper() for ch in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(ch.isdigit() for ch in password):
        return False, "Password must contain at least one digit."
    if not any(not ch.isalnum() and not ch.isspace() for ch in password):
        return False, "Password must contain at least one special character."
    return True, ""


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    """Return ``(salt_hex, hash_hex)`` using PBKDF2-HMAC-SHA256."""
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return salt.hex(), digest.hex()


def _verify_password(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
    _, computed_hash = _hash_password(password, salt_hex)
    return hmac.compare_digest(computed_hash, expected_hash_hex)


def _connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_db(admin_user: str, admin_pass: str) -> None:
    """Create tables and synchronize the admin account from secrets."""
    with _connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generated_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                prompt TEXT NOT NULL,
                model TEXT NOT NULL,
                filename TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                image_data BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES users (username) ON DELETE CASCADE
            )
            """
        )

        # Ensure only the configured secret admin user holds the admin role
        conn.execute(
            "UPDATE users SET role = ? WHERE role = ? AND username != ?",
            (ROLE_USER, ROLE_ADMIN, admin_user),
        )

        existing = conn.execute(
            "SELECT password_hash, salt, role FROM users WHERE username = ?",
            (admin_user,),
        ).fetchone()

        if existing is None:
            salt_hex, hash_hex = _hash_password(admin_pass)
            conn.execute(
                """
                INSERT INTO users (username, password_hash, salt, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (admin_user, hash_hex, salt_hex, ROLE_ADMIN, _utc_now()),
            )
        else:
            password_matches = _verify_password(
                admin_pass, str(existing["salt"]), str(existing["password_hash"])
            )
            if not password_matches or existing["role"] != ROLE_ADMIN:
                salt_hex, hash_hex = _hash_password(admin_pass)
                conn.execute(
                    """
                    UPDATE users
                    SET password_hash = ?, salt = ?, role = ?
                    WHERE username = ?
                    """,
                    (hash_hex, salt_hex, ROLE_ADMIN, admin_user),
                )


def _register_user(
    username: str, password: str, confirm_password: str
) -> tuple[bool, str]:
    """Register a standard user after validating username and password rules."""
    clean_user = username.strip()
    if not clean_user:
        return False, "Username cannot be empty."

    valid_pw, pw_error = _validate_password(password, confirm_password)
    if not valid_pw:
        return False, pw_error

    salt_hex, hash_hex = _hash_password(password)
    try:
        with _connect_db() as conn:
            conn.execute(
                """
                INSERT INTO users (username, password_hash, salt, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (clean_user, hash_hex, salt_hex, ROLE_USER, _utc_now()),
            )
        return True, "Account created. You can now sign in."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."


def _authenticate_user(username: str, password: str) -> str | None:
    """Verify credentials and return the user's role if valid, else ``None``."""
    clean_user = username.strip()
    if not clean_user or not password:
        return None
    with _connect_db() as conn:
        row = conn.execute(
            "SELECT password_hash, salt, role FROM users WHERE username = ?",
            (clean_user,),
        ).fetchone()
    if row is None:
        return None
    if _verify_password(password, str(row["salt"]), str(row["password_hash"])):
        return str(row["role"])
    return None


def _get_user_role(username: str | None) -> str | None:
    if not username:
        return None
    with _connect_db() as conn:
        row = conn.execute(
            "SELECT role FROM users WHERE username = ?", (username,)
        ).fetchone()
    return str(row["role"]) if row else None


def _is_admin(admin_user: str) -> bool:
    username = st.session_state.get("username")
    return username == admin_user and _get_user_role(username) == ROLE_ADMIN


def _save_generated_image(
    username: str, prompt: str, model: str, filename: str, image_bytes: bytes
) -> None:
    """Persist every generated image immediately in the database."""
    digest = hashlib.sha256(image_bytes).hexdigest()
    with _connect_db() as conn:
        conn.execute(
            """
            INSERT INTO generated_images
                (username, prompt, model, filename, size_bytes, sha256, image_data, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                prompt,
                model,
                filename,
                len(image_bytes),
                digest,
                sqlite3.Binary(image_bytes),
                _utc_now(),
            ),
        )


def _fetch_all_metadata(username_filter: str | None = None) -> list[sqlite3.Row]:
    clauses: list[str] = []
    params: list[object] = []
    if username_filter:
        clauses.append("username = ?")
        params.append(username_filter)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        "SELECT id, username, prompt, model, filename, size_bytes, sha256, created_at "
        f"FROM generated_images{where} ORDER BY id DESC"
    )
    with _connect_db() as conn:
        return list(conn.execute(sql, params))


def _fetch_image_record(record_id: int) -> sqlite3.Row | None:
    with _connect_db() as conn:
        return conn.execute(
            "SELECT * FROM generated_images WHERE id = ?", (int(record_id),)
        ).fetchone()


def _delete_image_record(record_id: int) -> None:
    with _connect_db() as conn:
        conn.execute("DELETE FROM generated_images WHERE id = ?", (int(record_id),))


def _archive_stats() -> sqlite3.Row:
    with _connect_db() as conn:
        return conn.execute(
            """
            SELECT COUNT(*) AS total_images,
                   COUNT(DISTINCT username) AS total_users,
                   COALESCE(SUM(size_bytes), 0) AS total_bytes
            FROM generated_images
            """
        ).fetchone()  # type: ignore[return-value]


def _human_bytes(num_bytes: int | float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} GB"


# --------------------------------------------------------------------------- #
# OpenRouter Image Generation
# --------------------------------------------------------------------------- #


def _generate_images(prompt: str, api_key: str, model: str) -> list[tuple[str, bytes]]:
    """Call OpenRouter Images API and return a list of ``(filename, image_bytes)``."""
    response = requests.post(
        url=OPENROUTER_IMAGES_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": model,
                "prompt": prompt,
            }
        ),
        timeout=120,
    )

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"OpenRouter returned an invalid response (HTTP {response.status_code})."
        ) from exc

    if not response.ok or "error" in result:
        error_detail = result.get("error", response.text)
        if isinstance(error_detail, dict):
            error_detail = error_detail.get("message", str(error_detail))
        raise RuntimeError(f"Image generation failed: {error_detail}")

    items = result.get("data", [])
    if not items:
        raise RuntimeError("OpenRouter returned no image data for this prompt.")

    outputs: list[tuple[str, bytes]] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for idx, image in enumerate(items):
        if "b64_json" in image and image["b64_json"]:
            raw_b64 = str(image["b64_json"])
            if "," in raw_b64 and raw_b64.startswith("data:"):
                raw_b64 = raw_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(raw_b64)
        elif "url" in image and image["url"]:
            img_resp = requests.get(str(image["url"]), timeout=60)
            img_resp.raise_for_status()
            image_bytes = img_resp.content
        else:
            continue
        outputs.append((f"generated_{stamp}_{idx}.png", image_bytes))

    if not outputs:
        raise RuntimeError("Could not decode image bytes from the API response.")
    return outputs


# --------------------------------------------------------------------------- #
# UI Views
# --------------------------------------------------------------------------- #


def _render_auth_screen() -> None:
    st.title(f"{PAINT_LOGO} {APP_NAME}")
    sign_in_tab, register_tab = st.tabs(["Sign in", "Register"])

    with sign_in_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            role = _authenticate_user(username, password)
            if role is None:
                st.error("Invalid username or password.")
            else:
                st.session_state.clear()
                st.session_state["username"] = username.strip()
                st.rerun()

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            st.caption(
                "Password rules: minimum 8 characters, with at least one lowercase "
                "letter, one uppercase letter, one digit, and one special character."
            )
            registered = st.form_submit_button("Register", type="primary")
        if registered:
            ok, message = _register_user(new_username, new_password, confirm_password)
            if ok:
                st.success(message)
            else:
                st.error(message)


def _render_sidebar(username: str, is_admin: bool) -> None:
    with st.sidebar:
        st.header("Account")
        st.write(f"Signed in as **{username}**")
        st.caption("Role: **Admin**" if is_admin else "Role: **User**")

        if is_admin:
            st.radio(
                "Navigation",
                ("Generate Image", "Admin Archive"),
                key="admin_nav",
            )

        st.divider()
        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()


def _render_generator_view(username: str, api_key: str) -> None:
    st.title(f"{PAINT_LOGO} {APP_NAME}")

    with st.form("prompt_form"):
        prompt = st.text_area(
            "Enter your image prompt",
            placeholder="A serene mountain landscape at sunset with dramatic clouds",
            height=110,
        )
        generate_clicked = st.form_submit_button("Generate Image", type="primary")

    if generate_clicked:
        # Clear any previous image immediately so normal users retain no past history
        st.session_state.pop("current_outputs", None)
        clean_prompt = prompt.strip()

        if not clean_prompt:
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner("Generating image…"):
                try:
                    outputs = _generate_images(clean_prompt, api_key, DEFAULT_MODEL)
                    # Store every generated image in the database immediately,
                    # regardless of whether the user downloads it.
                    for filename, image_bytes in outputs:
                        _save_generated_image(
                            username=username,
                            prompt=clean_prompt,
                            model=DEFAULT_MODEL,
                            filename=filename,
                            image_bytes=image_bytes,
                        )
                    st.session_state["current_outputs"] = {
                        "prompt": clean_prompt,
                        "items": outputs,
                    }
                except Exception as exc:  # noqa: BLE001 - surface API error cleanly
                    st.error(str(exc))

    current = st.session_state.get("current_outputs")
    if current:
        st.subheader("Generated Result")
        st.caption(f"Prompt: {current['prompt']}")
        for idx, (filename, image_bytes) in enumerate(current["items"]):
            st.image(image_bytes, caption=filename)
            st.download_button(
                label="Download Image",
                data=image_bytes,
                file_name=filename,
                mime="image/png",
                type="primary",
                on_click="ignore",
                key=f"download_current_{idx}_{filename}",
            )


def _render_admin_archive_view(admin_user: str) -> None:
    if not _is_admin(admin_user):
        st.error("Administrator access required.")
        return

    st.title(f"{PAINT_LOGO} Admin Image Archive")
    st.caption("All images generated across all users are stored here automatically.")

    stats = _archive_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", f"{int(stats['total_images']):,}")
    col2.metric("Unique Users", f"{int(stats['total_users']):,}")
    col3.metric("Total Storage", _human_bytes(stats["total_bytes"]))

    with _connect_db() as conn:
        distinct_users = [
            str(r["username"])
            for r in conn.execute(
                "SELECT DISTINCT username FROM generated_images ORDER BY username"
            )
        ]

    selected_user = st.selectbox("Filter by user", ("All users", *distinct_users))

    records = _fetch_all_metadata(
        username_filter=None if selected_user == "All users" else selected_user,
    )

    if not records:
        st.info("No generated images found in the database.")
        return

    st.dataframe(
        [
            {
                "ID": int(r["id"]),
                "User": r["username"],
                "Prompt": r["prompt"],
                "Model": r["model"],
                "Size": _human_bytes(r["size_bytes"]),
                "Created": r["created_at"],
            }
            for r in records
        ],
        hide_index=True,
        use_container_width=True,
    )

    record_labels = {
        int(r["id"]): f"#{int(r['id'])} · {r['username']} · {r['prompt'][:60]}"
        for r in records
    }
    chosen_id = st.selectbox(
        "Inspect generated image",
        options=list(record_labels.keys()),
        format_func=record_labels.get,
    )

    record = _fetch_image_record(int(chosen_id))
    if record is None:
        st.warning("Record not found.")
        return

    st.divider()
    st.write(f"**User:** {record['username']} · **Created:** {record['created_at']}")
    st.write(f"**Prompt:** {record['prompt']}")
    st.image(record["image_data"], caption=record["filename"])

    dl_col, del_col = st.columns([1, 2])
    with dl_col:
        st.download_button(
            label="Download Original",
            data=record["image_data"],
            file_name=record["filename"],
            mime="image/png",
            type="primary",
            on_click="ignore",
            key=f"admin_dl_{record['id']}",
        )
    with del_col:
        confirm_del = st.checkbox(
            "Confirm permanent deletion", key=f"confirm_del_{record['id']}"
        )
        if st.button(
            "Delete Record",
            disabled=not confirm_del,
            key=f"admin_del_{record['id']}",
        ):
            _delete_image_record(int(record["id"]))
            st.success("Record deleted.")
            st.rerun()


# --------------------------------------------------------------------------- #
# Entry Point
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon=PAINT_LOGO, layout="wide")
    if hasattr(st, "logo"):
        st.logo(PAINT_LOGO_SVG, size="large")

    api_key, admin_user, admin_pass = _load_required_secrets()
    _init_db(admin_user, admin_pass)

    username = st.session_state.get("username")
    role = _get_user_role(username)

    if not username or role is None:
        st.session_state.pop("username", None)
        _render_auth_screen()
        return

    is_admin = _is_admin(admin_user)
    _render_sidebar(username, is_admin)

    if is_admin and st.session_state.get("admin_nav") == "Admin Archive":
        _render_admin_archive_view(admin_user)
    else:
        _render_generator_view(username, api_key)


if __name__ == "__main__":
    main()
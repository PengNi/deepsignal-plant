"""
file_browser.py — Inline file-browser widget for DeepSignal-plant GUI.

Each ``path_input`` call renders a text box plus a 📁 toggle button.
Clicking the button opens a compact directory browser directly below the
field.  Only one browser is open across the whole app at any time
(controlled by ``st.session_state["_fb_active"]``).

All state mutations are done via on_click callbacks so that Streamlit's
widget-update phase commits the change before the next render pass —
this is more reliable than the if-button: ... st.rerun() pattern.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import streamlit as st

# Maximum number of directory or file entries shown per listing.
_MAX_DIRS  = 80
_MAX_FILES = 80


# ─── Callback helpers ─────────────────────────────────────────────────────────

def _cb_toggle(widget_key: str) -> None:
    """Toggle the file browser open/closed."""
    if st.session_state.get("_fb_active") == widget_key:
        st.session_state["_fb_active"] = None
    else:
        # Sync browser dir to current typed value before opening
        sk = st.session_state.get(f"_fb_sk_{widget_key}", "")
        cur = st.session_state.get(sk, "")
        if cur:
            candidate = cur if Path(cur).is_dir() else str(Path(cur).parent)
            if Path(candidate).is_dir():
                st.session_state[_dir_key(widget_key)] = candidate
        st.session_state["_fb_active"] = widget_key


def _cb_nav(dk: str, path: str) -> None:
    """Navigate browser to *path*."""
    st.session_state[dk] = path


def _cb_select_dir(session_key: str, path: str) -> None:
    """Select *path* as the value and close the browser."""
    st.session_state[session_key] = path
    st.session_state["_fb_active"] = None


def _cb_select_file(session_key: str, path: str) -> None:
    """Select *path* as the value and close the browser."""
    st.session_state[session_key] = path
    st.session_state["_fb_active"] = None


def _cb_path_bar(widget_key: str, dk: str) -> None:
    """Navigate to the path typed in the path bar."""
    typed = st.session_state.get(f"_fbpath_{widget_key}", "")
    if typed and Path(typed).is_dir():
        st.session_state[dk] = typed


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _dir_key(widget_key: str) -> str:
    """Session-state key that stores the browser's current directory."""
    return f"_fb_dir_{widget_key}"


def _init_dir(widget_key: str, session_key: str) -> None:
    """Seed the browser's start directory from the current field value."""
    dk = _dir_key(widget_key)
    if dk not in st.session_state:
        cur = st.session_state.get(session_key, "")
        if cur:
            candidate = cur if Path(cur).is_dir() else str(Path(cur).parent)
            st.session_state[dk] = (
                candidate if Path(candidate).is_dir() else os.path.expanduser("~")
            )
        else:
            st.session_state[dk] = os.path.expanduser("~")

    # Store session_key so the toggle callback can find it
    st.session_state[f"_fb_sk_{widget_key}"] = session_key


def _browser_panel(
    session_key: str,
    widget_key: str,
    *,
    select_dirs_only: bool,
    allowed_exts: Optional[List[str]],
) -> None:
    """Render the inline directory-browser panel."""
    dk   = _dir_key(widget_key)
    root = st.session_state.get(dk, os.path.expanduser("~"))

    with st.container(border=True):

        # ── Navigation bar ──────────────────────────────────────────────────
        nav1, nav2, nav3 = st.columns([1, 1, 8])
        with nav1:
            parent = str(Path(root).parent)
            st.button(
                "⬆", key=f"_fbup_{widget_key}", help="Go up one level",
                on_click=_cb_nav,
                args=(dk, parent if parent != root else root),
            )
        with nav2:
            st.button(
                "🏠", key=f"_fbhome_{widget_key}", help="Home directory",
                on_click=_cb_nav,
                args=(dk, os.path.expanduser("~")),
            )
        with nav3:
            st.text_input(
                "path_bar",
                value=root,
                key=f"_fbpath_{widget_key}",
                label_visibility="collapsed",
                placeholder="Type a path to jump to…",
                on_change=_cb_path_bar,
                args=(widget_key, dk),
            )

        # ── Select-this-folder button ───────────────────────────────────────
        folder_label = Path(root).name or root
        st.button(
            f"✓  Select this folder:  **{folder_label}**",
            key=f"_fbseldir_{widget_key}",
            use_container_width=True,
            help=root,
            on_click=_cb_select_dir,
            args=(session_key, root),
        )

        st.divider()

        # ── Directory listing ───────────────────────────────────────────────
        try:
            all_items = sorted(Path(root).iterdir())
        except PermissionError:
            st.warning(f"Permission denied: {root}")
            return
        except Exception as exc:
            st.warning(f"Cannot list directory: {exc}")
            return

        dirs = [p for p in all_items if p.is_dir()  and not p.name.startswith(".")]
        if select_dirs_only:
            files: list[Path] = []
        else:
            files = [
                p for p in all_items
                if p.is_file() and not p.name.startswith(".")
                and (
                    allowed_exts is None
                    or any(p.name.endswith(e) for e in allowed_exts)
                )
            ]

        truncated_dirs  = len(dirs)  > _MAX_DIRS
        truncated_files = len(files) > _MAX_FILES
        dirs  = dirs[:_MAX_DIRS]
        files = files[:_MAX_FILES]

        # Subdirectories
        for i, d in enumerate(dirs):
            c1, c2 = st.columns([9, 1])
            with c1:
                st.caption(f"📁  {d.name}")
            with c2:
                st.button(
                    "→", key=f"_fbd_{widget_key}_{i}",
                    help=f"Open  {d.name}",
                    on_click=_cb_nav,
                    args=(dk, str(d)),
                )

        if truncated_dirs:
            st.caption(f"_… {_MAX_DIRS}+ dirs — navigate or type path above_")

        # Files
        for j, f in enumerate(files):
            c1, c2 = st.columns([9, 1])
            with c1:
                st.caption(f"📄  {f.name}")
            with c2:
                st.button(
                    "✓", key=f"_fbf_{widget_key}_{j}",
                    help=f"Select  {f.name}",
                    on_click=_cb_select_file,
                    args=(session_key, str(f)),
                )

        if truncated_files:
            st.caption(f"_… {_MAX_FILES}+ files — type a path above to narrow down_")

        if not dirs and not files:
            st.caption("_(empty directory)_")


# ─── Public API ───────────────────────────────────────────────────────────────

def path_input(
    label: str,
    session_key: str,
    *,
    widget_key: str,
    help_text: str = "",
    placeholder: str = "",
    select_dirs_only: bool = False,
    allowed_exts: Optional[List[str]] = None,
) -> str:
    """
    Text input with an inline 📁 file-browser toggle.

    Parameters
    ----------
    label            : Field label shown above the text box.
    session_key      : ``st.session_state`` key that holds the selected path.
    widget_key       : Globally unique Streamlit widget key (must not clash).
    help_text        : Tooltip text on the text input.
    placeholder      : Placeholder string inside the text input.
    select_dirs_only : If True, only directories are selectable (no files shown).
    allowed_exts     : If given, only files ending with one of these strings are
                       shown (e.g. ``['.fa', '.fna']``).  ``None`` = show all.
    """
    _init_dir(widget_key, session_key)
    is_open = st.session_state.get("_fb_active") == widget_key

    # ── Input row ─────────────────────────────────────────────────────────
    col_in, col_btn = st.columns([11, 1])

    with col_in:
        raw = st.text_input(
            label,
            value=st.session_state.get(session_key, ""),
            key=widget_key,
            help=help_text,
            placeholder=placeholder,
        )
        st.session_state[session_key] = raw

    with col_btn:
        # Align button vertically with the text input (label height ≈ 1.6 rem)
        st.markdown(
            '<div style="height:1.6rem"></div>',
            unsafe_allow_html=True,
        )
        icon = "📂" if is_open else "📁"
        st.button(
            icon,
            key=f"_fbtn_{widget_key}",
            help="Browse files / folders",
            use_container_width=True,
            on_click=_cb_toggle,
            args=(widget_key,),
        )

    # ── Inline browser panel ───────────────────────────────────────────────
    if is_open:
        _browser_panel(
            session_key, widget_key,
            select_dirs_only=select_dirs_only,
            allowed_exts=allowed_exts,
        )

    return st.session_state.get(session_key, "")

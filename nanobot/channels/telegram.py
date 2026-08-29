"""Telegram channel implementation using python-telegram-bot."""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger
from pydantic import Field
from telegram import BotCommand, ReactionTypeEmoji, ReplyParameters, Update
from telegram.error import BadRequest, TimedOut
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

from nanobot.bus.events import Address, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.paths import get_media_dir
from nanobot.config.schema import Base
from nanobot.security.network import validate_url_target
from nanobot.utils.helpers import split_message
from nanobot.utils.telegram_utils import (
    load_topic_pins,
    save_topic_pins,
    set_cached_profile,
    set_cached_topic_name,
)

TELEGRAM_MAX_MESSAGE_LEN = 4000  # Telegram message character limit
TELEGRAM_REPLY_CONTEXT_MAX_LEN = (
    TELEGRAM_MAX_MESSAGE_LEN  # Max length for reply context in user message
)


def _strip_md(s: str) -> str:
    """Strip markdown inline formatting from text."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"__(.+?)__", r"\1", s)
    s = re.sub(r"~~(.+?)~~", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    return s.strip()


def _render_table_box(table_lines: list[str]) -> str:
    """Convert markdown pipe-table to compact aligned text for <pre> display."""

    def dw(s: str) -> int:
        return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)

    rows: list[list[str]] = []
    has_sep = False
    for line in table_lines:
        cells = [_strip_md(c) for c in line.strip().strip("|").split("|")]
        if all(re.match(r"^:?-+:?$", c) for c in cells if c):
            has_sep = True
            continue
        rows.append(cells)
    if not rows or not has_sep:
        return "\n".join(table_lines)

    ncols = max(len(r) for r in rows)
    for r in rows:
        r.extend([""] * (ncols - len(r)))
    widths = [max(dw(r[c]) for r in rows) for c in range(ncols)]

    def dr(cells: list[str]) -> str:
        return "  ".join(f"{c}{' ' * (w - dw(c))}" for c, w in zip(cells, widths))

    out = [dr(rows[0])]
    out.append("  ".join("─" * w for w in widths))
    for row in rows[1:]:
        out.append(dr(row))
    return "\n".join(out)


def _markdown_to_telegram_html(text: str) -> str:
    """
    Convert markdown to Telegram-safe HTML.
    """
    if not text:
        return ""

    # 1. Extract and protect code blocks (preserve content from other processing)
    code_blocks: list[str] = []

    def save_code_block(m: re.Match) -> str:
        code_blocks.append(m.group(1))
        return f"\x00CB{len(code_blocks) - 1}\x00"

    text = re.sub(r"```[\w]*\n?([\s\S]*?)```", save_code_block, text)

    # 1.5. Convert markdown tables to box-drawing (reuse code_block placeholders)
    lines = text.split("\n")
    rebuilt: list[str] = []
    li = 0
    while li < len(lines):
        if re.match(r"^\s*\|.+\|", lines[li]):
            tbl: list[str] = []
            while li < len(lines) and re.match(r"^\s*\|.+\|", lines[li]):
                tbl.append(lines[li])
                li += 1
            box = _render_table_box(tbl)
            if box != "\n".join(tbl):
                code_blocks.append(box)
                rebuilt.append(f"\x00CB{len(code_blocks) - 1}\x00")
            else:
                rebuilt.extend(tbl)
        else:
            rebuilt.append(lines[li])
            li += 1
    text = "\n".join(rebuilt)

    # 2. Extract and protect inline code
    inline_codes: list[str] = []

    def save_inline_code(m: re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", save_inline_code, text)

    # 3. Headers # Title -> just the title text
    text = re.sub(r"^#{1,6}\s+(.+)$", r"\1", text, flags=re.MULTILINE)

    # 4. Blockquotes > text -> just the text (before HTML escaping)
    text = re.sub(r"^>\s*(.*)$", r"\1", text, flags=re.MULTILINE)

    # 5. Escape HTML special characters
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 6. Links [text](url) - must be before bold/italic to handle nested cases
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    # 7. Bold **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

    # 8. Italic _text_ (avoid matching inside words like some_var_name)
    text = re.sub(r"(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])", r"<i>\1</i>", text)

    # 9. Strikethrough ~~text~~
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)

    # 10. Bullet lists - item -> • item
    text = re.sub(r"^[-*]\s+", "• ", text, flags=re.MULTILINE)

    # 11. Restore inline code with HTML tags
    for i, code in enumerate(inline_codes):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00IC{i}\x00", f"<code>{escaped}</code>")

    # 12. Restore code blocks with HTML tags
    for i, code in enumerate(code_blocks):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00CB{i}\x00", f"<pre><code>{escaped}</code></pre>")

    return text


_SEND_MAX_RETRIES = 3
_SEND_RETRY_BASE_DELAY = 0.5  # seconds, doubled each retry


@dataclass
class _StreamBuf:
    """Per-chat streaming accumulator for progressive message editing."""

    text: str = ""
    message_id: int | None = None
    last_edit: float = 0.0
    stream_id: str | None = None
    consecutive_updates: int = 0
    current_debounce_delay: float = 0.6


class TelegramConfig(Base):
    """Telegram channel configuration."""

    enabled: bool = False
    token: str = ""
    transcription_api_key: str = ""
    allow_from: list[str] = Field(default_factory=list)
    proxy: str | None = None
    reply_to_message: bool = False
    react_emoji: str = "👀"
    group_policy: Literal["open", "mention"] = "mention"
    connection_pool_size: int = 32
    pool_timeout: float = 5.0
    streaming: bool = True


class TelegramChannel(BaseChannel):
    """
    Telegram channel using long polling.

    Simple and reliable - no webhook/public IP needed.
    """

    name = "telegram"
    display_name = "Telegram"
    aliases = ["tg"]

    # Commands registered with Telegram's command menu
    BOT_COMMANDS = [
        BotCommand("start", "Start the bot"),
        BotCommand("new", "Start a new conversation"),
        BotCommand("compact", "Consolidate session history"),
        BotCommand("stage", "Stage messages to memory without dropping them"),
        BotCommand("forget", "Rewind to pinned context"),
        BotCommand("stop", "Cancel active tasks in this session"),
        BotCommand("profiles", "List available agent profiles"),
        BotCommand("topics", "List group topics and their assigned profiles"),
        BotCommand("pin", "Assign an agent profile to the current topic"),
        BotCommand("help", "Show available commands"),
        BotCommand("restart", "Refresh code in-place"),
        BotCommand("status", "Show system status and usage"),
        BotCommand("fallback", "Inspect or set provider fallback slot"),
        BotCommand("speak", "Synthesize text to speech"),
    ]

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return TelegramConfig().model_dump(by_alias=True)

    _STREAM_EDIT_INTERVAL = 0.6  # min seconds between edit_message_text calls

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = TelegramConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: TelegramConfig = config
        self._app: Application | None = None
        self._chat_ids: dict[str, int] = {}  # Map sender_id to chat_id for replies
        self._typing_tasks: dict[str, asyncio.Task] = {}  # chat_id -> typing loop task
        self._media_group_buffers: dict[str, dict] = {}
        self._media_group_tasks: dict[str, asyncio.Task] = {}
        self._message_threads: dict[tuple[str, int], int] = {}
        self._bot_user_id: int | None = None
        self._bot_username: str | None = None
        self._stream_bufs: dict[str, _StreamBuf] = {}  # address_uri -> streaming state
        self._progress_message_id: dict[str, int] = {}  # address_uri -> status message id
        self._progress_history: dict[str, list[str]] = {}  # address_uri -> audit trail
        self._topic_pins: dict[
            str, tuple[int | None, str | None, str | None, bool | None]
        ] = {}  # chat_id:thread_id -> (pinned_message_id, profile_name, topic_name, native_audio)

        # Initialize pins from disk
        persistence = load_topic_pins()
        for chat_id, topics in persistence.items():
            for thread_id, entry in topics.items():
                profile = entry.get("profile") if isinstance(entry, dict) else entry
                topic_name = entry.get("name") if isinstance(entry, dict) else None
                native_audio = entry.get("native_audio") if isinstance(entry, dict) else None
                self._topic_pins[f"{chat_id}:{thread_id}"] = (
                    None,
                    profile,
                    topic_name,
                    native_audio,
                )

    def is_allowed(self, sender_id: str) -> bool:
        """Preserve Telegram's legacy id|username allowlist matching."""
        if super().is_allowed(sender_id):
            return True

        allow_list = getattr(self.config, "allow_from", [])
        if not allow_list or "*" in allow_list:
            return False

        sender_str = str(sender_id)
        if sender_str.count("|") != 1:
            return False

        sid, username = sender_str.split("|", 1)
        if not sid.isdigit() or not username:
            return False

        return sid in allow_list or username in allow_list

    async def start(self) -> None:
        """Start the Telegram bot with long polling."""
        if not self.config.token:
            logger.error("Telegram bot token not configured")
            return

        self._running = True

        proxy = self.config.proxy or None

        # Separate pools so long-polling (getUpdates) never starves outbound sends.
        api_request = HTTPXRequest(
            connection_pool_size=self.config.connection_pool_size,
            pool_timeout=self.config.pool_timeout,
            connect_timeout=30.0,
            read_timeout=30.0,
            proxy=proxy,
        )
        poll_request = HTTPXRequest(
            connection_pool_size=4,
            pool_timeout=self.config.pool_timeout,
            connect_timeout=30.0,
            read_timeout=30.0,
            proxy=proxy,
        )
        builder = (
            Application.builder()
            .token(self.config.token)
            .request(api_request)
            .get_updates_request(poll_request)
        )
        self._app = builder.build()
        self._app.add_error_handler(self._on_error)

        # Add command handlers
        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CommandHandler("new", self._forward_command))
        self._app.add_handler(CommandHandler("stop", self._forward_command))
        self._app.add_handler(CommandHandler("profiles", self._on_profiles))
        self._app.add_handler(CommandHandler("topics", self._on_topics))
        self._app.add_handler(CommandHandler("pin", self._on_pin_command))
        self._app.add_handler(CommandHandler("repl", self._forward_command))
        self._app.add_handler(CommandHandler("restart", self._forward_command))
        self._app.add_handler(CommandHandler("status", self._forward_command))
        self._app.add_handler(CommandHandler("compact", self._forward_command))
        self._app.add_handler(CommandHandler("stage", self._forward_command))
        self._app.add_handler(CommandHandler("forget", self._forward_command))
        self._app.add_handler(CommandHandler("halt", self._forward_command))
        self._app.add_handler(CommandHandler("RIP", self._forward_command))
        self._app.add_handler(CommandHandler("help", self._forward_command))
        self._app.add_handler(CommandHandler("fallback", self._forward_command))
        self._app.add_handler(CommandHandler("speak", self._forward_command))

        # Add message handler for text, photos, voice, documents
        self._app.add_handler(
            MessageHandler(
                (
                    filters.TEXT
                    | filters.PHOTO
                    | filters.VOICE
                    | filters.AUDIO
                    | filters.Document.ALL
                )
                & ~filters.COMMAND,
                self._on_message,
            )
        )

        # Capture topic names when created or renamed
        self._app.add_handler(
            MessageHandler(filters.StatusUpdate.FORUM_TOPIC_CREATED, self._on_forum_topic_created)
        )
        self._app.add_handler(
            MessageHandler(filters.StatusUpdate.FORUM_TOPIC_EDITED, self._on_forum_topic_edited)
        )

        logger.info("Starting Telegram bot (polling mode)...")

        # Initialize and start polling
        await self._app.initialize()
        await self._app.start()

        # Get bot info and register command menu
        bot_info = await self._app.bot.get_me()
        self._bot_user_id = getattr(bot_info, "id", None)
        self._bot_username = getattr(bot_info, "username", None)
        logger.info("Telegram bot @{} connected", bot_info.username)

        try:
            await self._app.bot.set_my_commands(self.BOT_COMMANDS)
            logger.debug("Telegram bot commands registered")
        except Exception as e:
            logger.warning("Failed to register bot commands: {}", e)

        # Start polling (this runs until stopped)
        await self._app.updater.start_polling(
            allowed_updates=["message"],
            drop_pending_updates=True,  # Ignore old messages on startup
        )

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False

        # Cancel all typing indicators
        for chat_id in list(self._typing_tasks):
            self._stop_typing(chat_id)

        for task in self._media_group_tasks.values():
            task.cancel()
        self._media_group_tasks.clear()
        self._media_group_buffers.clear()

        if self._app:
            logger.info("Stopping Telegram bot...")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None

    @staticmethod
    def _get_media_type(path: str) -> str:
        """Guess media type from file extension."""
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in ("jpg", "jpeg", "png", "gif", "webp"):
            return "photo"
        if ext == "ogg":
            return "voice"
        if ext in ("mp3", "m4a", "wav", "aac"):
            return "audio"
        return "document"

    @staticmethod
    def _is_remote_media_url(path: str) -> bool:
        return path.startswith(("http://", "https://"))

    async def _update_audit_trail(self, msg: OutboundMessage) -> None:
        """Maintain a cumulative audit trail of actions."""
        if not msg.metadata.get("_tool_hint", False):
            return
        try:
            address_uri = msg.address.to_uri()
            chat_id_int = int(msg.chat_id)
            thread_id = msg.message_thread_id
            thread_kwargs = {"message_thread_id": thread_id} if thread_id else {}

            # Update audit trail history (keyed by full address URI to isolate topics)
            history = self._progress_history.get(address_uri, [])
            if history:
                # Convert the previous ongoing step to "Done"
                history[-1] = history[-1].replace("⚙️", "✅")
            history.append(f"⚙️ `{msg.content}`")
            self._progress_history[address_uri] = history

            status_text = "🔎 **Audit Trail:**\n" + "\n".join(history)
            html = _markdown_to_telegram_html(status_text)

            # If the status text is getting dangerously long, roll over to a new message
            if len(html) > (TELEGRAM_MAX_MESSAGE_LEN - 500):
                self._progress_message_id.pop(address_uri, None)

            if address_uri in self._progress_message_id:
                try:
                    await self._call_with_retry(
                        self._app.bot.edit_message_text,
                        chat_id=chat_id_int,
                        message_id=self._progress_message_id[address_uri],
                        text=html,
                        parse_mode="HTML",
                    )
                    return
                except Exception as e:
                    if self._is_not_modified_error(e):
                        return
                    # If edit fails (e.g. message too long or deleted), reset and send new
                    logger.debug("Progress edit failed for {}, rolling over: {}", address_uri, e)
                    self._progress_message_id.pop(address_uri, None)

            sent = await self._call_with_retry(
                self._app.bot.send_message,
                chat_id=chat_id_int,
                text=html,
                parse_mode="HTML",
                **thread_kwargs,
            )
            self._progress_message_id[address_uri] = sent.message_id
        except Exception as e:
            logger.warning("Failed to update audit trail for {}: {}", msg.address.to_uri(), e)

    async def _finalize_audit_trail(self, address: "Address") -> None:
        """Finalize the audit trail on delivery of the final response."""
        address_uri = address.to_uri()
        if address_uri not in self._progress_message_id:
            return
        try:
            history = self._progress_history.pop(address_uri, [])
            if history:
                history[-1] = history[-1].replace("⚙️", "✅")
            status_text = "📋 **Audit Trail (Complete):**\n" + "\n".join(history)
            await self._call_with_retry(
                self._app.bot.edit_message_text,
                chat_id=int(address.segments[0]),
                message_id=self._progress_message_id.pop(address_uri),
                text=_markdown_to_telegram_html(status_text),
                parse_mode="HTML",
            )
        except Exception as e:
            logger.debug("Failed to finalize audit trail for {}: {}", address_uri, e)

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Telegram."""
        # Wait up to 30 seconds for the bot to initialize (useful for startup notifications)
        wait_count = 0
        while not self._app and wait_count < 30:
            if wait_count == 0:
                logger.debug("Telegram bot not running yet, waiting for initialization...")
            await asyncio.sleep(1)
            wait_count += 1

        if not self._app:
            logger.warning("Telegram bot not running after 30s timeout, dropping message")
            return

        # Progress handling: Maintain a cumulative audit trail of actions
        if msg.metadata.get("_progress", False):
            await self._update_audit_trail(msg)
            return

        # Turn end: Finalize audit trail and stop typing
        if msg.metadata.get("_turn_end", False):
            await self._finalize_audit_trail(msg.address)
            self._stop_typing(msg.chat_id)
            return

        # Handle system events
        if msg.metadata.get("system_event") == "pin_invalid":
            if msg.chat_id:
                try:
                    await self._app.bot.send_message(
                        chat_id=msg.chat_id,
                        message_thread_id=msg.message_thread_id,
                        text=msg.content,
                        parse_mode="Markdown",
                    )
                except Exception as e:
                    logger.debug("Failed to send pin invalid warning: {}", e)

            # If we know the user's triggering message ID, react with warning
            user_msg_id = msg.metadata.get("message_id")
            if user_msg_id and msg.chat_id:
                await self._add_reaction(msg.chat_id, user_msg_id, "⚠️")
            return

        # Unpack the Address
        address = msg.address
        chat_id_str = address.segments[0]
        try:
            chat_id = int(chat_id_str)
        except ValueError:
            logger.error("Invalid chat_id: {}", chat_id_str)
            return

        message_thread_id = address.segments[1] if len(address.segments) > 1 else None
        reply_to_message_id = msg.metadata.get("message_id")

        # Final response: Finalize audit trail and clear state
        await self._finalize_audit_trail(address)
        self._stop_typing(chat_id_str)

        # If thread ID is missing, try to recover it from context
        if message_thread_id is None and reply_to_message_id is not None:
            message_thread_id = self._message_threads.get((chat_id_str, reply_to_message_id))

        thread_kwargs = {}
        if message_thread_id is not None:
            thread_kwargs["message_thread_id"] = message_thread_id

        reply_params = None
        if self.config.reply_to_message:
            if reply_to_message_id:
                reply_params = ReplyParameters(
                    message_id=reply_to_message_id, allow_sending_without_reply=True
                )

        # Send media files
        for media_path in msg.media or []:
            try:
                media_type = self._get_media_type(media_path)
                sender = {
                    "photo": self._app.bot.send_photo,
                    "voice": self._app.bot.send_voice,
                    "audio": self._app.bot.send_audio,
                }.get(media_type, self._app.bot.send_document)
                param = (
                    "photo"
                    if media_type == "photo"
                    else media_type
                    if media_type in ("voice", "audio")
                    else "document"
                )

                # Telegram Bot API accepts HTTP(S) URLs directly for media params.
                if self._is_remote_media_url(media_path):
                    ok, error = validate_url_target(media_path)
                    if not ok:
                        raise ValueError(f"unsafe media URL: {error}")
                    await self._call_with_retry(
                        sender,
                        chat_id=chat_id,
                        **{param: media_path},
                        reply_parameters=reply_params,
                        **thread_kwargs,
                    )
                    continue

                with open(media_path, "rb") as f:
                    await sender(
                        chat_id=chat_id,
                        **{param: f},
                        reply_parameters=reply_params,
                        **thread_kwargs,
                    )
            except Exception as e:
                filename = media_path.rsplit("/", 1)[-1]
                logger.error("Failed to send media {}: {}", media_path, e)
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=f"[Failed to send: {filename}]",
                    reply_parameters=reply_params,
                    **thread_kwargs,
                )

        # Send text content
        if msg.content and msg.content != "[empty message]":
            for chunk in split_message(msg.content, TELEGRAM_MAX_MESSAGE_LEN):
                await self._send_text(chat_id, chunk, reply_params, thread_kwargs)

    async def _call_with_retry(self, fn, *args, **kwargs):
        """Call an async Telegram API function with retry on pool/network timeout."""
        for attempt in range(1, _SEND_MAX_RETRIES + 1):
            try:
                return await fn(*args, **kwargs)
            except TimedOut:
                if attempt == _SEND_MAX_RETRIES:
                    raise
                delay = _SEND_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Telegram timeout (attempt {}/{}), retrying in {:.1f}s",
                    attempt,
                    _SEND_MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)

    async def _send_text(
        self,
        chat_id: int,
        text: str,
        reply_params=None,
        thread_kwargs: dict | None = None,
    ) -> None:
        """Send a plain text message with HTML fallback."""
        try:
            html = _markdown_to_telegram_html(text)
            sent = await self._call_with_retry(
                self._app.bot.send_message,
                chat_id=chat_id,
                text=html,
                parse_mode="HTML",
                reply_parameters=reply_params,
                **(thread_kwargs or {}),
            )
            logger.debug(
                "Telegram API accepted message (HTML): id={} chat={}", sent.message_id, chat_id
            )
        except Exception as e:
            logger.warning("HTML parse failed, falling back to plain text: {}", e)
            try:
                sent = await self._call_with_retry(
                    self._app.bot.send_message,
                    chat_id=chat_id,
                    text=text,
                    reply_parameters=reply_params,
                    **(thread_kwargs or {}),
                )
                logger.debug(
                    "Telegram API accepted message (Plain): id={} chat={}", sent.message_id, chat_id
                )
            except Exception as e2:
                logger.error("Error sending Telegram message: {}", e2)
                raise

    @staticmethod
    def _is_not_modified_error(exc: Exception) -> bool:
        return isinstance(exc, BadRequest) and "message is not modified" in str(exc).lower()

    async def send_delta(self, msg: OutboundMessage) -> None:
        """Progressive message editing: send on first delta, edit on subsequent ones."""
        if not self._app:
            return

        address_uri = msg.address.to_uri()
        int_chat_id = int(msg.address.segments[0])
        stream_id = msg.metadata.get("_stream_id")
        message_thread_id = msg.address.segments[1] if len(msg.address.segments) > 1 else None
        thread_kwargs = {"message_thread_id": message_thread_id} if message_thread_id else {}

        if msg.metadata.get("_stream_end"):
            buf = self._stream_bufs.get(address_uri)
            if not buf or not buf.text:
                return
            if stream_id is not None and buf.stream_id is not None and buf.stream_id != stream_id:
                return
            self._stop_typing(str(int_chat_id))

            if buf.message_id is None:
                # Final flush for short messages that never hit the initial threshold
                try:
                    sent = await self._call_with_retry(
                        self._app.bot.send_message,
                        chat_id=int_chat_id,
                        text=_markdown_to_telegram_html(buf.text),
                        parse_mode="HTML",
                        **thread_kwargs,
                    )
                    buf.message_id = sent.message_id
                except Exception as e:
                    logger.warning("Final stream send failed: {}", e)
                    self._stream_bufs.pop(address_uri, None)
                    return
            else:
                try:
                    html = _markdown_to_telegram_html(buf.text)
                    await self._call_with_retry(
                        self._app.bot.edit_message_text,
                        chat_id=int_chat_id,
                        message_id=buf.message_id,
                        text=html,
                        parse_mode="HTML",
                    )
                except Exception as e:
                    if self._is_not_modified_error(e):
                        logger.debug("Final stream edit already applied for {}", address_uri)
                        self._stream_bufs.pop(address_uri, None)
                        return
                    logger.debug("Final stream edit failed (HTML), trying plain: {}", e)
                    try:
                        await self._call_with_retry(
                            self._app.bot.edit_message_text,
                            chat_id=int_chat_id,
                            message_id=buf.message_id,
                            text=buf.text,
                        )
                    except Exception as e2:
                        if self._is_not_modified_error(e2):
                            logger.debug(
                                "Final stream plain edit already applied for {}", address_uri
                            )
                            self._stream_bufs.pop(address_uri, None)
                            return
                        logger.warning("Final stream edit failed: {}", e2)
                        raise  # Let ChannelManager handle retry

            self._stream_bufs.pop(address_uri, None)
            return

        buf = self._stream_bufs.get(address_uri)
        if buf is None or (
            stream_id is not None and buf.stream_id is not None and buf.stream_id != stream_id
        ):
            buf = _StreamBuf(stream_id=stream_id)
            self._stream_bufs[address_uri] = buf
        elif buf.stream_id is None:
            buf.stream_id = stream_id
        buf.text += msg.content

        if not buf.text.strip():
            return

        now = time.monotonic()

        # Check if we need to roll over to a new message
        if buf.message_id is not None and len(buf.text) > TELEGRAM_MAX_MESSAGE_LEN:
            chunks = split_message(buf.text, TELEGRAM_MAX_MESSAGE_LEN)
            first_chunk = chunks.pop(0)

            try:
                html = _markdown_to_telegram_html(first_chunk)
                await self._call_with_retry(
                    self._app.bot.edit_message_text,
                    chat_id=int_chat_id,
                    message_id=buf.message_id,
                    text=html,
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.debug("Rollover edit failed (HTML), trying plain: {}", e)
                await self._call_with_retry(
                    self._app.bot.edit_message_text,
                    chat_id=int_chat_id,
                    message_id=buf.message_id,
                    text=first_chunk,
                )

            for chunk in chunks:
                sent = await self._call_with_retry(
                    self._app.bot.send_message,
                    chat_id=int_chat_id,
                    text=chunk,
                    **thread_kwargs,
                )
                buf.message_id = sent.message_id

            buf.text = chunks[-1] if chunks else ""
            buf.last_edit = now
            return

        if buf.message_id is None:
            # Wait for a minimum amount of content before the first send to avoid
            # tiny notifications (e.g., "T", "The").
            # Send if: length > 60 OR contains a newline OR contains a sentence terminator.
            if len(buf.text) < 60 and not any(c in buf.text for c in ("\n", ".", "!", "?")):
                return

            # Initial send: don't split yet, just send the first chunk if too large
            # and let the next edit/flush handle the rest.
            text_to_send = buf.text
            if len(text_to_send) > TELEGRAM_MAX_MESSAGE_LEN:
                chunks = split_message(text_to_send, TELEGRAM_MAX_MESSAGE_LEN)
                text_to_send = chunks[0]
                buf.text = text_to_send  # Truncate buffer to what was actually sent

            try:
                sent = await self._call_with_retry(
                    self._app.bot.send_message,
                    chat_id=int_chat_id,
                    text=text_to_send,
                    **thread_kwargs,
                )
                buf.message_id = sent.message_id
                buf.last_edit = now
            except Exception as e:
                logger.warning("Stream initial send failed: {}", e)
                raise  # Let ChannelManager handle retry
        # --- Start Adaptive Debounce Logic ---
        now = time.monotonic()

        # 1. Check if the time elapsed is less than the current required debounce delay
        if (now - buf.last_edit) < buf.current_debounce_delay:
            buf.consecutive_updates += 1
            if buf.consecutive_updates % 5 == 0:
                logger.debug(
                    "Debouncing stream: Update {} arrived within {}s window. Next target: {}s.",
                    buf.consecutive_updates,
                    buf.current_debounce_delay,
                    buf.current_debounce_delay * 1.5,
                )

            # 2. Dynamically increase the debounce delay if we are receiving too many updates fast
            # Threshold: If we get 3 updates in a row within the current delay window, increase the delay.
            if buf.consecutive_updates >= 3 and buf.current_debounce_delay < 3.0:
                buf.current_debounce_delay = min(3.0, buf.current_debounce_delay * 1.5)
                logger.info(
                    "Stream spike detected. Increasing debounce delay to {}s.",
                    buf.current_debounce_delay,
                )

            # Early exit: We are in a burst; wait for the timer to pass naturally
            return

        # 3. If the time elapsed is greater than the required debounce delay: Time to flush!
        # This is equivalent to the old check, but dynamically adjusted.
        try:
            await self._call_with_retry(
                self._app.bot.edit_message_text,
                chat_id=int_chat_id,
                message_id=buf.message_id,
                text=buf.text,
            )
            # Success: Reset state and stabilize the delay
            buf.last_edit = now
            buf.consecutive_updates = 0
            buf.current_debounce_delay = 0.6  # Return to baseline minimum
            logger.debug("Stream delta sent successfully. Resetting debounce state.")
        except Exception as e:
            if self._is_not_modified_error(e):
                buf.last_edit = now
                buf.consecutive_updates = 0
                buf.current_debounce_delay = 0.6  # Return to baseline minimum
                logger.debug("Stream edit successful: message not modified (idempotent success)")
                return
            logger.warning("Stream edit failed: {}", e)
            # Crucially, do NOT reset the counter on failure, allowing aggressive retries/pauses.
            raise  # Let ChannelManager handle retry
        # --- End Adaptive Debounce Logic ---

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not update.message or not update.effective_user:
            return

        user = update.effective_user
        await update.message.reply_text(
            f"👋 Hi {user.first_name}! I'm nanobot.\n\n"
            "Send me a message and I'll respond!\n"
            "Type /help to see available commands."
        )

    async def _on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command, bypassing ACL so all users can access it."""
        if not update.message:
            return
        await update.message.reply_text(
            "🐈 nanobot commands:\n"
            "/new — Start a new conversation\n"
            "/stop — Stop current tasks\n"
            "/profiles — List available agent profiles\n"
            "/topics — List group topics and their assigned profiles\n"
            "/pin <profile> — Assign an agent profile to the current topic\n"
            "/restart — Restart the bot\n"
            "/status — Show system status and usage\n"
            "/help — Show available commands"
        )

    async def _on_profiles(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /profiles command."""
        if not update.message:
            return

        if not self.global_config or not self.global_config.agents:
            await update.message.reply_text("Agent configuration not available.")
            return

        profiles = []
        agents = self.global_config.agents

        # List defaults
        try:
            defaults = agents.defaults
            models = defaults.fallback_models if defaults.fallback_models else [defaults.model]
            profiles.append(f"• **defaults**: {', '.join(models)}")
        except Exception:
            pass

        # List extra agents
        agent_dict = agents.model_dump(by_alias=True)
        for name in agent_dict.keys():
            if name in ("defaults", "model_config", "model_fields", "model_computed_fields"):
                continue
            try:
                agent = agents.get_agent(name)
                models = agent.fallback_models if agent.fallback_models else [agent.model]
                profiles.append(f"• **{name}**: {', '.join(models)}")
            except Exception:
                continue

        if not profiles:
            await update.message.reply_text("No agent profiles configured.")
            return

        await update.message.reply_text(
            "🎭 **Available Agent Profiles:**\n\n" + "\n".join(profiles), parse_mode="Markdown"
        )

    async def _on_topics(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /topics command."""
        if not update.message:
            return

        chat_id = str(update.message.chat_id)
        lines = ["📌 **Topic Profile Mappings:**"]

        found = False
        # Collect and sort topics for consistent display
        topics_to_show = []
        for key, value in self._topic_pins.items():
            if key.startswith(f"{chat_id}:"):
                found = True
                _, topic_id_str = key.split(":", 1)
                _, profile, topic_name, native_audio = value
                topics_to_show.append((int(topic_id_str), profile, topic_name, native_audio))

        topics_to_show.sort()

        for topic_id, profile, topic_name, native_audio in topics_to_show:
            profile_str = f"`{profile}`" if profile else "_(none)_"
            audio_str = " 🔊" if native_audio else (" 🔇" if native_audio is False else "")
            if topic_name:
                lines.append(f'• Topic {topic_id} "{topic_name}" = {profile_str}{audio_str}')
            else:
                lines.append(f"• Topic {topic_id}: {profile_str}{audio_str}")

        if not found:
            lines.append("No topics discovered yet in this chat.")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _on_pin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /pin <profile> [flags] command."""
        if not update.message:
            return

        thread_id = getattr(update.message, "message_thread_id", None)
        if thread_id is None:
            if update.message.chat.type == "private":
                await update.message.reply_text(
                    "Profiles are assigned automatically in private chats."
                )
            else:
                await update.message.reply_text(
                    "This command can only be used within a forum topic."
                )
            return

        args = context.args
        if not args:
            await update.message.reply_text(
                "Usage: `/pin <profile_name> [--native-audio | --no-native-audio]`",
                parse_mode="Markdown",
            )
            return

        profile_name = args[0].lower()
        native_audio = None
        if "--native-audio" in args:
            native_audio = True
        elif "--no-native-audio" in args:
            native_audio = False

        # Validate profile name
        valid = False
        if profile_name == "defaults":
            valid = True
        elif self.global_config and self.global_config.agents:
            try:
                self.global_config.agents.get_agent(profile_name)
                valid = True
            except Exception:
                pass

        if not valid:
            await update.message.reply_text(
                f"❌ Invalid profile name: `{profile_name}`.\nUse /profiles to see available options.",
                parse_mode="Markdown",
            )
            return

        chat_id_str = str(update.message.chat_id)
        cache_key = f"{chat_id_str}:{thread_id}"

        # Update in-memory
        _, _, topic_name, existing_native = self._topic_pins.get(
            cache_key, (None, None, None, None)
        )
        self._topic_pins[cache_key] = (
            None,
            profile_name,
            topic_name,
            native_audio if native_audio is not None else existing_native,
        )

        # Save to disk
        persistence = load_topic_pins()
        set_cached_profile(
            persistence, chat_id_str, thread_id, profile_name, topic_name, native_audio=native_audio
        )
        save_topic_pins(persistence)

        msg_text = f"✅ Topic {thread_id} pinned to profile: `{profile_name}`"
        if native_audio is not None:
            msg_text += f" (Native Audio: {'Enabled' if native_audio else 'Disabled'})"
        await update.message.reply_text(msg_text, parse_mode="Markdown")

    @staticmethod
    def _sender_id(user) -> str:
        """Build sender_id with username for allowlist matching."""
        sid = str(user.id)
        return f"{sid}|{user.username}" if user.username else sid

    async def _get_topic_profile_pin(
        self, chat_id: int, thread_id: int
    ) -> tuple[str | None, bool | None]:
        """Return the cached profile and native_audio flag for a forum topic."""
        cache_key = f"{chat_id}:{thread_id}"
        entry = self._topic_pins.get(cache_key, (None, None, None, None))
        return entry[1], entry[3]

    async def _on_forum_topic_created(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Capture the topic name when a new forum topic is created."""
        msg = update.message
        if not msg:
            return
        thread_id = getattr(msg, "message_thread_id", None)
        topic = getattr(msg, "forum_topic_created", None)
        if not thread_id or not topic:
            return
        topic_name = getattr(topic, "name", None)
        cache_key = f"{msg.chat_id}:{thread_id}"
        cached_pid, cached_profile, cached_name, cached_native = self._topic_pins.get(
            cache_key, (None, None, None, None)
        )
        self._topic_pins[cache_key] = (
            cached_pid,
            cached_profile,
            topic_name or cached_name,
            cached_native,
        )
        logger.info(
            "Forum topic created: key {} topic {} name={}", cache_key, thread_id, topic_name
        )
        # Persist to disk
        persistence = load_topic_pins()
        set_cached_topic_name(persistence, str(msg.chat_id), thread_id, topic_name or cached_name)
        if cached_profile:
            set_cached_profile(persistence, str(msg.chat_id), thread_id, cached_profile)
        save_topic_pins(persistence)

    async def _on_forum_topic_edited(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Capture the new topic name when a forum topic is renamed."""
        msg = update.message
        if not msg:
            return
        thread_id = getattr(msg, "message_thread_id", None)
        topic_edit = getattr(msg, "forum_topic_edited", None)
        if not thread_id or not topic_edit:
            return
        topic_name = getattr(topic_edit, "name", None)
        cache_key = f"{msg.chat_id}:{thread_id}"
        cached_pid, cached_profile, cached_name, cached_native = self._topic_pins.get(
            cache_key, (None, None, None, None)
        )
        self._topic_pins[cache_key] = (
            cached_pid,
            cached_profile,
            topic_name or cached_name,
            cached_native,
        )
        logger.info("Forum topic edited: key {} topic {} name={}", cache_key, thread_id, topic_name)
        # Persist to disk
        persistence = load_topic_pins()
        set_cached_topic_name(persistence, str(msg.chat_id), thread_id, topic_name or cached_name)
        if cached_profile:
            set_cached_profile(persistence, str(msg.chat_id), thread_id, cached_profile)
        save_topic_pins(persistence)

    @staticmethod
    def _derive_topic_session_key(message) -> str | None:
        """Derive topic-scoped session key for non-private Telegram chats."""
        message_thread_id = getattr(message, "message_thread_id", None)
        if message.chat.type == "private" or message_thread_id is None:
            return None
        return f"telegram:{message.chat_id}:topic:{message_thread_id}"

    @staticmethod
    def _build_message_metadata(message, user) -> dict:
        """Build common Telegram inbound metadata payload."""
        reply_to = getattr(message, "reply_to_message", None)
        return {
            "message_id": message.message_id,
            "user_id": user.id,
            "username": user.username,
            "first_name": user.first_name,
            "is_group": message.chat.type != "private",
            "message_thread_id": getattr(message, "message_thread_id", None),
            "is_forum": bool(getattr(message.chat, "is_forum", False)),
            "reply_to_message_id": getattr(reply_to, "message_id", None) if reply_to else None,
        }

    @staticmethod
    def _extract_reply_context(message) -> str | None:
        """Extract text from the message being replied to, if any."""
        reply = getattr(message, "reply_to_message", None)
        if not reply:
            return None
        text = getattr(reply, "text", None) or getattr(reply, "caption", None) or ""
        if len(text) > TELEGRAM_REPLY_CONTEXT_MAX_LEN:
            text = text[:TELEGRAM_REPLY_CONTEXT_MAX_LEN] + "..."
        return f"[Reply to: {text}]" if text else None

    async def _download_message_media(
        self, msg, *, add_failure_content: bool = False, native_audio: bool | None = None
    ) -> tuple[list[str], list[str]]:
        """Download media from a message (current or reply). Returns (media_paths, content_parts).

        When native_audio is True, voice/audio files are passed as-is without transcription.
        When native_audio is False or None, audio is transcribed and only text is returned.
        """
        media_file = None
        media_type = None
        if getattr(msg, "photo", None):
            media_file = msg.photo[-1]
            media_type = "image"
        elif getattr(msg, "voice", None):
            media_file = msg.voice
            media_type = "voice"
        elif getattr(msg, "audio", None):
            media_file = msg.audio
            media_type = "audio"
        elif getattr(msg, "document", None):
            media_file = msg.document
            media_type = "file"
        elif getattr(msg, "video", None):
            media_file = msg.video
            media_type = "video"
        elif getattr(msg, "video_note", None):
            media_file = msg.video_note
            media_type = "video"
        elif getattr(msg, "animation", None):
            media_file = msg.animation
            media_type = "animation"
        if not media_file or not self._app:
            return [], []
        try:
            file = await self._app.bot.get_file(media_file.file_id)
            ext = self._get_extension(
                media_type,
                getattr(media_file, "mime_type", None),
                getattr(media_file, "file_name", None),
            )
            media_dir = get_media_dir("telegram")
            unique_id = getattr(media_file, "file_unique_id", media_file.file_id)
            file_path = media_dir / f"{unique_id}{ext}"
            await file.download_to_drive(str(file_path))
            path_str = str(file_path)
            if media_type in ("voice", "audio"):
                if native_audio:
                    logger.info("Passing raw {} file: {}", media_type, path_str)
                    return [path_str], []
                transcription = await self.transcribe_audio(file_path)
                if transcription:
                    logger.info("Transcribed {}: {}...", media_type, transcription[:50])
                    return [], [f"[transcription: {transcription}]"]
                return [], [f"[{media_type}: {path_str}]"]
            return [path_str], [f"[{media_type}: {path_str}]"]
        except Exception as e:
            logger.warning("Failed to download message media: {}", e)
            if add_failure_content:
                return [], [f"[{media_type}: download failed]"]
            return [], []

    async def _ensure_bot_identity(self) -> tuple[int | None, str | None]:
        """Load bot identity once and reuse it for mention/reply checks."""
        if self._bot_user_id is not None or self._bot_username is not None:
            return self._bot_user_id, self._bot_username
        if not self._app:
            return None, None
        bot_info = await self._app.bot.get_me()
        self._bot_user_id = getattr(bot_info, "id", None)
        self._bot_username = getattr(bot_info, "username", None)
        return self._bot_user_id, self._bot_username

    @staticmethod
    def _has_mention_entity(
        text: str,
        entities,
        bot_username: str,
        bot_id: int | None,
    ) -> bool:
        """Check Telegram mention entities against the bot username."""
        handle = f"@{bot_username}".lower()
        for entity in entities or []:
            entity_type = getattr(entity, "type", None)
            if entity_type == "text_mention":
                user = getattr(entity, "user", None)
                if user is not None and bot_id is not None and getattr(user, "id", None) == bot_id:
                    return True
                continue
            if entity_type != "mention":
                continue
            offset = getattr(entity, "offset", None)
            length = getattr(entity, "length", None)
            if offset is None or length is None:
                continue
            if text[offset : offset + length].lower() == handle:
                return True
        return handle in text.lower()

    async def _is_group_message_for_bot(self, message) -> bool:
        """Allow group messages when policy is open, @mentioned, or replying to the bot."""
        if message.chat.type == "private" or self.config.group_policy == "open":
            return True

        bot_id, bot_username = await self._ensure_bot_identity()
        if bot_username:
            text = message.text or ""
            caption = message.caption or ""
            if self._has_mention_entity(
                text,
                getattr(message, "entities", None),
                bot_username,
                bot_id,
            ):
                return True
            if self._has_mention_entity(
                caption,
                getattr(message, "caption_entities", None),
                bot_username,
                bot_id,
            ):
                return True

        reply_user = getattr(getattr(message, "reply_to_message", None), "from_user", None)
        return bool(bot_id and reply_user and reply_user.id == bot_id)

    def _remember_thread_context(self, message) -> None:
        """Cache topic thread id by chat/message id for follow-up replies."""
        message_thread_id = getattr(message, "message_thread_id", None)
        if message_thread_id is None:
            return
        key = (str(message.chat_id), message.message_id)
        self._message_threads[key] = message_thread_id
        if len(self._message_threads) > 1000:
            self._message_threads.pop(next(iter(self._message_threads)))

    async def _forward_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Forward slash commands to the bus for unified handling in AgentLoop."""
        if not update.message or not update.effective_user:
            return
        message = update.message
        user = update.effective_user
        self._remember_thread_context(message)

        metadata = self._build_message_metadata(message, user)

        # Profile pinning support for forum topics
        if (thread_id := getattr(message, "message_thread_id", None)) is not None:
            # Lazy-fill topic mapping
            cache_key = f"{message.chat_id}:{thread_id}"
            if cache_key not in self._topic_pins:
                self._topic_pins[cache_key] = (None, None, None, None)
                # Note: we don't save to disk on lazy-fill to avoid excessive writes
                # but it will appear in /topics for the current session.

            profile, _ = await self._get_topic_profile_pin(message.chat_id, thread_id)
            if profile:
                metadata["agent_profile"] = profile

        # Create the unified address
        segments = [str(message.chat_id)]
        if thread_id is not None:
            segments.append(str(thread_id))
        address = Address(channel="tg", segments=tuple(segments))

        await self._handle_message(
            address=address,
            sender_id=self._sender_id(user),
            content=message.text or "",
            metadata=metadata,
        )

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages (text, photos, voice, documents)."""
        if not update.message or not update.effective_user:
            return

        message = update.message
        user = update.effective_user
        chat_id = message.chat_id
        sender_id = self._sender_id(user)
        self._remember_thread_context(message)

        # Store chat_id for replies
        self._chat_ids[sender_id] = chat_id

        if not await self._is_group_message_for_bot(message):
            return

        # Build content from text and/or media
        content_parts = []
        media_paths = []

        # Text content
        if message.text:
            content_parts.append(message.text)
        if message.caption:
            content_parts.append(message.caption)

        # Fetch topic pin early for native_audio awareness
        str_chat_id = str(chat_id)
        thread_id = getattr(message, "message_thread_id", None)
        native_audio: bool | None = None
        pin_profile: str | None = None
        if thread_id is not None:
            cache_key = f"{str_chat_id}:{thread_id}"
            if cache_key not in self._topic_pins:
                self._topic_pins[cache_key] = (None, None, None, None)
            pin_profile, pin_native_audio = await self._get_topic_profile_pin(chat_id, thread_id)
            native_audio = pin_native_audio
            if pin_profile:
                # Will be set in metadata below
                pass

        # Download current message media
        current_media_paths, current_media_parts = await self._download_message_media(
            message, add_failure_content=True, native_audio=native_audio
        )
        media_paths.extend(current_media_paths)
        content_parts.extend(current_media_parts)
        if current_media_paths:
            logger.debug("Downloaded message media to {}", current_media_paths[0])

        # Reply context: text and/or media from the replied-to message
        reply = getattr(message, "reply_to_message", None)
        if reply is not None:
            reply_ctx = self._extract_reply_context(message)
            reply_media, reply_media_parts = await self._download_message_media(
                reply, native_audio=native_audio
            )
            if reply_media:
                media_paths = reply_media + media_paths
                logger.debug("Attached replied-to media: {}", reply_media[0])
            tag = reply_ctx or (
                f"[Reply to: {reply_media_parts[0]}]" if reply_media_parts else None
            )
            if tag:
                content_parts.insert(0, tag)
        content = "\n".join(content_parts) if content_parts else "[empty message]"

        logger.debug("Telegram message from {}: {}...", sender_id, content[:50])

        metadata = self._build_message_metadata(message, user)
        if pin_profile:
            metadata["agent_profile"] = pin_profile

        # Create the unified address
        segments = [str_chat_id]
        if thread_id is not None:
            segments.append(str(thread_id))
        address = Address(channel="tg", segments=tuple(segments))

        # Telegram media groups: buffer briefly, forward as one aggregated turn.
        if media_group_id := getattr(message, "media_group_id", None):
            key = f"{str_chat_id}:{media_group_id}"
            if key not in self._media_group_buffers:
                self._media_group_buffers[key] = {
                    "sender_id": sender_id,
                    "address": address,
                    "contents": [],
                    "media": [],
                    "metadata": metadata,
                }
                self._start_typing(str_chat_id)
                await self._add_reaction(str_chat_id, message.message_id, self.config.react_emoji)
            buf = self._media_group_buffers[key]
            if content and content != "[empty message]":
                buf["contents"].append(content)
            buf["media"].extend(media_paths)
            if key not in self._media_group_tasks:
                self._media_group_tasks[key] = asyncio.create_task(self._flush_media_group(key))
            return

        # Start typing indicator before processing
        self._start_typing(str_chat_id)
        await self._add_reaction(str_chat_id, message.message_id, self.config.react_emoji)

        # Forward to the message bus
        segments = [str_chat_id]
        if thread_id is not None:
            segments.append(str(thread_id))
        address = Address(channel="tg", segments=tuple(segments))

        await self._handle_message(
            address=address,
            sender_id=sender_id,
            content=content,
            media=media_paths,
            metadata=metadata,
        )

    async def _flush_media_group(self, key: str) -> None:
        """Wait briefly, then forward buffered media-group as one turn."""
        try:
            await asyncio.sleep(0.6)
            if not (buf := self._media_group_buffers.pop(key, None)):
                return
            content = "\n".join(buf["contents"]) or "[empty message]"
            await self._handle_message(
                address=buf["address"],
                sender_id=buf["sender_id"],
                content=content,
                media=list(dict.fromkeys(buf["media"])),
                metadata=buf["metadata"],
            )
        finally:
            self._media_group_tasks.pop(key, None)

    def _start_typing(self, chat_id: str) -> None:
        """Start sending 'typing...' indicator for a chat."""
        # Cancel any existing typing task for this chat
        self._stop_typing(chat_id)
        self._typing_tasks[chat_id] = asyncio.create_task(self._typing_loop(chat_id))

    def _stop_typing(self, chat_id: str) -> None:
        """Stop the typing indicator for a chat."""
        task = self._typing_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()

    async def _add_reaction(self, chat_id: str, message_id: int, emoji: str) -> None:
        """Add emoji reaction to a message (best-effort, non-blocking)."""
        if not self._app or not emoji:
            return
        try:
            await self._app.bot.set_message_reaction(
                chat_id=int(chat_id),
                message_id=message_id,
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
        except Exception as e:
            logger.debug("Telegram reaction failed: {}", e)

    async def _typing_loop(self, chat_id: str) -> None:
        """Repeatedly send 'typing' action until cancelled."""
        try:
            while self._app:
                await self._app.bot.send_chat_action(chat_id=int(chat_id), action="typing")
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("Typing indicator stopped for {}: {}", chat_id, e)

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log polling / handler errors instead of silently swallowing them."""
        from telegram.error import NetworkError, TimedOut

        if isinstance(context.error, (NetworkError, TimedOut)):
            logger.warning("Telegram network issue: {}", str(context.error))
        else:
            logger.error("Telegram error: {}", context.error)

    def _get_extension(
        self,
        media_type: str,
        mime_type: str | None,
        filename: str | None = None,
    ) -> str:
        """Get file extension based on media type or original filename."""
        if mime_type:
            ext_map = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif",
                "audio/ogg": ".ogg",
                "audio/mpeg": ".mp3",
                "audio/mp4": ".m4a",
            }
            if mime_type in ext_map:
                return ext_map[mime_type]

        type_map = {"image": ".jpg", "voice": ".ogg", "audio": ".mp3", "file": ""}
        if ext := type_map.get(media_type, ""):
            return ext

        if filename:
            from pathlib import Path

            return "".join(Path(filename).suffixes)

        return ""

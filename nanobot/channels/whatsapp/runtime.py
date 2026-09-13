# pyright: reportConstantRedefinition=false, reportMissingTypeStubs=false, reportUnusedFunction=false
"""WhatsApp channel implementation using neonize."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast
from urllib.parse import urlparse

import httpx
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.protocol import parse_silent_ack
from nanobot.config.paths import get_media_dir, get_runtime_subdir
from nanobot.config.schema import Base
from nanobot.security.network import PinnedDNSAsyncTransport


class WhatsAppRoutingConfig(Base):
    """WhatsApp deterministic dual-instance routing configuration."""

    enabled: bool = True
    staff_numbers: list[str] = Field(default_factory=list)
    staff_groups: list[str] = Field(default_factory=list)
    guide_instance_url: str = "http://127.0.0.1:18791/v1/chat/completions"
    guide_model_name: str = "nanobot"
    forward_timeout_seconds: float = 45.0


class WhatsAppConfig(Base):
    """WhatsApp channel configuration."""

    enabled: bool = False
    allow_from: list[str] = Field(default_factory=list)
    group_policy: Literal["open", "mention"] = "open"
    database_path: str = ""
    lid_mappings: dict[str, str] = Field(default_factory=dict)
    routing: WhatsAppRoutingConfig = Field(default_factory=WhatsAppRoutingConfig)



class _NeonizeAPI(NamedTuple):
    NewAClient: Any
    ConnectedEv: Any
    DisconnectedEv: Any
    MessageEv: Any
    PairStatusEv: Any
    build_jid: Any
    detect_mime: Any
    detect_buffer: Any


class _MediaInfo(NamedTuple):
    kind: str
    message: Any
    mimetype: str
    filename: str
    is_voice: bool = False


_NEONIZE_API: _NeonizeAPI | None = None
_JID_RE = re.compile(r"^(?P<user>[^@]+)@(?P<server>[^@]+)$")
_LEGACY_BRIDGE_CONFIG_FIELDS = ("bridgeUrl", "bridgeToken", "bridge_url", "bridge_token")
_REMOTE_MEDIA_MAX_BYTES = 32 * 1024 * 1024
_REMOTE_MEDIA_MAX_REDIRECTS = 5
_REMOTE_MEDIA_TIMEOUT_SECONDS = 120.0
# OGG is intentionally excluded: WhatsApp accepts only mono Opus, which MIME sniffing cannot prove.
_DIRECT_AUDIO_MIMETYPES = {"audio/aac", "audio/amr", "audio/mp4", "audio/mpeg"}
_MIMETYPE_ALIASES = {
    "audio/x-hx-aac-adts": "audio/aac",
    "audio/x-m4a": "audio/mp4",
}


def _default_database_path() -> Path:
    return get_runtime_subdir("whatsapp-auth") / "neonize.db"


def _legacy_bridge_config_fields(config: dict[str, Any]) -> list[str]:
    return [field for field in _LEGACY_BRIDGE_CONFIG_FIELDS if field in config]


def _load_neonize() -> _NeonizeAPI:
    global _NEONIZE_API
    if _NEONIZE_API is not None:
        return _NEONIZE_API

    try:
        import magic
        from neonize.aioze.client import NewAClient
        from neonize.aioze.events import ConnectedEv, DisconnectedEv, MessageEv, PairStatusEv
        from neonize.utils.jid import build_jid

        detect_mime = getattr(magic, "from_file", None)
        detect_buffer = getattr(magic, "from_buffer", None)
        if not callable(detect_mime) or not callable(detect_buffer):
            raise ImportError("python-magic does not expose from_file/from_buffer")
    except ImportError as exc:
        raise RuntimeError(
            "WhatsApp dependencies not installed. Run: nanobot plugins enable whatsapp"
        ) from exc

    _NEONIZE_API = _NeonizeAPI(
        NewAClient=NewAClient,
        ConnectedEv=ConnectedEv,
        DisconnectedEv=DisconnectedEv,
        MessageEv=MessageEv,
        PairStatusEv=PairStatusEv,
        build_jid=build_jid,
        detect_mime=detect_mime,
        detect_buffer=detect_buffer,
    )
    return _NEONIZE_API


def _has_field(message: Any, name: str) -> bool:
    if message is None:
        return False

    has_field = getattr(message, "HasField", None)
    if callable(has_field):
        try:
            return bool(has_field(name))
        except ValueError:
            pass

    list_fields = getattr(message, "ListFields", None)
    if callable(list_fields):
        try:
            fields = cast(list[tuple[Any, Any]], list_fields())
            return any(getattr(field, "name", "") == name for field, _ in fields)
        except Exception:
            pass

    value = getattr(message, name, None)
    return value is not None and value != "" and value != b""


def _message_field(message: Any, *names: str) -> Any:
    for name in names:
        if _has_field(message, name):
            return getattr(message, name)
    return None


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _jid_to_string(jid: Any) -> str:
    if jid is None:
        return ""
    if isinstance(jid, str):
        return jid.strip()
    if bool(_safe_attr(jid, "IsEmpty", False)):
        return ""

    user = str(_safe_attr(jid, "User", "") or "").strip()
    server = str(_safe_attr(jid, "Server", "") or "").strip()
    if user and server:
        return f"{user}@{server}"
    return server or user


def _normalize_jid(raw: Any) -> str:
    jid = _jid_to_string(raw).strip()
    if not jid:
        return ""
    if jid.endswith("@lid.whatsapp.net"):
        return jid[: -len(".whatsapp.net")]
    return jid


def _bare_jid(raw: Any) -> str:
    jid = _normalize_jid(raw)
    if "@" not in jid:
        return jid
    return jid.split("@", 1)[0].split(":", 1)[0]


def _classify_sender_ids(jids: list[Any]) -> tuple[str, str]:
    phone_id = ""
    lid_id = ""

    for raw in jids:
        jid = _normalize_jid(raw)
        if not jid:
            continue
        match = _JID_RE.match(jid)
        if match:
            user = match.group("user").split(":", 1)[0]
            server = match.group("server")
            if server in {"s.whatsapp.net", "c.us"}:
                phone_id = phone_id or user
            elif server in {"lid", "lid.whatsapp.net"}:
                lid_id = lid_id or user
            continue

        if not phone_id:
            phone_id = jid

    return phone_id, lid_id


def _context_infos(message: Any) -> list[Any]:
    infos: list[Any] = []
    for container in (
        message,
        _message_field(message, "extendedTextMessage"),
        _message_field(message, "imageMessage"),
        _message_field(message, "videoMessage"),
        _message_field(message, "audioMessage"),
        _message_field(message, "documentMessage"),
        _message_field(message, "stickerMessage"),
    ):
        context = _message_field(container, "contextInfo")
        if context is not None:
            infos.append(context)
    return infos


def _message_text(message: Any) -> str:
    conversation = str(_safe_attr(message, "conversation", "") or "").strip()
    if conversation:
        return conversation

    extended = _message_field(message, "extendedTextMessage")
    text = str(_safe_attr(extended, "text", "") or "").strip()
    if text:
        return text

    for field_name in ("imageMessage", "videoMessage", "documentMessage", "stickerMessage"):
        media_message = _message_field(message, field_name)
        caption = str(_safe_attr(media_message, "caption", "") or "").strip()
        if caption:
            return caption

    return ""


def _media_message(message: Any) -> _MediaInfo | None:
    image = _message_field(message, "imageMessage")
    if image is not None:
        return _MediaInfo(
            kind="image",
            message=image,
            mimetype=str(_safe_attr(image, "mimetype", "") or "image/jpeg"),
            filename=str(_safe_attr(image, "fileName", "") or ""),
        )

    video = _message_field(message, "videoMessage")
    if video is not None:
        return _MediaInfo(
            kind="video",
            message=video,
            mimetype=str(_safe_attr(video, "mimetype", "") or "video/mp4"),
            filename=str(_safe_attr(video, "fileName", "") or ""),
        )

    audio = _message_field(message, "audioMessage")
    if audio is not None:
        return _MediaInfo(
            kind="audio",
            message=audio,
            mimetype=str(_safe_attr(audio, "mimetype", "") or "audio/ogg"),
            filename=str(_safe_attr(audio, "fileName", "") or ""),
            is_voice=bool(_safe_attr(audio, "PTT", False) or _safe_attr(audio, "ptt", False)),
        )

    document = _message_field(message, "documentMessage")
    if document is not None:
        return _MediaInfo(
            kind="file",
            message=document,
            mimetype=str(_safe_attr(document, "mimetype", "") or "application/octet-stream"),
            filename=str(
                _safe_attr(document, "fileName", "")
                or _safe_attr(document, "title", "")
                or ""
            ),
        )

    sticker = _message_field(message, "stickerMessage")
    if sticker is not None:
        return _MediaInfo(
            kind="sticker",
            message=sticker,
            mimetype=str(_safe_attr(sticker, "mimetype", "") or "image/webp"),
            filename=str(_safe_attr(sticker, "fileName", "") or ""),
        )

    return None


class WhatsAppChannel(BaseChannel):
    """WhatsApp channel using neonize's async WhatsApp client."""

    name = "whatsapp"
    display_name = "WhatsApp"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WhatsAppConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        legacy_bridge_fields = (
            _legacy_bridge_config_fields(cast(dict[str, Any], config))
            if isinstance(config, dict) else []
        )
        if isinstance(config, dict):
            config = WhatsAppConfig.model_validate(config)
        super().__init__(config, bus)
        if legacy_bridge_fields:
            self.logger.warning(
                "Ignoring deprecated WhatsApp bridge config fields: {}. "
                "Run 'nanobot channels login whatsapp' to create a neonize session.",
                ", ".join(legacy_bridge_fields),
            )
        self._client: Any | None = None
        self._connected = False
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()
        self._lid_to_phone = self._load_lid_mappings()
        self._self_jids: set[str] = set()
        self._started_at = 0.0
        self._config_mtime: float | None = None

    def _check_and_reload_config(self) -> None:
        """Reload configuration from disk if config.json was modified."""
        try:
            from nanobot.config.loader import get_config_path, load_config
            cfg_path = get_config_path()
            if not cfg_path.exists():
                return
            mtime = cfg_path.stat().st_mtime
            if self._config_mtime is not None and mtime != self._config_mtime:
                full_config = load_config()
                raw_channels = getattr(full_config, "channels", None)
                raw_whatsapp: Any = None
                if isinstance(raw_channels, dict):
                    channels_dict = cast(dict[str, Any], raw_channels)
                    raw_whatsapp = channels_dict.get("whatsapp")
                elif raw_channels is not None:
                    raw_whatsapp = getattr(raw_channels, "whatsapp", None)

                if isinstance(raw_whatsapp, dict):
                    self.config = WhatsAppConfig.model_validate(raw_whatsapp)
                elif isinstance(raw_whatsapp, WhatsAppConfig):
                    self.config = raw_whatsapp
                elif isinstance(raw_whatsapp, object) and hasattr(raw_whatsapp, "model_dump"):
                    dump_fn: Callable[..., dict[str, Any]] | None = getattr(raw_whatsapp, "model_dump", None)
                    if callable(dump_fn):
                        self.config = WhatsAppConfig.model_validate(dump_fn(by_alias=True))
                self.logger.info("[WhatsApp] Hot-reloaded configuration from disk")
            self._config_mtime = mtime
        except Exception as e:
            self.logger.debug("Config reload check skipped: {}", e)

    def _database_path(self) -> Path:
        configured = self.config.database_path.strip()
        return Path(configured).expanduser() if configured else _default_database_path()

    def _load_lid_mappings(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for lid, phone in self.config.lid_mappings.items():
            phone_text = str(phone).strip()
            if phone_text:
                mapping[str(lid).strip()] = phone_text
        return mapping

    def _new_client(self) -> Any:
        api = _load_neonize()
        db_path = self._database_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return api.NewAClient(str(db_path))

    async def login(self, force: bool = False) -> bool:
        db_path = self._database_path()
        if force:
            self._reset_database(db_path)

        client = self._new_client()
        login_result = asyncio.get_running_loop().create_future()
        self._register_handlers(client, login_result=login_result, handle_messages=False)

        try:
            self.logger.info("Starting WhatsApp login with neonize...")
            connect_task = await client.connect()
            self._fail_login_on_connect_task_done(connect_task, login_result)
            await login_result
            self.logger.info("WhatsApp login complete")
            return True
        except Exception as exc:
            self.logger.error("WhatsApp login failed: {}", exc)
            return False
        finally:
            with suppress(Exception):
                await client.stop()

    async def start(self) -> None:
        self._running = True
        self._started_at = time.time()
        client = self._new_client()
        self._client = client
        self._register_handlers(client, handle_messages=True)

        try:
            self.logger.info("Connecting WhatsApp channel with neonize...")
            await client.connect()
            await client.idle()
        except asyncio.CancelledError:
            raise
        finally:
            self._running = False
            self._connected = False
            if self._client is client:
                self._client = None
            with suppress(Exception):
                await client.stop()

    async def stop(self) -> None:
        self._running = False
        self._connected = False
        client = self._client
        self._client = None
        if client is not None:
            await client.stop()

    @staticmethod
    def _fail_login_on_connect_task_done(
        connect_task: asyncio.Task[Any] | None,
        login_result: asyncio.Future[None],
    ) -> None:
        if connect_task is None:
            return

        def _on_done(task: asyncio.Task[Any]) -> None:
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                return
            if login_result.done():
                return
            if exc is not None:
                login_result.set_exception(exc)
            else:
                login_result.set_exception(
                    RuntimeError("WhatsApp connection ended before login completed")
                )

        connect_task.add_done_callback(_on_done)

    async def send(self, msg: OutboundMessage) -> None:
        client = self._client
        if client is None or not self._connected:
            raise RuntimeError("WhatsApp channel is not connected")

        to = self._build_jid(msg.chat_id)

        # Check for silent ack protocol ([REACTION: <emoji>] or [SILENT])
        silent_signal = parse_silent_ack(msg.content)
        if silent_signal.action != "text":
            with suppress(Exception):
                await self._send_chat_presence(to, composing=False)

            if silent_signal.action == "reaction" and silent_signal.emoji:
                target_msg_id = msg.metadata.get("message_id")
                if target_msg_id:
                    with suppress(Exception):
                        send_reaction: Any = getattr(client, "send_reaction", None)
                        if callable(send_reaction):
                            reaction_call = send_reaction(to, str(target_msg_id), silent_signal.emoji)
                            if asyncio.iscoroutine(reaction_call):
                                await reaction_call
            self.logger.info(
                "[Silent Ack] {} handled for WhatsApp chat {} (trigger msg_id: {})",
                silent_signal.action.upper(),
                msg.chat_id,
                msg.metadata.get("message_id"),
            )
            return

        if msg.content:
            await client.send_message(to, msg.content)

        for media_path in msg.media or []:
            await self._send_media(client, to, media_path)

    def _build_jid(self, raw: str) -> Any:
        api = _load_neonize()
        target = raw.strip()
        match = _JID_RE.match(_normalize_jid(target))
        if not match:
            return api.build_jid(target)

        user = match.group("user").split(":", 1)[0]
        server = match.group("server")
        return api.build_jid(user, server)

    async def _send_media(self, client: Any, to: Any, media_path: str) -> None:
        source: str | bytes
        if media_path.startswith(("http://", "https://")):
            source = await self._fetch_remote_media(media_path)
            filename = Path(urlparse(media_path).path).name or "attachment"
        else:
            source = str(Path(media_path).expanduser())
            filename = Path(source).name

        mimetype = self._detect_mimetype(source)
        if mimetype.startswith("image/"):
            await client.send_image(to, source)
        elif mimetype.startswith("video/"):
            await client.send_video(to, source)
        elif mimetype in _DIRECT_AUDIO_MIMETYPES:
            await client.send_audio(to, source)
        else:
            await client.send_document(
                to,
                source,
                filename=filename,
                mimetype=mimetype,
            )

    async def _fetch_remote_media(self, url: str) -> bytes:
        timeout = httpx.Timeout(_REMOTE_MEDIA_TIMEOUT_SECONDS, connect=10.0)
        async with httpx.AsyncClient(
            transport=PinnedDNSAsyncTransport(),
            follow_redirects=True,
            max_redirects=_REMOTE_MEDIA_MAX_REDIRECTS,
            timeout=timeout,
            trust_env=False,
        ) as http:
            async with http.stream("GET", url) as response:
                response.raise_for_status()
                declared_size = response.headers.get("content-length")
                if (
                    declared_size
                    and declared_size.isdigit()
                    and int(declared_size) > _REMOTE_MEDIA_MAX_BYTES
                ):
                    raise ValueError(
                        f"Remote WhatsApp media exceeds the {_REMOTE_MEDIA_MAX_BYTES}-byte limit"
                    )

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > _REMOTE_MEDIA_MAX_BYTES:
                        raise ValueError(
                            f"Remote WhatsApp media exceeds the {_REMOTE_MEDIA_MAX_BYTES}-byte limit"
                        )
                    chunks.append(chunk)
        return b"".join(chunks)

    def _detect_mimetype(self, source: str | bytes) -> str:
        try:
            api = _load_neonize()
            detected = (
                api.detect_buffer(source, mime=True)
                if isinstance(source, bytes)
                else api.detect_mime(source, mime=True)
            )
        except Exception as exc:
            label = f"{len(source)} downloaded bytes" if isinstance(source, bytes) else source
            self.logger.debug("Failed to inspect WhatsApp media {}: {}", label, exc)
            detected = None

        if isinstance(detected, str) and "/" in detected:
            mimetype = detected.partition(";")[0].strip().lower()
            return _MIMETYPE_ALIASES.get(mimetype, mimetype)

        if isinstance(source, bytes):
            return "application/octet-stream"

        guessed, _ = mimetypes.guess_type(source)
        return guessed or "application/octet-stream"

    def _register_handlers(
        self,
        client: Any,
        *,
        login_result: asyncio.Future[None] | None = None,
        handle_messages: bool,
    ) -> None:
        api = _load_neonize()

        @client.qr
        async def _on_qr(_: Any, qr_data: bytes) -> None:
            import segno

            self.logger.info("Scan the WhatsApp QR code with Linked Devices")
            segno.make_qr(qr_data).terminal(compact=True)

        @client.event(api.ConnectedEv)
        async def _on_connected(current_client: Any, _: Any) -> None:
            self._connected = True
            try:
                await self._remember_self_jids(current_client)
            except Exception as exc:
                if login_result is not None and not login_result.done():
                    login_result.set_exception(exc)
                raise
            if login_result is not None and not login_result.done():
                login_result.set_result(None)
            self.logger.info("WhatsApp connected")

        @client.event(api.DisconnectedEv)
        async def _on_disconnected(_: Any, event: Any) -> None:
            self._connected = False
            if login_result is not None and not login_result.done():
                login_result.set_exception(
                    RuntimeError(f"WhatsApp disconnected before login completed: {event}")
                )
            self.logger.warning("WhatsApp disconnected: {}", event)

        @client.event(api.PairStatusEv)
        async def _on_pair_status(_: Any, event: Any) -> None:
            error = str(_safe_attr(event, "Error", "") or "")
            if error:
                exc = RuntimeError(f"WhatsApp pair status error: {error}")
                if login_result is not None and not login_result.done():
                    login_result.set_exception(exc)
                raise exc
            self.logger.info("WhatsApp pair status: {}", event)

        if not handle_messages:
            return

        @client.event(api.MessageEv)
        async def _on_message(current_client: Any, event: Any) -> None:
            try:
                await self._handle_neonize_message(current_client, event)
            except Exception:
                self.logger.exception("Error handling WhatsApp message")
                raise

    async def _remember_self_jids(self, client: Any) -> None:
        device = _safe_attr(client, "me")
        if device is None:
            device = await client.get_me()

        for attr in ("JID", "LID"):
            jid = _normalize_jid(_safe_attr(device, attr))
            if jid:
                self._self_jids.add(jid)
                self._self_jids.add(_bare_jid(jid))

    async def _send_read_receipt(self, client: Any, source: Any, message_id: str) -> None:
        """Send a read receipt (blue double-check) for an incoming message.

        Best-effort: any failure is logged at debug level and swallowed so it
        never blocks message processing.
        """
        if not message_id:
            return
        try:
            from neonize.utils.enum import ReceiptType

            chat = _safe_attr(source, "Chat")
            sender = _safe_attr(source, "Sender")
            if chat is None or sender is None:
                return
            await client.mark_read(
                message_id,
                chat=chat,
                sender=sender,
                receipt=ReceiptType.READ,
            )
        except Exception as exc:  # noqa: BLE001 - read receipt is best-effort
            self.logger.debug("Failed to send WhatsApp read receipt: {}", exc)

    async def _handle_neonize_message(self, client: Any, event: Any) -> None:
        info = _safe_attr(event, "Info")
        message = _safe_attr(event, "Message")
        source = _safe_attr(info, "MessageSource")
        if info is None or message is None or source is None:
            raise ValueError("WhatsApp MessageEv is missing Info, Message, or MessageSource")

        if bool(_safe_attr(source, "IsFromMe", False)):
            return

        chat_jid = _normalize_jid(_safe_attr(source, "Chat"))
        if not chat_jid:
            raise ValueError("WhatsApp message has no chat JID")
        if chat_jid == "status@broadcast":
            return

        timestamp = float(_safe_attr(info, "Timestamp", 0) or 0)
        if self._started_at and timestamp and timestamp < self._started_at:
            return

        is_group = bool(_safe_attr(source, "IsGroup", False))
        if is_group and self.config.group_policy == "mention":
            if not self._is_addressed_to_bot(message):
                return

        message_id = str(_safe_attr(info, "ID", "") or "")
        if message_id:
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None
            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)

        # Mark the incoming message as read (blue double-check). Best-effort.
        await self._send_read_receipt(client, source, message_id)

        participant_jid = _normalize_jid(_safe_attr(source, "Sender"))
        sender_alt_jid = _normalize_jid(_safe_attr(source, "SenderAlt"))
        sender_candidates = [sender_alt_jid, participant_jid]
        if not is_group:
            sender_candidates.append(chat_jid)

        phone_id, lid_id = _classify_sender_ids(sender_candidates)
        if phone_id and lid_id:
            self._lid_to_phone[lid_id] = phone_id

        sender_id = phone_id or self._lid_to_phone.get(lid_id, "") or lid_id
        if not sender_id:
            raise ValueError("WhatsApp message has no resolvable sender ID")
        metadata = {
            "message_id": message_id or None,
            "timestamp": int(timestamp) if timestamp else None,
            "is_group": is_group,
            "is_forwarded": self._is_forwarded(message),
            "participant": participant_jid or None,
            "sender_alt": sender_alt_jid or None,
            "lid": lid_id or None,
            "phone": phone_id or None,
            "is_reply_to_bot": self._is_reply_to_bot(message),
        }
        self._check_and_reload_config()
        routing_cfg = self.config.routing
        is_staff = self._is_staff_sender(
            sender_id=sender_id,
            chat_jid=chat_jid,
            participant_jid=participant_jid,
            sender_alt_jid=sender_alt_jid,
            lid_id=lid_id,
            phone_id=phone_id,
        )

        if not is_staff:
            text, _ = await self._extract_message_content_and_media(client, event, message)
            if text:
                await self._forward_to_guide_instance(
                    chat_jid=chat_jid,
                    sender_id=sender_id,
                    text=text,
                    routing_cfg=routing_cfg,
                    client_instance=client,
                )
            return

        sender_allowed = self.is_allowed(sender_id)
        group_allow_id = self._group_allow_id(chat_jid) if is_group else None
        authorization_id = sender_id if sender_allowed else group_allow_id
        if authorization_id is None:
            self.logger.info(
                "Passing unauthorized WhatsApp sender {} to pairing flow "
                "(phone={}, lid={}, chat={})",
                sender_id,
                phone_id or "",
                lid_id or "",
                chat_jid,
            )
            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_jid,
                content=_message_text(message),
                media=[],
                metadata=metadata,
                is_dm=not is_group,
            )
            return

        text, media_paths = await self._extract_message_content_and_media(client, event, message)
        if not text and not media_paths:
            return

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_jid,
            content=text,
            media=media_paths,
            metadata=metadata,
            is_dm=not is_group,
            authorization_id=authorization_id,
        )

    async def _extract_message_content_and_media(
        self,
        client: Any,
        event: Any,
        message: Any,
    ) -> tuple[str, list[str]]:
        text = _message_text(message)
        media_paths: list[str] = []
        media = _media_message(message)
        if media is not None:
            path = await self._download_media(client, event, media)
            if media.kind == "audio" and media.is_voice:
                transcription = await self.transcribe_audio(path)
                if transcription:
                    text = transcription
                else:
                    media_paths.append(path)
                    text = self._append_media_tag(text, "audio", path)
            else:
                media_paths.append(path)
                text = self._append_media_tag(text, media.kind, path)

        return text, media_paths



    def _is_staff_sender(
        self,
        sender_id: str,
        chat_jid: str,
        participant_jid: str | None = None,
        sender_alt_jid: str | None = None,
        lid_id: str | None = None,
        phone_id: str | None = None,
    ) -> bool:
        routing_cfg = self.config.routing
        if not routing_cfg.enabled:
            return True

        if not routing_cfg.staff_numbers and not routing_cfg.staff_groups:
            return True


        def _clean_number(val: str) -> str:
            v = val.strip()
            if "@" in v:
                v = v.split("@", 1)[0]
            if ":" in v:
                v = v.split(":", 1)[0]
            if v.startswith("+"):
                v = v[1:]
            return v

        staff_nums = {_clean_number(n) for n in routing_cfg.staff_numbers if n.strip()}
        staff_grps = {_clean_number(g) for g in routing_cfg.staff_groups if g.strip()}

        sender_candidates = {
            _clean_number(sender_id),
            _clean_number(chat_jid),
        }
        if participant_jid:
            sender_candidates.add(_clean_number(participant_jid))
        if sender_alt_jid:
            sender_candidates.add(_clean_number(sender_alt_jid))
        if lid_id:
            sender_candidates.add(_clean_number(lid_id))
        if phone_id:
            sender_candidates.add(_clean_number(phone_id))

        if staff_nums and any(cand in staff_nums for cand in sender_candidates if cand):
            return True

        chat_candidates = {_clean_number(chat_jid), chat_jid.strip()}
        if staff_grps and any(cand in staff_grps for cand in chat_candidates if cand):
            return True

        return False

    async def _send_chat_presence(
        self,
        chat_jid: str,
        composing: bool,
        client: Any | None = None,
    ) -> None:
        """Send chat presence indicator (composing/paused) if supported by client."""
        cli = client if (client is not None and (hasattr(client, "send_chat_presence") or hasattr(client, "send_presence"))) else self._client
        if cli is None:
            return
        try:
            presence_fn = getattr(cli, "send_chat_presence", None) or getattr(cli, "send_presence", None)
            if callable(presence_fn):
                presence_state = "composing" if composing else "paused"
                res = presence_fn(chat_jid, presence_state, "")
                if asyncio.iscoroutine(res):
                    await res
        except Exception as e:
            self.logger.debug("Failed to set chat presence for {}: {}", chat_jid, e)

    async def _forward_to_guide_instance(
        self,
        chat_jid: str,
        sender_id: str,
        text: str,
        routing_cfg: WhatsAppRoutingConfig,
        client_instance: Any | None = None,
    ) -> None:
        payload = {
            "model": routing_cfg.guide_model_name,
            "messages": [{"role": "user", "content": text}],
            "user": f"whatsapp:{sender_id}",
            "session_id": f"whatsapp:{chat_jid}",
            "stream": True,
        }

        await self._send_chat_presence(chat_jid, composing=True, client=client_instance)
        try:
            async with httpx.AsyncClient(timeout=routing_cfg.forward_timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    routing_cfg.guide_instance_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    reply_text = ""

                    if "text/event-stream" in content_type:
                        chunks: list[str] = []
                        last_presence_refresh = time.monotonic()
                        async for line in response.aiter_lines():
                            line_str = line.strip()
                            if not line_str.startswith("data:"):
                                continue
                            data_str = line_str[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                sse_data_obj = json.loads(data_str)
                                if isinstance(sse_data_obj, dict):
                                    sse_data: dict[str, Any] = cast(dict[str, Any], sse_data_obj)
                                    raw_sse_choices = sse_data.get("choices")
                                    if isinstance(raw_sse_choices, list) and raw_sse_choices:
                                        sse_choices: list[Any] = cast(list[Any], raw_sse_choices)
                                        sse_first_obj: Any = sse_choices[0]
                                        if isinstance(sse_first_obj, dict):
                                            sse_first: dict[str, Any] = cast(dict[str, Any], sse_first_obj)
                                            delta_obj = sse_first.get("delta")
                                            if isinstance(delta_obj, dict):
                                                delta: dict[str, Any] = cast(dict[str, Any], delta_obj)
                                                content = delta.get("content")
                                                if isinstance(content, str) and content:
                                                    chunks.append(content)
                            except Exception:
                                pass

                            now = time.monotonic()
                            if now - last_presence_refresh > 5.0:
                                await self._send_chat_presence(chat_jid, composing=True, client=client_instance)
                                last_presence_refresh = now

                        reply_text = "".join(chunks).strip()
                    else:
                        body_bytes = await response.aread()
                        body_data_obj = json.loads(body_bytes.decode("utf-8"))
                        if isinstance(body_data_obj, dict):
                            body_data: dict[str, Any] = cast(dict[str, Any], body_data_obj)
                            raw_body_choices = body_data.get("choices")
                            if isinstance(raw_body_choices, list) and raw_body_choices:
                                body_choices: list[Any] = cast(list[Any], raw_body_choices)
                                body_first_obj: Any = body_choices[0]
                                if isinstance(body_first_obj, dict):
                                    body_first: dict[str, Any] = cast(dict[str, Any], body_first_obj)
                                    body_message_obj = body_first.get("message")
                                    if isinstance(body_message_obj, dict):
                                        body_message: dict[str, Any] = cast(dict[str, Any], body_message_obj)
                                        reply_text = str(body_message.get("content") or "").strip()

                    if reply_text:
                        outbound = OutboundMessage(channel=self.name, chat_id=chat_jid, content=reply_text)
                        await self.send(outbound)
                    else:
                        self.logger.warning(
                            "[WhatsApp Routing] Empty or non-standard response from Guide instance",
                        )

        except httpx.ConnectError:
            self.logger.error(
                "[WhatsApp Routing] Guide instance at {} unreachable.",
                routing_cfg.guide_instance_url,
            )
            outbound = OutboundMessage(
                channel=self.name,
                chat_id=chat_jid,
                content="The studio assistant is briefly performing maintenance. Please try again in a few moments!",
            )
            await self.send(outbound)
        except httpx.TimeoutException:
            self.logger.warning(
                "[WhatsApp Routing] Timeout waiting for Guide response for chat {}.",
                chat_jid,
            )
            outbound = OutboundMessage(
                channel=self.name,
                chat_id=chat_jid,
                content="I am looking into that for you. One moment please...",
            )
            await self.send(outbound)
        except Exception as e:
            self.logger.error(
                "[WhatsApp Routing] Unexpected error during forward: {}",
                e,
                exc_info=True,
            )
        finally:
            await self._send_chat_presence(chat_jid, composing=False, client=client_instance)


    def _group_allow_id(self, chat_jid: str) -> str | None:
        if self.is_allowed(chat_jid):
            return chat_jid
        bare_chat_id = _bare_jid(chat_jid)
        if bare_chat_id and bare_chat_id != chat_jid and self.is_allowed(bare_chat_id):
            return bare_chat_id
        return None

    def _is_addressed_to_bot(self, message: Any) -> bool:
        return self._was_mentioned(message) or self._is_reply_to_bot(message)

    def _was_mentioned(self, message: Any) -> bool:
        if not self._self_jids:
            return False
        for context in _context_infos(message):
            raw_mentioned: Any = (
                _safe_attr(context, "mentionedJID")
                or _safe_attr(context, "mentionedJid")
                or _safe_attr(context, "mentioned_jid")
                or []
            )
            mentioned: list[Any] = cast(list[Any], raw_mentioned)
            for jid in mentioned:
                normalized = _normalize_jid(jid)
                if normalized in self._self_jids or _bare_jid(normalized) in self._self_jids:
                    return True
        return False

    def _is_reply_to_bot(self, message: Any) -> bool:
        if not self._self_jids:
            return False
        for context in _context_infos(message):
            participant = _normalize_jid(
                _safe_attr(context, "participant")
                or _safe_attr(context, "Participant")
                or ""
            )
            if participant in self._self_jids or _bare_jid(participant) in self._self_jids:
                return True
        return False

    @staticmethod
    def _is_forwarded(message: Any) -> bool:
        for context in _context_infos(message):
            if bool(_safe_attr(context, "isForwarded", False)):
                return True
            if int(_safe_attr(context, "forwardingScore", 0) or 0) > 0:
                return True
        return False

    async def _download_media(self, client: Any, event: Any, media: _MediaInfo) -> str:
        info = _safe_attr(event, "Info")
        message_id = str(_safe_attr(info, "ID", "") or "")
        path = self._media_path(message_id, media)
        await client.download_any(_safe_attr(event, "Message"), str(path))
        return str(path)

    def _media_path(self, message_id: str, media: _MediaInfo) -> Path:
        media_dir = get_media_dir("whatsapp")
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", message_id or str(int(time.time())))
        filename = Path(media.filename).name if media.filename else ""
        suffix = Path(filename).suffix if filename else ""
        if not suffix:
            suffix = mimetypes.guess_extension(media.mimetype) or {
                "image": ".jpg",
                "video": ".mp4",
                "audio": ".ogg",
                "sticker": ".webp",
            }.get(media.kind, ".bin")
        return media_dir / f"wa_{safe_id}_{secrets.token_hex(4)}{suffix}"

    @staticmethod
    def _append_media_tag(text: str, kind: str, path: str) -> str:
        label = kind if kind in {"image", "video", "audio", "sticker"} else "file"
        tag = f"[{label}: {path}]"
        return f"{text}\n{tag}" if text else tag

    @staticmethod
    def _reset_database(path: Path) -> None:
        for candidate in (
            path,
            path.with_suffix(path.suffix + "-shm"),
            path.with_suffix(path.suffix + "-wal"),
        ):
            if candidate.exists():
                candidate.unlink()

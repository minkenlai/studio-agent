"""Tests for SpeakTool and TTSService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.tools.speak import SpeakTool
from nanobot.bus.events import Address
from nanobot.services.tts_service import TTSService

# ---------------------------------------------------------------------------
# TTSService._clean_text  unit tests
# ---------------------------------------------------------------------------


class TestTTSServiceCleanText:
    """Verify markdown-stripping logic produces speech-friendly text."""

    @pytest.fixture
    def service(self):
        return TTSService(MagicMock())

    def test_strips_bold(self, service):
        assert service._clean_text("**hello**") == "hello"

    def test_strips_italic_asterisk(self, service):
        assert service._clean_text("*hello*") == "hello"

    def test_strips_italic_underscore(self, service):
        assert service._clean_text("_hello_") == "hello"

    def test_strips_inline_code(self, service):
        assert service._clean_text("`hello`") == "hello"

    def test_strips_strike(self, service):
        assert service._clean_text("~~hello~~") == "hello"

    def test_strips_links(self, service):
        assert service._clean_text("[text](http://example.com)") == "text"

    def test_strips_headings(self, service):
        assert service._clean_text("# Hello\n## World") == "Hello\nWorld"

    def test_strips_unordered_list(self, service):
        assert service._clean_text("- one\n- two") == "one\ntwo"

    def test_strips_ordered_list(self, service):
        assert service._clean_text("1. one\n2. two") == "one\ntwo"

    def test_strips_blockquotes(self, service):
        assert service._clean_text("> hello") == "hello"

    def test_strips_fenced_code_blocks(self, service):
        result = service._clean_text("before\n```\nhello()\n```\nafter")
        assert "hello()" not in result
        assert "before" in result
        assert "after" in result

    def test_normalizes_multiple_newlines(self, service):
        assert service._clean_text("a\n\n\n\nb") == "a\nb"

    def test_strips_and_trims(self, service):
        assert service._clean_text("  **bold**  ") == "bold"

    def test_empty_after_cleanup(self, service):
        # "** **  " → bold strips to "  " → strip() → ""
        assert service._clean_text("** **  ") == ""

    def test_complex_mixed_markdown(self, service):
        raw = "# Title\n\n**bold** and `code` and [link](http://x)\n> quote\n- item"
        result = service._clean_text(raw)
        assert "#" not in result
        assert "**" not in result
        assert "`" not in result
        assert "http://x" not in result
        assert "Title" in result
        assert "bold" in result
        assert "code" in result
        assert "link" in result


# ---------------------------------------------------------------------------
# TTSService.speak  integration-style tests (mocked TTS client)
# ---------------------------------------------------------------------------


class TestTTSServiceSpeak:
    """Test TTSService.speak() end-to-end with mocked loop/client."""

    @pytest.fixture
    def mock_loop(self):
        loop = MagicMock()
        tts_client = MagicMock()
        tts_client.output_ext = ".wav"
        tts_client.synthesize = AsyncMock(return_value=True)
        loop.get_tts_client = AsyncMock(return_value=tts_client)
        return loop

    @pytest.fixture
    def service(self, mock_loop):
        return TTSService(mock_loop)

    @pytest.mark.asyncio
    async def test_speak_with_text(self, service, mock_loop):
        """Providing text triggers synthesis directly."""
        with patch("nanobot.services.tts_service.get_media_dir") as mock_media:
            mock_dir = MagicMock()
            mock_dir.__truediv__ = MagicMock(return_value=Path("/tmp/tts_abc.wav"))
            mock_media.return_value = mock_dir

            ok, path = await service.speak(text="Hello", session_key="sk")

        assert ok is True
        assert "tts_" in path
        mock_loop.get_tts_client.return_value.synthesize.assert_called_once()
        call_text = mock_loop.get_tts_client.return_value.synthesize.call_args[0][0]
        assert call_text == "Hello"

    @pytest.mark.asyncio
    async def test_speak_no_text_fetches_last_assistant(self, mock_loop):
        """No text → fetch last assistant message from session."""
        session = MagicMock()
        session.messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hey there!"},
        ]
        mock_loop.sessions.get_or_create.return_value = session

        service = TTSService(mock_loop)

        with patch("nanobot.services.tts_service.get_media_dir") as mock_media:
            mock_dir = MagicMock()
            mock_dir.__truediv__ = MagicMock(return_value=Path("/tmp/tts_def.wav"))
            mock_media.return_value = mock_dir

            ok, path = await service.speak(text=None, session_key="sk")

        assert ok is True
        call_text = mock_loop.get_tts_client.return_value.synthesize.call_args[0][0]
        assert call_text == "Hey there!"

    @pytest.mark.asyncio
    async def test_speak_no_text_no_session_key(self, service):
        """No text + no session key → error."""
        ok, msg = await service.speak(text=None)
        assert ok is False
        assert "No session key" in msg

    @pytest.mark.asyncio
    async def test_speak_no_text_no_assistant_msg(self, mock_loop):
        """No text + session has no assistant messages → error."""
        session = MagicMock()
        session.messages = [{"role": "user", "content": "hello?"}]
        mock_loop.sessions.get_or_create.return_value = session

        service = TTSService(mock_loop)
        ok, msg = await service.speak(text=None, session_key="sk")

        assert ok is False
        assert "No prior agent message" in msg

    @pytest.mark.asyncio
    async def test_speak_empty_text_after_cleanup(self, mock_loop):
        """Text that becomes empty after cleanup → error."""
        session = MagicMock()
        # "** **  " → bold strips to "  " → strip() → ""
        session.messages = [{"role": "assistant", "content": "** **  "}]
        mock_loop.sessions.get_or_create.return_value = session

        service = TTSService(mock_loop)
        ok, msg = await service.speak(text=None, session_key="sk")

        assert ok is False
        assert "empty after cleanup" in msg

    @pytest.mark.asyncio
    async def test_speak_no_tts_client(self, mock_loop, service):
        """TTS client is None → error."""
        mock_loop.get_tts_client = AsyncMock(return_value=None)
        ok, msg = await service.speak(text="hello")
        assert ok is False
        assert "not configured" in msg

    @pytest.mark.asyncio
    async def test_speak_synthesis_failure(self, mock_loop, service):
        """TTS synthesize returns False → error."""
        mock_loop.get_tts_client.return_value.synthesize = AsyncMock(return_value=False)

        with patch("nanobot.services.tts_service.get_media_dir") as mock_media:
            mock_dir = MagicMock()
            mock_dir.__truediv__ = MagicMock(return_value=Path("/tmp/tts_fail.wav"))
            mock_media.return_value = mock_dir

            ok, msg = await service.speak(text="hello")

        assert ok is False
        assert "failed" in msg

    @pytest.mark.asyncio
    async def test_speak_custom_output_path(self, mock_loop, service):
        """Custom output_path is used instead of auto-generated."""
        custom_path = "/custom/audio.wav"
        ok, path = await service.speak(text="Hi", output_path=custom_path)

        assert ok is True
        assert path == custom_path
        call_args = mock_loop.get_tts_client.return_value.synthesize.call_args
        assert call_args[0][1] == custom_path


# ---------------------------------------------------------------------------
# SpeakTool  tests
# ---------------------------------------------------------------------------


class TestSpeakTool:
    """Verify SpeakTool wiring and execute() behavior."""

    @pytest.fixture
    def mock_loop(self):
        return MagicMock()

    @pytest.fixture
    def tool(self, mock_loop):
        return SpeakTool(mock_loop)

    def test_name_and_description(self, tool):
        assert tool.name == "speak"
        assert "synthes" in tool.description.lower() or "speech" in tool.description.lower()

    def test_parameters_schema(self, tool):
        schema = tool.parameters
        assert schema["type"] == "object"
        assert "text" in schema["properties"]
        assert "output_path" in schema["properties"]
        assert "required" not in schema or len(schema["required"]) == 0

    def test_text_param_allows_null(self, tool):
        types = tool.parameters["properties"]["text"]["type"]
        assert "null" in types

    def test_set_context(self, tool):
        addr = Address(channel="telegram", segments=("123",))
        tool.set_context(addr, "tk:123")
        assert tool._address == addr
        assert tool._session_key == "tk:123"

    @pytest.mark.asyncio
    async def test_execute_success(self, tool, mock_loop):
        """Successful synthesis returns the file path message."""
        tool.set_context(
            Address(channel="tg", segments=("c1",)),
            "tg:c1",
        )
        tool.tts_service.speak = AsyncMock(return_value=(True, "/tmp/out.wav"))

        result = await tool.execute(text="Hello world")

        assert "successfully" in result.lower() or "saved to" in result.lower()
        assert "/tmp/out.wav" in result
        tool.tts_service.speak.assert_called_once_with(
            text="Hello world",
            output_path=None,
            session_key="tg:c1",
            msg_address=tool._address,
        )

    @pytest.mark.asyncio
    async def test_execute_failure(self, tool):
        """Failed synthesis returns error message."""
        tool.tts_service.speak = AsyncMock(return_value=(False, "TTS not configured"))

        result = await tool.execute(text="hi")

        assert result.startswith("Error:")
        assert "TTS not configured" in result

    @pytest.mark.asyncio
    async def test_execute_no_context(self, tool):
        """Execute without set_context passes None for session/address."""
        tool.tts_service.speak = AsyncMock(return_value=(True, "/tmp/no_ctx.wav"))

        result = await tool.execute(text="test")

        assert "saved to" in result.lower()
        call_kwargs = tool.tts_service.speak.call_args[1]
        assert call_kwargs["session_key"] is None
        assert call_kwargs["msg_address"] is None

    @pytest.mark.asyncio
    async def test_execute_with_output_path(self, tool):
        """Execute with output_path passes it through."""
        tool.tts_service.speak = AsyncMock(return_value=(True, "/custom/out.wav"))

        await tool.execute(text="hi", output_path="/custom/out.wav")

        call_kwargs = tool.tts_service.speak.call_args[1]
        assert call_kwargs["output_path"] == "/custom/out.wav"

    @pytest.mark.asyncio
    async def test_execute_null_text(self, tool):
        """Execute with null text passes None for session fallback."""
        tool.tts_service.speak = AsyncMock(return_value=(True, "/tmp/null_text.wav"))

        await tool.execute(text=None)

        call_kwargs = tool.tts_service.speak.call_args[1]
        assert call_kwargs["text"] is None


# ---------------------------------------------------------------------------
# Edge-case: TTSService with markdown in session messages
# ---------------------------------------------------------------------------


class TestTTSServiceMarkdownFromSession:
    """Ensure markdown from session messages is cleaned before synthesis."""

    @pytest.fixture
    def mock_loop(self):
        loop = MagicMock()
        tts_client = MagicMock()
        tts_client.output_ext = ".wav"
        tts_client.synthesize = AsyncMock(return_value=True)
        loop.get_tts_client = AsyncMock(return_value=tts_client)
        return loop

    @pytest.mark.asyncio
    async def test_markdown_stripped_from_session_msg(self, mock_loop):
        """Assistant message with markdown → cleaned before synthesis."""
        session = MagicMock()
        session.messages = [
            {"role": "user", "content": "explain AI"},
            {"role": "assistant", "content": "**AI** is `artificial` **intelligence**."},
        ]
        mock_loop.sessions.get_or_create.return_value = session

        service = TTSService(mock_loop)

        with patch("nanobot.services.tts_service.get_media_dir") as mock_media:
            mock_dir = MagicMock()
            mock_dir.__truediv__ = MagicMock(return_value=Path("/tmp/tts_md.wav"))
            mock_media.return_value = mock_dir

            ok, _ = await service.speak(text=None, session_key="sk")

        assert ok is True
        call_text = mock_loop.get_tts_client.return_value.synthesize.call_args[0][0]
        assert "**" not in call_text
        assert "`" not in call_text
        assert "AI is artificial intelligence" in call_text

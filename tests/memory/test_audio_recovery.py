import base64

from nanobot.agent.memory import MemoryStore


def test_audio_archive(tmp_path):
    # Setup workspace using pytest tmp_path
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = MemoryStore(workspace)

    # Create dummy audio data
    dummy_audio_bytes = b"this is some fake audio data" * 100
    b64_data = base64.b64encode(dummy_audio_bytes).decode()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is some audio"},
                {
                    "type": "audio",
                    "data": f"data:audio/wav;base64,{b64_data}",
                    "mime_type": "audio/wav",
                },
            ],
        }
    ]

    # Execute the raw archive
    store._raw_archive(messages)

    # Find the recovery file (should be .json now)
    recovery_files = list((workspace / "memory" / "recovery").glob("*.json"))
    assert len(recovery_files) == 1, "Recovery JSON file not created"
    rec_file = recovery_files[0]

    # Read recovery file content
    content = rec_file.read_text()
    assert b64_data not in content, "Base64 data still present in JSON!"
    assert "FILE:asset_" in content, "Pointer to asset not found in JSON"

    # Check if asset file exists
    assets_dir = workspace / "memory" / "recovery" / "assets"
    assets = list(assets_dir.glob("*.bin"))
    assert len(assets) == 1, "Asset binary file not created"
    assert assets[0].read_bytes() == dummy_audio_bytes, "Asset data mismatch!"


def test_raw_archive_without_audio_creates_no_assets_dir(tmp_path):
    """Lazy assets/ creation: no audio blocks means no empty assets/ subdir."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = MemoryStore(workspace)

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "plain text only"}],
        }
    ]

    store._raw_archive(messages)

    recovery_dir = workspace / "memory" / "recovery"
    recovery_files = list(recovery_dir.glob("*.json"))
    assert len(recovery_files) == 1, "Recovery JSON file not created"
    assert not (recovery_dir / "assets").exists(), "Empty assets/ dir should not be created"

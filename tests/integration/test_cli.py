import pytest

from rag import cli
from rag.core.config import get_settings
from tests.conftest import VIDEO_ID

@pytest.fixture
def cli_env(tmp_path, monkeypatch, make_pipeline):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    pipeline, _, _ = make_pipeline()
    
    from rag.store.artifacts import ArtifactStore

    pipeline.artifacts = ArtifactStore(tmp_path / "data")
    monkeypatch.setattr(cli, "build_pipeline", lambda settings: pipeline)
    yield
    get_settings.cache_clear()


def test_process_then_show(cli_env, capsys):
    assert cli.main(["process", f"https://youtu.be/{VIDEO_ID}"]) == 0
    out = capsys.readouterr()
    assert "segments:" in out.out and "document.json" in out.out
    assert "fetching" in out.err

    assert cli.main(["show", VIDEO_ID, "--limit", "3"]) == 0
    out = capsys.readouterr().out
    assert "[S0] 0:00-" in out and "more" in out


def test_show_missing_document(cli_env, capsys):
    assert cli.main(["show", "aaaaaaaaaaa"]) == 1
    assert "No document" in capsys.readouterr().err


def test_bad_url_exits_with_message(cli_env, capsys):
    assert cli.main(["process", "https://vimeo.com/1"]) == 2
    assert "Not a YouTube URL" in capsys.readouterr().err

from pathlib import Path

from dungeon_apprentice.dashboard import DASHBOARD_HTML, prepare_dashboard


def test_dashboard_uses_v01_trained_and_collected_counters() -> None:
    assert "d.collected_timesteps" in DASHBOARD_HTML
    assert "d.trained_timesteps" in DASHBOARD_HTML
    assert "x.trained_timesteps" in DASHBOARD_HTML
    assert "x.timesteps" not in DASHBOARD_HTML
    assert "Segment speed" in DASHBOARD_HTML
    assert "Training heartbeat delayed" in DASHBOARD_HTML


def test_prepare_dashboard_publishes_complete_page(tmp_path: Path) -> None:
    prepare_dashboard(tmp_path)

    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert page == DASHBOARD_HTML
    assert page.startswith("<!doctype html>")
    assert page.endswith("</html>")


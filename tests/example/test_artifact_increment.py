from typing import Any, Dict

import pytest

from app.core.model.artifact import ContentType
from app.core.service.artifact_service import ArtifactService


@pytest.fixture
def initial_graph():
    return {
        "vertices": [
            {"id": "1", "label": "A", "properties": {}},
            {"id": "2", "label": "B", "properties": {}},
        ],
        "edges": [{"source": "1", "target": "2", "label": "A_to_B", "properties": {}}],
    }


@pytest.fixture
def new_graph():
    return {
        "vertices": [
            {"id": "2", "label": "B", "properties": {"updated": True}},
            {"id": "3", "label": "C", "properties": {}},
        ],
        "edges": [{"source": "2", "target": "3", "label": "B_to_C", "properties": {}}],
    }


def _inc(content_type: ContentType, current_content: Any, new_content: Any) -> Any:
    # ArtifactService._increment_content is pure and does not use `self`.
    return ArtifactService._increment_content(  # type: ignore[misc]
        None,
        content_type=content_type,
        current_content=current_content,
        new_content=new_content,
    )


def test_graph_merge(initial_graph: Dict[str, Any], new_graph: Dict[str, Any]):
    result = _inc(ContentType.GRAPH, initial_graph, new_graph)

    assert "vertices" in result
    assert "edges" in result
    assert len(result["vertices"]) == 3
    assert len(result["edges"]) == 2

    vertex_2 = next((v for v in result["vertices"] if v["id"] == "2"), None)
    assert vertex_2 is not None
    assert vertex_2["properties"]["updated"] is True

    assert any(v["id"] == "3" for v in result["vertices"])

    edge_keys = [(e["source"], e["target"], e["label"]) for e in result["edges"]]
    assert ("1", "2", "A_to_B") in edge_keys
    assert ("2", "3", "B_to_C") in edge_keys


def test_graph_none_current_content(initial_graph: Dict[str, Any]):
    result = _inc(ContentType.GRAPH, None, initial_graph)
    assert len(result["vertices"]) == len(initial_graph["vertices"])
    assert len(result["edges"]) == len(initial_graph["edges"])
    assert {v["id"] for v in result["vertices"]} == {"1", "2"}


def test_graph_none_new_content(initial_graph: Dict[str, Any]):
    result = _inc(ContentType.GRAPH, initial_graph, None)
    assert result == initial_graph


def test_graph_invalid_inputs():
    result = _inc(ContentType.GRAPH, "not a dict", {"vertices": [], "edges": []})
    assert isinstance(result, dict)
    assert "vertices" in result
    assert "edges" in result

    initial = {"vertices": [], "edges": []}
    result = _inc(ContentType.GRAPH, initial, "not a dict")
    assert result == initial


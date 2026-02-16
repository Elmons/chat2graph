import json
from pathlib import Path
import random
import time
import types
from typing import Iterable, List

import pytest

from app.core.common.type import MessageSourceType
from app.core.model.message import ModelMessage
from app.core.workflow.dataset_synthesis.generator import SamplingDatasetGenerator
from app.core.workflow.dataset_synthesis.model import Row
from app.core.workflow.dataset_synthesis.sampler import RandomWalkSampler, SubGraphSampler
from app.core.workflow.dataset_synthesis.task_subtypes import GraphTaskTypesInfo
from app.core.workflow.dataset_synthesis.utils import (
    intent_verifier_alignment_ok,
    load_workflow_train_dataset,
    normalize_intent_set,
    qa_gate_reason,
)


class SequenceModelService:
    """Deterministic LLM stub returning payloads in sequence."""

    def __init__(self, payloads: Iterable[str]):
        self._payloads: List[str] = list(payloads)
        self.calls = 0

    async def generate(self, sys_prompt, messages, tools=None, tool_call_ctx=None):
        if self.calls >= len(self._payloads):
            raise AssertionError("SequenceModelService has no payload left to consume")
        payload = self._payloads[self.calls]
        self.calls += 1
        message = messages[-1]
        return ModelMessage(
            payload=payload,
            job_id=message.get_job_id(),
            step=message.get_step() + 1,
            source_type=MessageSourceType.MODEL,
        )


class StubSampler(SubGraphSampler):
    """Sampler stub that returns pre-seeded subgraph strings."""

    def __init__(self, results: Iterable[str] = ("{\"nodes\": [], \"relationships\": []}",)):
        self._results: List[str] = list(results)
        if not self._results:
            self._results.append("{\"nodes\": [], \"relationships\": []}")
        self.calls = 0

    def get_random_subgraph(self, graph_db, max_depth, max_nodes, max_edges) -> str:
        if self.calls >= len(self._results):
            return self._results[-1]
        result = self._results[self.calls]
        self.calls += 1
        return result


def _create_generator(monkeypatch, *, sampler=None, llm_payloads=None, strategy=None):
    llm = SequenceModelService(llm_payloads or [""])
    monkeypatch.setattr(
        "app.core.workflow.dataset_synthesis.generator.ModelServiceFactory.create",
        lambda *args, **kwargs: llm,
    )
    generator = SamplingDatasetGenerator(
        graph_db=object(),
        sampler=sampler or StubSampler(),
        strategy=strategy,
        max_depth=1,
        max_noeds=3,
        max_edges=3,
        nums_per_subgraph=2,
    )
    return generator, llm


def test_extract_pairs_parses_valid_rows(monkeypatch):
    generator, _ = _create_generator(monkeypatch)
    payload = (
        '{"level": "L1", "task_subtype": "SubtypeA", "task": "Q1", "verifier": "A1"}'
        '\n'
        '{"level": "L2", "task_subtype": "SubtypeB", "task": "Q2", "verifier": "A2"}'
    )
    rows = generator.extract_pairs("query", payload)
    assert len(rows) == 2
    assert rows[0].task_subtype == "SubtypeA"
    assert rows[0].task_type == "query"
    assert rows[0].task == "Q1"
    assert rows[1].task_subtype == "SubtypeB"
    assert rows[1].task_type == "query"
    assert rows[1].level == "L2"
    assert rows[1].task == "Q2"
    


def test_extract_pairs_skips_invalid_objects(monkeypatch):
    generator, _ = _create_generator(monkeypatch)
    payload = '{"level": "L1", "task": "missing fields"}'
    rows = generator.extract_pairs("query", payload)
    assert rows == []


def test_normalize_intent_set_canonicalizes_aliases():
    intents = normalize_intent_set(
        ["query", "relationship.exists", "attribute.get", "path.exists"],
        task_subtype="any",
    )
    assert intents == ["query.lookup", "query.neighbor", "query.path.reachability"]


def test_normalize_intent_set_canonicalizes_extended_l3_aliases():
    intents = normalize_intent_set(
        ["query.topology", "query.degree", "cycle.detect", "triangle.count", "query.motif.triangle", "shared_neighbors"],
        task_subtype="any",
    )
    assert intents == [
        "query.topology.degree",
        "query.cycle.exists",
        "query.motif.triangle_count",
        "query.similarity.shared_neighbors",
    ]


def test_graph_task_types_info_registers_extended_l3_subtypes():
    info = GraphTaskTypesInfo(strategy="query")
    assert "q_l3_path_constrained" in info.count_info["L3"]
    assert "q_l3_cycle_exists" in info.count_info["L3"]
    assert "q_l3_motif_triangle_count" in info.count_info["L3"]
    assert "q_l3_similarity_shared_neighbors" in info.count_info["L3"]


def test_intent_alignment_ignores_non_enforced_alias_intents():
    row = Row(
        level="L1",
        task_type="query",
        task_subtype="Direct Relationship and Neighbor Query",
        task="Who transferred to Bob?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier="MATCH (a:Account)-[:TRANSFER]->(b:Account {nickname:'Bob'}) RETURN a",
        expected_global='["Alice"]',
        intent_set=["query", "relationship.exists", "attribute.get"],
    )
    assert intent_verifier_alignment_ok(row) is True


def test_intent_alignment_keeps_strict_checks_for_ranking():
    row = Row(
        level="L2",
        task_type="query",
        task_subtype="Ranking Query",
        task="Top 3 transfer targets",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier="MATCH (a)-[:TRANSFER]->(b) RETURN b",
        expected_global="[]",
        intent_set=["query.lookup", "query.ranking.topk"],
    )
    assert intent_verifier_alignment_ok(row) is False


def test_qa_gate_rejects_unbounded_path_pattern():
    row = Row(
        level="L3",
        task_type="query",
        task_subtype="Path Analysis",
        task="shortest path",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier=(
            "MATCH p=shortestPath((a:Account {id:'1'})-[*]-(b:Account {id:'2'})) RETURN p"
        ),
        expected_global="[]",
        intent_set=["query.lookup", "query.path.shortest"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "path_missing_hop_bound"


def test_qa_gate_rejects_legacy_size_pattern_expression():
    row = Row(
        level="L2",
        task_type="query",
        task_subtype="Local Topological Index Calculation",
        task="degree",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier="MATCH (l:Loan {id:'1'}) RETURN size((l)--()) AS degree",
        expected_global="1",
        intent_set=["query.lookup"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "syntax_legacy_size_pattern_expression"


def test_qa_gate_rejects_shortest_path_variable_shadowing():
    row = Row(
        level="L3",
        task_type="query",
        task_subtype="Path Analysis",
        task="shortest path",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier=(
            "MATCH p=shortestPath((p:Person {personName:'A'})-[:INVEST*..3]->"
            "(c:Company {companyName:'B'})) RETURN p"
        ),
        expected_global="[]",
        intent_set=["query.lookup", "query.path.shortest"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "syntax_variable_shadowing"


def test_qa_gate_rejects_exact_hops_not_enforced():
    row = Row(
        level="L3",
        task_type="query",
        task_subtype="Constrained Path Query",
        task="Is there a path of exactly two hops from 'A' to 'B'?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier=(
            "MATCH p=(a {name:'A'})-[*..4]-(b {name:'B'}) "
            "RETURN CASE WHEN count(p) > 0 THEN true ELSE false END AS reachable"
        ),
        expected_global="[]",
        intent_set=["query.path.constrained"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "path_exact_hops_not_enforced"


def test_qa_gate_rejects_email_domain_mismatch():
    row = Row(
        level="L1",
        task_type="query",
        task_subtype="Single-Attribute Filtering Query",
        task="Which accounts have email domain 'gmail.com'?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier="MATCH (a:Account {email: 'gmail.com'}) RETURN a.nickname",
        expected_global="[]",
        intent_set=["query.filter.single"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "email_domain_filter_not_encoded"


def test_qa_gate_rejects_freq_login_return_mismatch():
    row = Row(
        level="L1",
        task_type="query",
        task_subtype="Attribute Lookup Query",
        task="If an account frequently logs in via PHONE, what is its freqLoginType?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier="MATCH (a:Account {freqLoginType: 'PHONE'}) RETURN a.nickname",
        expected_global="[]",
        intent_set=["query.lookup"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "task_query_semantic_mismatch"


def test_qa_gate_rejects_shortest_path_length_mismatch():
    row = Row(
        level="L3",
        task_type="query",
        task_subtype="Path Analysis",
        task="What is the shortest path length (in hops) from 'A' to 'B'?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier="MATCH (a {name:'A'}), (b {name:'B'}) RETURN shortestPath((a)-[*..4]-(b))",
        expected_global="[]",
        intent_set=["query.path.shortest"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "shortest_path_length_not_returned"


def test_qa_gate_rejects_boolean_task_with_non_boolean_return():
    row = Row(
        level="L1",
        task_type="query",
        task_subtype="Single-Step Logic Inference Query",
        task="If an Account signs in via a Medium, can that Medium be considered associated with the Account?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier="MATCH (m:Medium)-[:SIGNIN]->(a:Account) RETURN m.id, a.id LIMIT 1",
        expected_global="[]",
        intent_set=["query.reasoning.single_step"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "boolean_task_not_boolean_query"


def test_qa_gate_rejects_followed_by_path_without_ordered_chain():
    row = Row(
        level="L3",
        task_type="query",
        task_subtype="Constrained Path Query",
        task="Is there a path from 'A' to 'B' that follows a TRANSFER followed by a SIGNIN?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier=(
            "MATCH p=(a {name:'A'})-[*..4]-(b {name:'B'}) "
            "WHERE all(rel IN relationships(p) WHERE type(rel) IN ['TRANSFER','SIGNIN']) "
            "RETURN CASE WHEN count(p) > 0 THEN true ELSE false END AS constrained_reachable"
        ),
        expected_global="[]",
        intent_set=["query.path.constrained"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "path_step_constraints_not_encoded"


def test_qa_gate_rejects_two_hop_without_exact_hops():
    row = Row(
        level="L2",
        task_type="query",
        task_subtype="Path Reachability Query",
        task="Is there a two-hop path from 'A' to 'B' via WITHDRAW relationships?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier=(
            "MATCH p=(a {name:'A'})-[*..4]-(b {name:'B'}) "
            "WHERE all(rel IN relationships(p) WHERE type(rel) IN ['WITHDRAW']) "
            "RETURN CASE WHEN count(p) > 0 THEN true ELSE false END AS reachable"
        ),
        expected_global="[]",
        intent_set=["query.path.reachability"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "path_exact_hops_not_enforced"


def test_qa_gate_rejects_length_exact_pattern_without_exact_hops():
    row = Row(
        level="L2",
        task_type="query",
        task_subtype="Path Reachability Query",
        task="Is there a path of length exactly 2 from 'A' to 'B'?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier=(
            "MATCH p=(a {name:'A'})-[*..4]-(b {name:'B'}) "
            "RETURN CASE WHEN count(p) > 0 THEN true ELSE false END AS reachable"
        ),
        expected_global="[]",
        intent_set=["query.path.reachability"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "path_exact_hops_not_enforced"


def test_qa_gate_rejects_accountlevel_filter_without_accountlevel_in_query():
    row = Row(
        level="L2",
        task_type="query",
        task_subtype="Combined Attribute Filtering Query",
        task="How many accounts have accountLevel 'Silver level' and email 'gmx.com'?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier=(
            "MATCH (a:Account {name: 'Silver level', nickname: 'gmx.com'}) RETURN count(a)"
        ),
        expected_global="[]",
        intent_set=["query.filter.combined"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "task_query_semantic_mismatch"


def test_qa_gate_rejects_sum_amount_on_node_property():
    row = Row(
        level="L2",
        task_type="query",
        task_subtype="Combined Attribute Filtering Query",
        task=(
            "What is the total amount deposited from loan '1' into accounts owned by person 'A'?"
        ),
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier=(
            "MATCH (p:Person {personName:'A'})-[:OWN]->(a:Account)<-[:DEPOSIT]-(l:Loan {id:'1'}) "
            "RETURN sum(a.amount) AS total"
        ),
        expected_global="[]",
        intent_set=["query.filter.combined"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "amount_aggregation_not_on_relationship"


def test_qa_gate_rejects_shortest_path_relationship_count_mismatch():
    row = Row(
        level="L3",
        task_type="query",
        task_subtype="Path Analysis",
        task="What is the shortest path in terms of relationship count between 'A' and 'B'?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier="MATCH (a {name:'A'}), (b {name:'B'}) RETURN shortestPath((a)-[*..4]-(b))",
        expected_global="[]",
        intent_set=["query.path.shortest"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "shortest_path_length_not_returned"


def test_qa_gate_rejects_boolean_chain_without_entity_anchors():
    row = Row(
        level="L1",
        task_type="query",
        task_subtype="Single-Step Logic Inference Query",
        task="If Peper guarantees Jean, and Jean guarantees Smet, does Peper indirectly guarantee Smet?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier="MATCH ()-[r:GUARANTEE]->() RETURN count(r) > 0 AS answer",
        expected_global="[]",
        intent_set=["query.reasoning.single_step"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "task_query_semantic_mismatch"


def test_qa_gate_rejects_company_literal_bound_to_wrong_label():
    row = Row(
        level="L3",
        task_type="query",
        task_subtype="Path Analysis",
        task="What is the shortest path from the account 'A' to the company 'B'?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier=(
            "MATCH (a:Account {nickname:'A'}), (b:Account {companyName:'B'}) "
            "RETURN shortestPath((a)-[*..4]-(b))"
        ),
        expected_global="[]",
        intent_set=["query.path.shortest"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "task_query_semantic_mismatch"


def test_qa_gate_rejects_mediumtype_literal_bound_to_id_field():
    row = Row(
        level="L3",
        task_type="query",
        task_subtype="Path Analysis",
        task="What is the shortest path between Account 'A' and Medium with mediumType 'RFID'?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier=(
            "MATCH (a:Account {nickname:'A'}), (b:Medium {id:'RFID'}) "
            "RETURN shortestPath((a)-[*..4]-(b))"
        ),
        expected_global="[]",
        intent_set=["query.path.shortest"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "task_query_semantic_mismatch"


def test_qa_gate_rejects_blocked_transfer_boolean_without_blocked_filter():
    row = Row(
        level="L1",
        task_type="query",
        task_subtype="Single-Step Logic Inference Query",
        task="If an Account is blocked, can it be used for transfers?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier="MATCH ()-[r:TRANSFER]->() RETURN count(r) > 0 AS answer",
        expected_global="[]",
        intent_set=["query.reasoning.single_step"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "task_query_semantic_mismatch"


def test_qa_gate_rejects_shared_neighbor_transfer_without_transfer_constraint():
    row = Row(
        level="L3",
        task_type="query",
        task_subtype="Structural Similarity (Shared Neighbors) Query",
        task="How many shared neighboring accounts do A and B have via incoming TRANSFER relationships?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier=(
            "MATCH (a:Account {nickname:'A'}), (b:Account {nickname:'B'}) "
            "MATCH (a)--(x)--(b) RETURN count(DISTINCT x) AS shared_neighbors"
        ),
        expected_global="[]",
        intent_set=["query.similarity.shared_neighbors"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "shared_neighbor_scope_not_encoded"


def test_qa_gate_rejects_has_question_without_boolean_return():
    row = Row(
        level="L1",
        task_type="query",
        task_subtype="Direct Relationship and Neighbor Query",
        task="Has the Medium with ID '1' ever signed in to any account?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier="MATCH (m:Medium {id:'1'})-[:SIGNIN]->(a:Account) RETURN a.id LIMIT 1",
        expected_global="[]",
        intent_set=["query.neighbor"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "boolean_task_not_boolean_query"


def test_qa_gate_rejects_login_type_task_without_freq_return():
    row = Row(
        level="L1",
        task_type="query",
        task_subtype="Single-Step Logic Inference Query",
        task="If an account signs in using a Medium of type 'RFID', what login type does the account likely use frequently?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier="MATCH ()-[r:SIGNIN]->() RETURN count(r) > 0 AS answer",
        expected_global="[]",
        intent_set=["query.reasoning.single_step"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "task_query_semantic_mismatch"


def test_qa_gate_rejects_possessive_person_anchor_missing():
    row = Row(
        level="L2",
        task_type="query",
        task_subtype="Multi-Hop Chain Reasoning Query",
        task="Which account used the same medium as Berta Batko's account?",
        verifier="text verifier",
        answer_scope="global_graph",
        global_verifier="MATCH (src)-[:SIGNIN]->(mid)-[:SIGNIN]->(target) RETURN src LIMIT 3",
        expected_global="[]",
        intent_set=["query.reasoning.chain"],
    )
    assert qa_gate_reason(row, engine_hint="Neo4jGraphDb") == "task_query_semantic_mismatch"


@pytest.mark.asyncio
async def test_identify_strategy_uses_llm_response(monkeypatch):
    generator, llm = _create_generator(monkeypatch, llm_payloads=["This is clearly a query task"])
    strategy = await generator.identify_strategy("desc")
    assert strategy == "query"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_identify_strategy_returns_cached_strategy(monkeypatch):
    generator, llm = _create_generator(monkeypatch, strategy="non-query")
    result = await generator.identify_strategy("ignored")
    assert result == "query"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_generate_pairs_returns_rows(monkeypatch):
    payload = (
        '{"level": "L1", "task_subtype": "SubtypeA", "task": "Q1", "verifier": "A1"}'
        '\n'
        '{"level": "L1", "task_subtype": "SubtypeB", "task": "Q2", "verifier": "A2"}'
    )
    generator, _ = _create_generator(monkeypatch, llm_payloads=[payload])
    info = GraphTaskTypesInfo(strategy="query")
    rows = await generator.generate_pairs(
        task_type="query",
        task_types_info=info,
        subgraph="{\"nodes\": [], \"relationships\": []}",
        task_description="desc",
        nums=2,
    )
    assert len(rows) == 2
    assert all(row.task_type == "query" for row in rows)
    assert {row.task for row in rows} == {"Q1", "Q2"}
    assert {row.task_subtype for row in rows} == {"SubtypeA", "SubtypeB"}
    assert {row.verifier for row in rows} == {"A1", "A2"}


def test_get_task_type_from_strategy_direct(monkeypatch):
    generator, _ = _create_generator(monkeypatch, strategy="query")
    assert generator.get_task_type_from_strategy("query") == "query"


def test_get_task_type_from_strategy_mixed(monkeypatch):
    generator, _ = _create_generator(monkeypatch)
    n = 1000
    for _ in range(n):
        random.seed(time.time())
        t = generator.get_task_type_from_strategy("mixed")
        assert t == "query"

@pytest.mark.asyncio
async def test_filter_returns_filtered_rows(monkeypatch):
    response = (
        '{"level": "L2", "task_subtype": "SubtypeB", "task": "Q", "verifier": "A"}'
        '\n'
        '{"level": "L3", "task_subtype": "SubtypeC", "task": "Task", "verifier": "Verify"}'
    )
    generator, _ = _create_generator(monkeypatch, llm_payloads=[response])
    dataset = [
        Row(
            level="L1",
            task_type="query",
            task_subtype="SubtypeA",
            task="Question",
            verifier="Answer",
        ),
        Row(
            level="L2",
            task_type="query",
            task_subtype="SubtypeB",
            task="Q",
            verifier="A",
        ),
        Row(
            level="L3",
            task_type="query",
            task_subtype="SubtypeC",
            task="Task",
            verifier="Verify",
        ),
    ]
    result, rejected = await generator.filter(
        task_type="query",
        task_desc="desc",
        subgraph="{\"nodes\": [], \"relationships\": []}",
        dataset=dataset,
    )
    assert rejected == []
    assert len(result) == 3
    assert result[0].level == "L2"
    assert result[0].task == "Q"
    assert result[1].level == "L3"


@pytest.mark.asyncio
async def test_filter_salvages_soft_reject_enumeration(monkeypatch):
    generator, _ = _create_generator(monkeypatch, llm_payloads=["[]"])
    dataset = [
        Row(
            level="L1",
            task_type="query",
            task_subtype="Direct Relationship and Neighbor Query",
            task="Which accounts did Alice transfer money to?",
            verifier="local answer",
            generation_scope="local_subgraph",
            answer_scope="global_graph",
            intent_set=["query.neighbor"],
            global_verifier="MATCH (a:Account {nickname:'Alice'})-[:TRANSFER]->(b:Account) RETURN b.nickname",
            expected_global='["Bob"]',
        )
    ]
    result, rejected = await generator.filter(
        task_type="query",
        task_desc="desc",
        subgraph="{\"nodes\": [], \"relationships\": []}",
        dataset=dataset,
        attempt=1,
    )
    assert rejected == []
    assert len(result) == 1
    assert "top 5 results only" in result[0].task.lower()
    assert "limit 5" in (result[0].global_verifier or "").lower()


@pytest.mark.asyncio
async def test_generate_compiles_dataset(monkeypatch):
    generator, _ = _create_generator(monkeypatch, sampler=StubSampler(["stub-subgraph"]))

    async def fake_identify(self, task_desc):
        return "query"

    async def fake_generate_pairs(
        self,
        task_type,
        task_types_info,
        subgraph,
        task_description,
        nums,
    ):
        return [
            Row(
                level="L1",
                task_type=task_type,
                task_subtype="Subtype",
                task="task1",
                verifier="verify1",
            ),
            Row(
                level="L1",
                task_type=task_type,
                task_subtype="Subtype",
                task="task2",
                verifier="verify2",
            ),
        ]

    async def fake_filter(self, task_type, task_desc, subgraph, dataset, **kwargs):
        return dataset, []

    generator.identify_strategy = types.MethodType(fake_identify, generator)
    generator.generate_pairs = types.MethodType(fake_generate_pairs, generator)
    generator.filter = types.MethodType(fake_filter, generator)

    dataset = await generator.generate(
        task_desc="desc",
        dataset_name="final",
        size=2,
    )

    assert dataset.name == "final"
    assert len(dataset.data) == 2
    assert {row.task for row in dataset.data} == {"task1", "task2"}


def test_random_walk_sampler_serializes_graph(monkeypatch):
    sampler = RandomWalkSampler()

    def fake_get_random_subgraph(self, graph_db, max_depth, max_nodes, max_edges):
        nodes = [
            {"node_id": "1", "labels": ["Person"], "properties": {"name": "Alice"}},
            {"node_id": "2", "labels": ["Person"], "properties": {"name": "Bob"}},
        ]
        rels = [
            {
                "rel_id": "r1",
                "rel_type": "KNOWS",
                "start_node_id": "1",
                "end_node_id": "2",
                "properties": {},
            }
        ]
        return nodes, rels

    monkeypatch.setattr(RandomWalkSampler, "_get_random_subgraph", fake_get_random_subgraph)
    result = sampler.get_random_subgraph(graph_db=object(), max_depth=1, max_nodes=5, max_edges=5)
    graph = json.loads(result)
    assert len(graph["nodes"]) == 2
    assert graph["relationships"][0]["type"] == "KNOWS"


def test_load_workflow_train_dataset_reads_ratio():
    data_file = Path(__file__).parent / "data_example.json"
    with open(data_file, encoding="utf-8") as f:
        raw = json.load(f)

    dataset = load_workflow_train_dataset(
        task_desc="desc",
        path=str(data_file),
        ratio=0.5,
    )

    expected_len = int(len(raw) * 0.5)
    assert len(dataset.data) == expected_len
    assert dataset.task_desc == "desc"
    assert dataset.name == "test"
    assert all(isinstance(row, Row) for row in dataset.data)

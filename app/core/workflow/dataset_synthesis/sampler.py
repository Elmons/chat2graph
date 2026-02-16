from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import random
import time
from typing import Dict, List, Set, Tuple

from app.core.toolkit.graph_db.graph_db import GraphDb
from app.core.workflow.dataset_synthesis.utils import normalize_intent_set


class SubGraphSampler(ABC):
    """Abstract interface for sampling subgraphs from a graph database.

    Implementations must provide get_random_subgraph(...) which returns a
    serialized subgraph (JSON string) according to the provided constraints:
      - graph_db: graph database client/connection
      - max_depth: maximum traversal depth
      - max_nodes: maximum number of nodes in the sample
      - max_edges: maximum number of edges in the sample

    The interface separates sampling strategies from dataset generation logic.
    """
    
    @abstractmethod
    def get_random_subgraph(
        self, graph_db: GraphDb, max_depth: int, max_nodes: int, max_edges: int
    ) -> str: ...

    def get_targeted_subgraph(
        self,
        graph_db: GraphDb,
        max_depth: int,
        max_nodes: int,
        max_edges: int,
        required_intents: list[str] | None = None,
    ) -> str:
        """Return a subgraph targeted for required intents (default fallback)."""
        return self.get_random_subgraph(
            graph_db=graph_db,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    def register_acceptance(
        self,
        required_intents: list[str] | None,
        accepted_rows_count: int,
    ) -> None:
        """Record acceptance feedback from generator (optional hook)."""
        return None

    def get_sampling_metrics(self) -> dict:
        """Return sampling metrics for observability."""
        return {}


@dataclass
class _IntentSamplingStats:
    attempts: int = 0
    feasible_hits: int = 0
    accepted_subgraphs: int = 0
    accepted_rows: int = 0
    attempt_budget: int = 5

    def feasible_hit_rate(self) -> float:
        return self.feasible_hits / self.attempts if self.attempts > 0 else 0.0

    def accepted_rate(self) -> float:
        return self.accepted_subgraphs / self.attempts if self.attempts > 0 else 0.0


class StratifiedHybridSampler(SubGraphSampler):
    """SHS sampler with feasibility probes and coverage feedback.

    This implementation keeps RandomWalk behavior as the backbone and adds:
      1) targeted retries for required intents,
      2) cheap feasibility probes (accept/reject before LLM generation),
      3) per-intent metrics (attempts, feasible-hit-rate, accepted-rate).
    """

    def __init__(self):
        self._base = RandomWalkSampler()
        self._intent_stats: dict[str, _IntentSamplingStats] = {}
        self._default_attempt_budget = 6

    @staticmethod
    def _normalize_required_intents(required_intents: list[str] | None) -> list[str]:
        normalized = normalize_intent_set(required_intents or ["query.lookup"], task_subtype="")
        if not normalized:
            return ["query.lookup"]
        return normalized

    def get_random_subgraph(
        self, graph_db: GraphDb, max_depth: int, max_nodes: int, max_edges: int
    ) -> str:
        return self._base.get_random_subgraph(
            graph_db=graph_db,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    @staticmethod
    def _probe_intent_feasibility(subgraph: str, intent: str) -> bool:
        try:
            obj = json.loads(subgraph)
        except Exception:
            return False

        nodes = obj.get("nodes", []) if isinstance(obj, dict) else []
        rels = obj.get("relationships", []) if isinstance(obj, dict) else []
        node_cnt = len(nodes)
        rel_cnt = len(rels)
        labels = set()
        has_numeric_rel_prop = False
        has_numeric_node_prop = False
        for node in nodes:
            labels.update(node.get("labels", []) if isinstance(node, dict) else [])
            props = node.get("properties", {}) if isinstance(node, dict) else {}
            if isinstance(props, dict):
                for value in props.values():
                    if isinstance(value, (int, float)):
                        has_numeric_node_prop = True
                        break
        for rel in rels:
            props = rel.get("properties", {}) if isinstance(rel, dict) else {}
            if isinstance(props, dict):
                for value in props.values():
                    if isinstance(value, (int, float)):
                        has_numeric_rel_prop = True
                        break
            if has_numeric_rel_prop:
                break

        low = (intent or "").lower()
        if low in {"query.lookup", "query.reasoning.single_step", ""}:
            return node_cnt > 0
        if low in {"query.neighbor", "query.filter.single"}:
            return node_cnt >= 1 and rel_cnt >= 1
        if low in {"query.filter.combined", "query.reasoning.chain"}:
            return node_cnt >= 2 and rel_cnt >= 1
        if low == "query.path.reachability":
            return node_cnt >= 2 and rel_cnt >= 1
        if low == "query.path.shortest":
            return node_cnt >= 2 and rel_cnt >= 1
        if low == "query.path.constrained":
            return node_cnt >= 2 and rel_cnt >= 2
        if low == "query.cycle.exists":
            return node_cnt >= 2 and rel_cnt >= 2
        if low == "query.motif.triangle_count":
            return node_cnt >= 3 and rel_cnt >= 3
        if low == "query.similarity.shared_neighbors":
            return node_cnt >= 3 and rel_cnt >= 2
        if low == "query.ranking.topk":
            return node_cnt >= 2 and rel_cnt >= 1 and (has_numeric_rel_prop or has_numeric_node_prop)
        if low == "query.aggregation.count":
            return rel_cnt >= 1
        if low == "query.aggregation.group_count":
            return node_cnt >= 2 and (len(labels) >= 2 or rel_cnt >= 2)
        if low == "query.topology.degree":
            return node_cnt >= 2 and rel_cnt >= 1
        return node_cnt > 0 and rel_cnt > 0

    def _get_stats(self, intent: str) -> _IntentSamplingStats:
        key = (intent or "query.lookup").lower()
        if key not in self._intent_stats:
            self._intent_stats[key] = _IntentSamplingStats(
                attempts=0,
                feasible_hits=0,
                accepted_subgraphs=0,
                accepted_rows=0,
                attempt_budget=self._default_attempt_budget,
            )
        return self._intent_stats[key]

    def get_targeted_subgraph(
        self,
        graph_db: GraphDb,
        max_depth: int,
        max_nodes: int,
        max_edges: int,
        required_intents: list[str] | None = None,
    ) -> str:
        intents = [i.lower() for i in self._normalize_required_intents(required_intents) if i]
        primary = intents[0] if intents else "query.lookup"
        budget = max(1, self._get_stats(primary).attempt_budget)

        fallback_subgraph = ""
        for _ in range(budget):
            for intent in intents:
                self._get_stats(intent).attempts += 1
            candidate = self.get_random_subgraph(
                graph_db=graph_db,
                max_depth=max_depth,
                max_nodes=max_nodes,
                max_edges=max_edges,
            )
            if not candidate:
                continue
            if not fallback_subgraph:
                fallback_subgraph = candidate

            feasible = all(
                self._probe_intent_feasibility(candidate, intent) for intent in intents
            )
            if feasible:
                for intent in intents:
                    self._get_stats(intent).feasible_hits += 1
                return candidate
        return fallback_subgraph

    def register_acceptance(
        self,
        required_intents: list[str] | None,
        accepted_rows_count: int,
    ) -> None:
        intents = [i.lower() for i in self._normalize_required_intents(required_intents) if i]
        if not intents:
            intents = ["query.lookup"]
        for intent in intents:
            stats = self._get_stats(intent)
            if accepted_rows_count > 0:
                stats.accepted_subgraphs += 1
                stats.accepted_rows += accepted_rows_count
                # Saturated: reduce retries for this intent.
                if stats.accepted_rate() > 0.75:
                    stats.attempt_budget = max(2, stats.attempt_budget - 1)
            else:
                # Low yield: increase retries moderately.
                if stats.accepted_rate() < 0.2:
                    stats.attempt_budget = min(12, stats.attempt_budget + 1)

    def get_sampling_metrics(self) -> dict:
        rows: dict[str, dict] = {}
        for intent, stats in self._intent_stats.items():
            rows[intent] = {
                "attempts": stats.attempts,
                "feasible_hit_rate": round(stats.feasible_hit_rate(), 6),
                "accepted_rate": round(stats.accepted_rate(), 6),
                "accepted_rows": stats.accepted_rows,
                "attempt_budget": stats.attempt_budget,
            }
        attempts_by_intent = {intent: item["attempts"] for intent, item in rows.items()}
        feasible_hits_by_intent = {
            intent: self._intent_stats[intent].feasible_hits for intent in rows
        }
        accepted_hits_by_intent = {
            intent: self._intent_stats[intent].accepted_subgraphs for intent in rows
        }
        accepted_rows_by_intent = {
            intent: self._intent_stats[intent].accepted_rows for intent in rows
        }
        return {
            "intent_sampling": rows,
            "attempts_by_intent": attempts_by_intent,
            "feasible_hits_by_intent": feasible_hits_by_intent,
            "accepted_hits_by_intent": accepted_hits_by_intent,
            "accepted_rows_by_intent": accepted_rows_by_intent,
        }

class RandomWalkSampler(SubGraphSampler):
    """Sampler that builds a subgraph via biased random walks.

    This sampler performs multiple stochastic walk steps starting from a chosen
    seed node, balancing DFS/BFS tendencies via a randomized bias to increase
    sampling diversity. It maintains bookkeeping to avoid repeated seeds across
    successive samples.

    Key attributes:
      - sampled_nodes: set of previously sampled node ids (to reduce duplicates)
      - sample_counter: how many sampling attempts have been made
      - dfs_bias_range: range from which a per-sample DFS/BFS bias is drawn
    """
    def __init__(self):
        self.sampled_nodes: Set[str] = set()
        self.sample_counter = 0
        self.dfs_bias_range = (0.3, 0.7)

    def get_random_subgraph(
        self, graph_db: GraphDb, max_depth: int, max_nodes: int, max_edges: int
    ) -> str:
        """Return a JSON string for a sampled subgraph.

        Internally calls _get_random_subgraph(...) to collect node and
        relationship records, then serializes them as a JSON document.
        """
        start_time = time.time()
        print("start sampling")
        nodes, relationships = self._get_random_subgraph(
            graph_db=graph_db, max_depth=max_depth, max_nodes=max_nodes, max_edges=max_edges
        )
        if nodes and relationships:
            subgraph_json = {
                "nodes": [
                    {
                        "elementId": node["node_id"],
                        "labels": node["labels"],
                        "properties": node["properties"],
                    }
                    for node in nodes
                ],
                "relationships": [
                    {
                        "elementId": rel["rel_id"],
                        "type": rel["rel_type"],
                        "start_node_elementId": rel["start_node_id"],
                        "end_node_elementId": rel["end_node_id"],
                        "properties": rel["properties"],
                    }
                    for rel in relationships
                ],
            }
            elapsed = time.time() - start_time
            print(
                f"Successfully retrieved subgraph with {len(nodes)} nodes and {len(relationships)} "
                f"relationships. elapse: {elapsed: .2f}"
            )
            info = [node["node_id"] for node in nodes[:3]]
            print(f"first 3 nodes id: {info}")
            return json.dumps(subgraph_json, indent=4)
        else:
            return ""

    def _get_available_start_node(self, graph_db: GraphDb) -> str:
        """Select a start node that has not been sampled recently.

        Returns:
          elementId(n) as a string, or empty string on failure.
        """
        # construct exclusion clause if we have sampled nodes
        exclude_clause = ""
        params = {}
        if self.sampled_nodes:
            exclude_clause = "WHERE NOT elementId(n) IN $excluded_nodes"
            params["excluded_nodes"] = list(self.sampled_nodes)

        # randomly select one available node
        query = f"""
        MATCH (n)
        {exclude_clause}
        WITH n, rand() AS r
        ORDER BY r
        LIMIT 1
        RETURN elementId(n) AS node_id
        """

        # try to retrieve a node
        try:
            with graph_db.conn.session() as session:
                result = session.run(query, params)
                record = result.single()
                if record.get("node_id", "") != "":
                    return record["node_id"]
                self.sampled_nodes.clear()
                result = session.run(query)
                return result.single()["node_id"]
        except Exception as e:
            print(f"[_get_available_start_node] failed: {str(e)}")
            return ""

    def _random_walk_step(
        self,
        graph_db: GraphDb,
        current_nodes: Set[str],
        depth: int,
        max_depth: int,
        max_nodes: int,
        max_edges: int,
        dfs_bias: float,
    ) -> Tuple[Set[int], Set[int]]:
        """Perform one biased random-walk step and return newly discovered nodes and relationships.

        The step uses a Cypher query that blends DFS/BFS preferences via dfs_bias.
        """
        if not current_nodes or depth >= max_depth:
            return set(), set()

        # construct current node parameters
        params = {
            "current_nodes": list(current_nodes),
            "dfs_bias": dfs_bias,
            "max_possible": min(
                max_nodes - len(self.current_sample_nodes),
                max_edges - len(self.current_sample_edges),
            ),
        }

        # Cypher query to perform one random-walk step with DFS/BFS bias
        query = """
        UNWIND $current_nodes AS current_id
        MATCH (current)-[r]-(neighbor)
        WHERE elementId(current) = current_id
        
        WITH current, r, neighbor,
             rand() AS random_val,
             (1.0 / (size([n IN $current_nodes WHERE n = elementId(current)]) + 1)) * $dfs_bias +
             (CASE WHEN elementId(neighbor) IN $current_nodes THEN 0 ELSE 1 END) * (1 - $dfs_bias) AS weight
        
        ORDER BY weight DESC, random_val
        LIMIT $max_possible
        
        RETURN DISTINCT elementId(neighbor) AS node_id, elementId(r) AS rel_id
        """  # noqa: E501
        new_nodes: Set[int] = set()
        new_rels: Set[int] = set()
        try:
            with graph_db.conn.session() as session:
                result = session.run(query, params)
                for record in result:
                    if record.get("node_id", "") != "":
                        new_nodes.add(record["node_id"])
                    if record.get("rel_id", "") != "":
                        new_rels.add(record["rel_id"])
                return new_nodes, new_rels
        except Exception as e:
            print(f"[_random_walk_step] failed: {str(e)}")
            return new_nodes, new_rels

    def _get_random_subgraph(
        self, graph_db: GraphDb, max_depth: int, max_nodes: int, max_edges: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Core routine that assembles a subgraph by iterating random-walk steps.

        Enforces limits (max_depth, max_nodes, max_edges), optionally fills
        missing nodes/edges to respect limits, collects node/relationship
        details and returns them as Python lists for later serialization.
        """
        # parameter validation
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if max_nodes < 1:
            raise ValueError("max_nodes must be at least 1")
        if max_edges < 1:
            raise ValueError("max_edges must be at least 1")

        self.sample_counter += 1

        # try to retrieve a node
        start_node = self._get_available_start_node(graph_db)
        if not start_node or len(start_node) == "":
            raise Exception("[_get_random_subgraph] Cann't find start_node")

        # initialize sampling set
        self.current_sample_nodes = {start_node}
        self.current_sample_edges: Set[int] = set()
        current_frontier = {start_node}
        dfs_bias = random.uniform(*self.dfs_bias_range)  

        # perform random walk steps
        for depth in range(max_depth):
            # one step of random walk
            new_nodes, new_edges = self._random_walk_step(
                graph_db, current_frontier, depth, max_depth, max_nodes, max_edges, dfs_bias
            )

            # update sampling set
            nodes_to_add = new_nodes - self.current_sample_nodes
            edges_to_add = new_edges - self.current_sample_edges

            # check remaining slots
            remaining_node_slots = max_nodes - len(self.current_sample_nodes)
            remaining_edge_slots = max_edges - len(self.current_sample_edges)

            if remaining_node_slots <= 0 and remaining_edge_slots <= 0:
                break

            # add new nodes, not exceeding max limit
            if nodes_to_add and remaining_node_slots > 0:
                nodes_to_add = list(nodes_to_add)[:remaining_node_slots]
                self.current_sample_nodes.update(nodes_to_add)
                current_frontier = set(nodes_to_add)  # next round starts from new nodes

            # add new edges, not exceeding max limit
            if new_edges and remaining_edge_slots > 0:
                edges_to_add = list(edges_to_add)[:remaining_edge_slots]
                self.current_sample_edges.update(edges_to_add)

            # check if we need to stop
            if not nodes_to_add and not edges_to_add:
                break

        # record sampled nodes to ensure diversity in future samples
        self.sampled_nodes.update(self.current_sample_nodes)

        try:
            with graph_db.conn.session() as session:
                # 1. If node slots are full but edge slots are not: supplement edges based on selected nodes  # noqa: E501
                remaining_edges = max_edges - len(self.current_sample_edges)
                if len(self.current_sample_nodes) >= max_nodes and remaining_edges > 0:
                    # query for edges between selected nodes that have not been sampled
                    query = """
                    UNWIND $node_ids AS nid
                    MATCH (a)-[r]-(b)
                    WHERE elementId(a) IN $node_ids 
                        AND elementId(b) IN $node_ids
                        AND NOT elementId(r) IN $edge_ids
                    WITH r ORDER BY rand()  
                    LIMIT $remaining  
                    RETURN elementId(r) AS rel_id
                    """

                    result = session.run(
                        query,
                        {
                            "node_ids": list(self.current_sample_nodes),
                            "edge_ids": list(self.current_sample_edges),
                            "remaining": remaining_edges,
                        },
                    )
                    supply_edges = [record["rel_id"] for record in result]
                    self.current_sample_edges.update(supply_edges)

                # If edges are full but nodes are not: supplement nodes based on selected edges
                remaining_nodes = max_nodes - len(self.current_sample_nodes)
                if len(self.current_sample_edges) >= max_edges and remaining_nodes > 0:
                    # query for un-sampled neighbor nodes connected by selected edges
                    query = """
                    UNWIND $edge_ids AS rid
                    MATCH ()-[r]->(m) WHERE elementId(r) = rid
                    WITH DISTINCT m  
                    WHERE NOT elementId(m) IN $node_ids
                    WITH m ORDER BY rand()  
                    LIMIT $remaining  
                    RETURN elementId(m) AS node_id
                    """
                    result = session.run(
                        query,
                        {
                            "edge_ids": list(self.current_sample_edges),
                            "node_ids": list(self.current_sample_nodes),
                            "remaining": remaining_nodes,
                        },
                    )
                    supply_nodes = [record["node_id"] for record in result]
                    self.current_sample_nodes.update(supply_nodes)
                    # sync sampled nodes
                    self.sampled_nodes.update(supply_nodes)
        except Exception as e:
            print(f"[_get_random_subgraph] supply failed: {str(e)}")
            return [], []

        # retrieve detailed node information
        nodes_query = """
        UNWIND $node_ids AS id
        MATCH (n) WHERE elementId(n) = id
        RETURN elementId(n) AS node_id, labels(n) AS labels, properties(n) AS properties
        """

        # retrieve detailed relationship information
        rels_query = """
        UNWIND $rel_ids AS id
        MATCH ()-[r]->() WHERE elementId(r) = id
        RETURN elementId(r) AS rel_id, type(r) AS rel_type,
               elementId(startNode(r)) AS start_node_id,
               elementId(endNode(r)) AS end_node_id,
               properties(r) AS properties
        """

        # collect node and relationship details
        try:
            with graph_db.conn.session() as session:
                nodes_result = session.run(
                    nodes_query, {"node_ids": list(self.current_sample_nodes)}
                )
                nodes = list(nodes_result)
                rels_result = session.run(rels_query, {"rel_ids": list(self.current_sample_edges)})
                rels = list(rels_result)
        except Exception as e:
            print(f"[_get_random_subgraph] failed: {str(e)}")
            return [], []

        return nodes, rels

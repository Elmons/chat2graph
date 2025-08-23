from abc import ABC, abstractmethod
import json
import random
import time
from typing import Dict, List, Set, Tuple

from app.core.toolkit.graph_db.graph_db import GraphDb


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
        try:
            with graph_db.conn.session() as session:
                result = session.run(query, params)
                new_nodes = set()
                new_rels = set()
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
        dfs_bias = random.uniform(*self.dfs_bias_range)  # 随机DFS偏向，增加多样性

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

import os
import json
import uuid
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..config import Config
from .llm_client import LLMClient
from .logger import get_logger

logger = get_logger('mirofish.local_graph_store')

class LocalEpisode:
    def __init__(self, ep_uuid: str, processed: bool = True):
        self.uuid = ep_uuid
        self.uuid_ = ep_uuid
        self.processed = processed

class LocalNode:
    def __init__(self, node_uuid: str, name: str, labels: List[str], summary: str, attributes: Dict[str, Any], created_at: Optional[str] = None):
        self.uuid = node_uuid
        self.uuid_ = node_uuid
        self.name = name
        self.labels = labels
        self.summary = summary
        self.attributes = attributes
        self.created_at = created_at or datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "created_at": self.created_at
        }

class LocalEdge:
    def __init__(self, edge_uuid: str, name: str, fact: str, source_node_uuid: str, target_node_uuid: str, attributes: Dict[str, Any], created_at: Optional[str] = None, valid_at: Optional[str] = None, invalid_at: Optional[str] = None, expired_at: Optional[str] = None):
        self.uuid = edge_uuid
        self.uuid_ = edge_uuid
        self.name = name
        self.fact = fact
        self.source_node_uuid = source_node_uuid
        self.target_node_uuid = target_node_uuid
        self.attributes = attributes
        self.created_at = created_at or datetime.now().isoformat()
        self.valid_at = valid_at
        self.invalid_at = invalid_at
        self.expired_at = expired_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "attributes": self.attributes,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }

class LocalSearchResult:
    def __init__(self, nodes: List[LocalNode], edges: List[LocalEdge]):
        self.nodes = nodes
        self.edges = edges

class LocalEpisodeNamespace:
    def __init__(self, client: 'LocalZepClient'):
        self.client = client

    def get(self, uuid_: str) -> LocalEpisode:
        return LocalEpisode(uuid_, processed=True)

class LocalNodeNamespace:
    def __init__(self, client: 'LocalZepClient'):
        self.client = client

    def get(self, uuid_: str) -> Optional[LocalNode]:
        # Scan all graphs to find the node with this uuid
        graphs_dir = self.client._get_graphs_dir()
        if not os.path.exists(graphs_dir):
            return None
        
        for filename in os.listdir(graphs_dir):
            if filename.endswith(".json"):
                graph_id = filename[:-5]
                graph = self.client._load_graph(graph_id)
                if uuid_ in graph.get("nodes", {}):
                    n = graph["nodes"][uuid_]
                    return LocalNode(
                        node_uuid=n["uuid"],
                        name=n["name"],
                        labels=n["labels"],
                        summary=n["summary"],
                        attributes=n["attributes"],
                        created_at=n.get("created_at")
                    )
        return None

    def get_entity_edges(self, node_uuid: str) -> List[LocalEdge]:
        edges = []
        graphs_dir = self.client._get_graphs_dir()
        if not os.path.exists(graphs_dir):
            return []
            
        for filename in os.listdir(graphs_dir):
            if filename.endswith(".json"):
                graph_id = filename[:-5]
                graph = self.client._load_graph(graph_id)
                for e_uuid, e in graph.get("edges", {}).items():
                    if e["source_node_uuid"] == node_uuid or e["target_node_uuid"] == node_uuid:
                        edges.append(LocalEdge(
                            edge_uuid=e["uuid"],
                            name=e["name"],
                            fact=e["fact"],
                            source_node_uuid=e["source_node_uuid"],
                            target_node_uuid=e["target_node_uuid"],
                            attributes=e["attributes"],
                            created_at=e.get("created_at"),
                            valid_at=e.get("valid_at"),
                            invalid_at=e.get("invalid_at"),
                            expired_at=e.get("expired_at")
                        ))
        return edges

    def get_by_graph_id(self, graph_id: str, limit: int = 100, uuid_cursor: Optional[str] = None) -> List[LocalNode]:
        graph = self.client._load_graph(graph_id)
        nodes_dict = graph.get("nodes", {})
        
        nodes_list = []
        for n_uuid, n in nodes_dict.items():
            nodes_list.append(LocalNode(
                node_uuid=n["uuid"],
                name=n["name"],
                labels=n["labels"],
                summary=n["summary"],
                attributes=n["attributes"],
                created_at=n.get("created_at")
            ))
            
        # Sort by uuid to ensure stable paging order
        nodes_list.sort(key=lambda x: x.uuid)
        
        if uuid_cursor:
            # Find cursor index
            idx = -1
            for i, n in enumerate(nodes_list):
                if n.uuid == uuid_cursor:
                    idx = i
                    break
            if idx != -1:
                nodes_list = nodes_list[idx + 1:]
                
        return nodes_list[:limit]

class LocalEdgeNamespace:
    def __init__(self, client: 'LocalZepClient'):
        self.client = client

    def get_by_graph_id(self, graph_id: str, limit: int = 100, uuid_cursor: Optional[str] = None) -> List[LocalEdge]:
        graph = self.client._load_graph(graph_id)
        edges_dict = graph.get("edges", {})
        
        edges_list = []
        for e_uuid, e in edges_dict.items():
            edges_list.append(LocalEdge(
                edge_uuid=e["uuid"],
                name=e["name"],
                fact=e["fact"],
                source_node_uuid=e["source_node_uuid"],
                target_node_uuid=e["target_node_uuid"],
                attributes=e["attributes"],
                created_at=e.get("created_at"),
                valid_at=e.get("valid_at"),
                invalid_at=e.get("invalid_at"),
                expired_at=e.get("expired_at")
            ))
            
        # Sort by uuid to ensure stable paging order
        edges_list.sort(key=lambda x: x.uuid)
        
        if uuid_cursor:
            idx = -1
            for i, e in enumerate(edges_list):
                if e.uuid == uuid_cursor:
                    idx = i
                    break
            if idx != -1:
                edges_list = edges_list[idx + 1:]
                
        return edges_list[:limit]

class LocalGraphNamespace:
    def __init__(self, client: 'LocalZepClient'):
        self.client = client
        self.episode = LocalEpisodeNamespace(client)
        self.node = LocalNodeNamespace(client)
        self.edge = LocalEdgeNamespace(client)

    def create(self, graph_id: str, name: str, description: str = ""):
        logger.info(f"Creating local graph '{name}' with ID '{graph_id}'")
        graph_data = {
            "graph_id": graph_id,
            "name": name,
            "description": description,
            "ontology": {
                "entity_types": [],
                "edge_types": []
            },
            "nodes": {},
            "edges": {},
            "episodes": {}
        }
        self.client._save_graph(graph_id, graph_data)

    def set_ontology(self, graph_ids: List[str], entities: Optional[Dict[str, Any]] = None, edges: Optional[Dict[str, Any]] = None):
        # Extract ontology from dynamic classes/tuples
        entity_types_list = []
        if entities:
            for name, entity_class in entities.items():
                description = getattr(entity_class, '__doc__', f"A {name} entity.") or f"A {name} entity."
                # Extract attributes
                attributes = []
                fields = getattr(entity_class, 'model_fields', {}) or getattr(entity_class, '__fields__', {})
                for attr_name, field_val in fields.items():
                    attr_desc = getattr(field_val, 'description', attr_name) or attr_name
                    attributes.append({
                        "name": attr_name,
                        "type": "text",
                        "description": attr_desc
                    })
                entity_types_list.append({
                    "name": name,
                    "description": description[:100],
                    "attributes": attributes
                })
                
        edge_types_list = []
        if edges:
            for name, edge_def in edges.items():
                edge_class = edge_def[0]
                source_targets = edge_def[1]
                description = getattr(edge_class, '__doc__', f"A {name} relationship.") or f"A {name} relationship."
                
                # Extract source targets list
                st_list = []
                for st in source_targets:
                    st_list.append({
                        "source": getattr(st, 'source', 'Person'),
                        "target": getattr(st, 'target', 'Person')
                    })
                    
                # Extract attributes
                attributes = []
                fields = getattr(edge_class, 'model_fields', {}) or getattr(edge_class, '__fields__', {})
                for attr_name, field_val in fields.items():
                    attr_desc = getattr(field_val, 'description', attr_name) or attr_name
                    attributes.append({
                        "name": attr_name,
                        "type": "text",
                        "description": attr_desc
                    })
                    
                edge_types_list.append({
                    "name": name,
                    "description": description[:100],
                    "source_targets": st_list,
                    "attributes": attributes
                })

        for graph_id in graph_ids:
            graph = self.client._load_graph(graph_id)
            graph["ontology"] = {
                "entity_types": entity_types_list,
                "edge_types": edge_types_list
            }
            self.client._save_graph(graph_id, graph)
            logger.info(f"Ontology updated successfully for graph '{graph_id}'")

    def add_batch(self, graph_id: str, episodes: List[Any]) -> List[LocalEpisode]:
        import concurrent.futures
        graph = self.client._load_graph(graph_id)
        llm = self.client._get_llm()
        results = []

        def process_episode(ep):
            text = getattr(ep, 'data', '') or ''
            ep_uuid = f"episode_{uuid.uuid4().hex[:16]}"
            
            logger.info(f"Adding text chunk to graph '{graph_id}': {text[:100]}...")
            
            # Extract entities and relations using LLM
            extracted = self.client._extract_ontology_instances(llm, text, graph.get("ontology", {}))
            
            return {
                "ep_uuid": ep_uuid,
                "text": text,
                "extracted": extracted
            }

        # Run extraction in parallel
        processed_episodes = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            processed_episodes = list(executor.map(process_episode, episodes))

        # Merge results sequentially to avoid race conditions
        for result in processed_episodes:
            ep_uuid = result["ep_uuid"]
            
            # Save episode text
            graph["episodes"][ep_uuid] = {
                "uuid": ep_uuid,
                "data": result["text"],
                "type": "text",
                "created_at": datetime.now().isoformat()
            }
            
            # Merge extracted data into graph
            self.client._merge_extracted_data(graph, result["extracted"])
            
            results.append(LocalEpisode(ep_uuid, processed=True))

        self.client._save_graph(graph_id, graph)
        return results

    def add(self, graph_id: str, type: str, data: str):
        # Merge single text (used by memory updater)
        graph = self.client._load_graph(graph_id)
        llm = self.client._get_llm()
        
        ep_uuid = f"episode_{uuid.uuid4().hex[:16]}"
        graph["episodes"][ep_uuid] = {
            "uuid": ep_uuid,
            "data": data,
            "type": type,
            "created_at": datetime.now().isoformat()
        }
        
        # Extract entities and relations
        extracted = self.client._extract_ontology_instances(llm, data, graph.get("ontology", {}))
        self.client._merge_extracted_data(graph, extracted)
        self.client._save_graph(graph_id, graph)

    def search(self, graph_id: str, query: str, limit: int = 10, scope: str = "edges", reranker: str = "") -> LocalSearchResult:
        graph = self.client._load_graph(graph_id)
        
        matched_nodes = []
        matched_edges = []
        
        # Tokenize query
        keywords = [w.strip().lower() for w in query.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        query_lower = query.lower()
        
        def score_text(text: str) -> int:
            if not text:
                return 0
            text_lower = text.lower()
            if query_lower in text_lower:
                return 100
            score = 0
            for kw in keywords:
                if kw in text_lower:
                    score += 10
            return score

        # Search edges
        if scope in ["edges", "both"]:
            scored_edges = []
            for e_uuid, e in graph.get("edges", {}).items():
                score = score_text(e["fact"]) + score_text(e["name"])
                if score > 0:
                    scored_edges.append((score, e))
            scored_edges.sort(key=lambda x: x[0], reverse=True)
            for _, e in scored_edges[:limit]:
                matched_edges.append(LocalEdge(
                    edge_uuid=e["uuid"],
                    name=e["name"],
                    fact=e["fact"],
                    source_node_uuid=e["source_node_uuid"],
                    target_node_uuid=e["target_node_uuid"],
                    attributes=e["attributes"],
                    created_at=e.get("created_at"),
                    valid_at=e.get("valid_at"),
                    invalid_at=e.get("invalid_at"),
                    expired_at=e.get("expired_at")
                ))

        # Search nodes
        if scope in ["nodes", "both"]:
            scored_nodes = []
            for n_uuid, n in graph.get("nodes", {}).items():
                score = score_text(n["name"]) + score_text(n["summary"])
                if score > 0:
                    scored_nodes.append((score, n))
            scored_nodes.sort(key=lambda x: x[0], reverse=True)
            for _, n in scored_nodes[:limit]:
                matched_nodes.append(LocalNode(
                    node_uuid=n["uuid"],
                    name=n["name"],
                    labels=n["labels"],
                    summary=n["summary"],
                    attributes=n["attributes"],
                    created_at=n.get("created_at")
                ))
                
        return LocalSearchResult(matched_nodes, matched_edges)

    def delete(self, graph_id: str):
        path = self.client._get_graph_path(graph_id)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Local graph file deleted: {path}")

class LocalZepClient:
    def __init__(self):
        self.graph = LocalGraphNamespace(self)
        self._llm = None

    def _get_graphs_dir(self) -> str:
        d = os.path.join(Config.UPLOAD_FOLDER, 'graphs')
        os.makedirs(d, exist_ok=True)
        return d

    def _get_graph_path(self, graph_id: str) -> str:
        return os.path.join(self._get_graphs_dir(), f"{graph_id}.json")

    def _load_graph(self, graph_id: str) -> Dict[str, Any]:
        path = self._get_graph_path(graph_id)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading graph JSON: {e}")
        
        return {
            "graph_id": graph_id,
            "name": "",
            "description": "",
            "ontology": {"entity_types": [], "edge_types": []},
            "nodes": {},
            "edges": {},
            "episodes": {}
        }

    def _save_graph(self, graph_id: str, graph_data: Dict[str, Any]):
        path = self._get_graph_path(graph_id)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving graph JSON: {e}")

    def _get_llm(self) -> LLMClient:
        if self._llm is None:
            # Use the fast boost LLM for graph extraction (runs many times)
            self._llm = LLMClient.create_boost()
        return self._llm

    def _extract_ontology_instances(self, llm: LLMClient, text: str, ontology: dict) -> dict:
        entity_types = ontology.get("entity_types", [])
        edge_types = ontology.get("edge_types", [])
        
        if not entity_types and not edge_types:
            logger.warning("Empty ontology, skipping LLM extraction.")
            return {"entities": [], "edges": []}
        
        # Build a compact ontology description to reduce tokens
        entity_names = [e.get("name", "") for e in entity_types]
        edge_names = [e.get("name", "") for e in edge_types]
        
        # Build source_targets map for edges
        edge_st_map = {}
        for e in edge_types:
            st = e.get("source_targets", [])
            if st:
                edge_st_map[e["name"]] = st
            
        prompt = f"""Extract ALL possible entities and relationships from the text below.

Entity types allowed: {json.dumps(entity_names)}
Relationship types allowed: {json.dumps(edge_names)}

Text:
\"\"\"
{text}
\"\"\"

Return a JSON object with this exact schema:
{{
  "entities": [
    {{"name": "Entity Name", "label": "EntityType", "summary": "Brief description", "attributes": {{}}}}
  ],
  "edges": [
    {{"name": "EDGE_TYPE", "fact": "Fact sentence", "source_node_name": "Source", "target_node_name": "Target", "attributes": {{}}}}
  ]
}}

Rules:
- Extract EVERY named entity mentioned or implied in the text (people, organizations, places, institutions, etc.)
- Also extract entities that are IMPLICITLY referenced (e.g., "students" implies a university/school entity, "government" implies a government entity)
- For each entity pair, infer at least one relationship even if not explicitly stated
- You MUST extract at least 5 entities and 4 relationships from any text - be thorough and creative
- Entity label MUST be one of the allowed entity types. If an entity doesn't fit exactly, use the CLOSEST matching type
- Edge name MUST be one of the allowed relationship types
- Output ONLY the JSON object, no other text"""

        messages = [
            {"role": "system", "content": "You are a JSON extraction engine. Output only valid JSON, nothing else."},
            {"role": "user", "content": prompt}
        ]
        try:
            res = llm.chat_json(messages, temperature=0.1)
            entities_found = len(res.get("entities", []))
            edges_found = len(res.get("edges", []))
            logger.info(f"Extracted {entities_found} entities and {edges_found} edges from text chunk")
            return res
        except Exception as e:
            logger.error(f"Failed to extract ontology instances: {e}")
            logger.error(f"Text chunk was: {text[:200]}...")
            # Try one more time with plain chat (no json_object mode)
            try:
                logger.info("Retrying extraction without json_object mode...")
                raw = llm.chat(messages, temperature=0.1, max_tokens=2000)
                res = LLMClient._parse_json_response(raw)
                entities_found = len(res.get("entities", []))
                edges_found = len(res.get("edges", []))
                logger.info(f"Retry succeeded: {entities_found} entities and {edges_found} edges")
                return res
            except Exception as e2:
                logger.error(f"Retry also failed: {e2}")
                return {"entities": [], "edges": []}

    def _merge_extracted_data(self, graph: dict, extracted: dict):
        nodes = graph.setdefault("nodes", {})
        edges = graph.setdefault("edges", {})
        
        # 1. Add/merge nodes
        name_to_uuid = {}
        # Prepopulate name to uuid mapping from existing nodes
        for u, n in nodes.items():
            name_to_uuid[n["name"].lower()] = u
            
        for ent in extracted.get("entities", []):
            name = ent.get("name", "").strip()
            label = ent.get("label", "").strip()
            summary = ent.get("summary", "").strip()
            attributes = ent.get("attributes", {})
            
            if not name or not label:
                continue
                
            name_lower = name.lower()
            if name_lower in name_to_uuid:
                # Merge into existing node
                u = name_to_uuid[name_lower]
                n = nodes[u]
                if summary:
                    n["summary"] = (n["summary"] + "; " + summary) if n["summary"] else summary
                if attributes:
                    n["attributes"].update(attributes)
                # Keep labels unique
                if label not in n["labels"]:
                    n["labels"].append(label)
            else:
                # Create new node
                u = f"node_{uuid.uuid4().hex[:16]}"
                nodes[u] = {
                    "uuid": u,
                    "name": name,
                    "labels": ["Entity", label],
                    "summary": summary,
                    "attributes": attributes,
                    "created_at": datetime.now().isoformat()
                }
                name_to_uuid[name_lower] = u

        # 2. Add edges
        for edge in extracted.get("edges", []):
            edge_name = edge.get("name", "").strip().upper()
            fact = edge.get("fact", "").strip()
            src_name = edge.get("source_node_name", "").strip()
            tgt_name = edge.get("target_node_name", "").strip()
            attributes = edge.get("attributes", {})
            
            if not edge_name or not src_name or not tgt_name:
                continue
                
            # Lookup source/target uuids. If not exist, dynamically create fallback nodes
            src_lower = src_name.lower()
            if src_lower not in name_to_uuid:
                src_u = f"node_{uuid.uuid4().hex[:16]}"
                nodes[src_u] = {
                    "uuid": src_u,
                    "name": src_name,
                    "labels": ["Entity", "Person"],  # Default fallback label
                    "summary": f"Dynamically created entity named {src_name}",
                    "attributes": {},
                    "created_at": datetime.now().isoformat()
                }
                name_to_uuid[src_lower] = src_u
            src_uuid = name_to_uuid[src_lower]
            
            tgt_lower = tgt_name.lower()
            if tgt_lower not in name_to_uuid:
                tgt_u = f"node_{uuid.uuid4().hex[:16]}"
                nodes[tgt_u] = {
                    "uuid": tgt_u,
                    "name": tgt_name,
                    "labels": ["Entity", "Organization"],  # Default fallback label
                    "summary": f"Dynamically created entity named {tgt_name}",
                    "attributes": {},
                    "created_at": datetime.now().isoformat()
                }
                name_to_uuid[tgt_lower] = tgt_u
            tgt_uuid = name_to_uuid[tgt_lower]
            
            # Create edge
            e_uuid = f"edge_{uuid.uuid4().hex[:16]}"
            edges[e_uuid] = {
                "uuid": e_uuid,
                "name": edge_name,
                "fact": fact,
                "source_node_uuid": src_uuid,
                "target_node_uuid": tgt_uuid,
                "attributes": attributes,
                "created_at": datetime.now().isoformat(),
                "valid_at": datetime.now().isoformat(),
                "invalid_at": None,
                "expired_at": None
            }

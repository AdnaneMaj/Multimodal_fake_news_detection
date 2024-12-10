from .KnowledgeBase  import Entity , EntityLinker
from  .DBpediakb import DBpediaKB
from typing import List, Set
import logging
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class KnowledgeDistiller:
    """Main class for knowledge distillation from text."""
    
    def __init__(self, max_workers: int = 4):
        self.entity_linker = EntityLinker()
        self.knowledge_bases = [
            DBpediaKB()
            # WikidataKB()
            # YagoKB(),
            # ProbaseKB()
        ]
        self.max_workers = max_workers
     # WikidataKB()
    def query_all_kbs(self, entity: Entity) -> Set[str]:
        """Query all knowledge bases for an entity's concepts."""
        concepts = set()
        for kb in self.knowledge_bases:
            try:
                concepts.update(kb.cached_query(entity.name))
            except Exception as e:
                logger.warning(f"Failed to query {kb.__class__.__name__} for {entity.name}: {str(e)}")
        return concepts
    
    def distill_knowledge(self, text: str) -> Set[str]:
        """Extract and distill knowledge from input text."""
        entities = self.entity_linker.link_entities(text)
        if not entities:
            logger.warning("No entities found in text")
            return set()
            
        # Use ThreadPoolExecutor for parallel KB queries
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            concept_sets = list(executor.map(self.query_all_kbs, entities))
            
        # Combine all concept sets
        return set().union(*concept_sets)
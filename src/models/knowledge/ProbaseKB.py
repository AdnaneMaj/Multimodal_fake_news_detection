from KnowledgeBase  import KnowledgeBase , Entity
from typing import List, Set
import logging
import requests
from SPARQLWrapper import SPARQLWrapper, JSON


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





class ProbaseKB(KnowledgeBase):
    # def __init__(self):
    #     self.probase_data = self.load_probase_data()
        
    # def load_probase_data(self) -> dict:
    #     """Load and cache Probase data from local file."""
    #     try:
    #         # Simplified example data
    #         return {
    #             "michael bloomberg": {
    #                 "concepts": ["businessman", "politician", "philanthropist"],
    #                 "scores": [0.9, 0.8, 0.7]
    #             },
    #             "donald trump": {
    #                 "concepts": ["president", "businessman", "television personality"],
    #                 "scores": [0.95, 0.85, 0.75]
    #             }
    #         }
    #     except Exception as e:
    #         logger.error(f"Failed to load Probase data: {str(e)}")
    #         return {}

    def query_concepts(self, entity: Entity) -> Set[str]:
        """Query local Probase dataset for entity concepts."""
        try:
            entity_data = self.probase_data.get(entity.name.lower(), {})
            if not entity_data:
                return set()
            
            concepts = set()
            for concept, score in zip(entity_data["concepts"], entity_data["scores"]):
                if score >= 0.7:
                    concepts.add(concept.lower())
                    
            return concepts
        except Exception as e:
            logger.error(f"Probase query failed for {entity.name}: {str(e)}")
            return set()
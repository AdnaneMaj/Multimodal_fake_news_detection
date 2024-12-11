
from KnowledgeBase  import KnowledgeBase , Entity
from typing import List, Set
import logging
import requests
from SPARQLWrapper import SPARQLWrapper, JSON


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YagoKB(KnowledgeBase):
    def __init__(self):
        self.endpoint = SPARQLWrapper("https://yago-knowledge.org/sparql/query")
        self.endpoint.setReturnFormat(JSON)
        
    def query_concepts(self, entity: Entity) -> Set[str]:
        """Query YAGO for entity concepts."""
        try:
            query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX yago: <http://yago-knowledge.org/resource/>
                
                SELECT DISTINCT ?concept WHERE {{
                    ?entity rdfs:label "{entity.name}"@eng .
                    ?entity rdf:type ?concept .
                    FILTER(STRSTARTS(STR(?concept), "http://yago-knowledge.org/resource/"))
                }}
                LIMIT 7
            """
            
            self.endpoint.setQuery(query)
            results = self.endpoint.query().convert()
            
            concepts = set()
            for result in results["results"]["bindings"]:
                concept = result["concept"]["value"]
                concept_name = concept.split("/")[-1].lower()
                concepts.add(concept_name)
                
            return concepts
        except Exception as e:
            logger.error(f"YAGO query failed for {entity.name}: {str(e)}")
            return set()
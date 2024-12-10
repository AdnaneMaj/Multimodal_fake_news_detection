from KnowledgeBase  import KnowledgeBase , Entity
from typing import List, Set
import logging
import requests
from SPARQLWrapper import SPARQLWrapper, JSON


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class WikidataKB(KnowledgeBase):
    def __init__(self):
        self.endpoint = "https://query.wikidata.org/sparql"
        
    def query_concepts(self, entity: Entity) -> Set[str]:
        """Query Wikidata for entity concepts using SPARQL."""
        try:
            query = f"""
                SELECT DISTINCT ?conceptLabel WHERE {{
                    ?entity ?label "{entity.name}"@en.
                    ?entity wdt:P31 ?concept.
                    SERVICE wikibase:label {{
                        bd:serviceParam wikibase:language "en".
                        ?concept rdfs:label ?conceptLabel.
                    }}
                }}
                LIMIT 10
            """
            
            headers = {
                'User-Agent': 'KnowledgeDistillationBot/1.0',
                'Accept': 'application/json'
            }
            
            response = requests.get(
                self.endpoint,
                params={'query': query, 'format': 'json'},
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            concepts = set()
            for result in data['results']['bindings']:
                concepts.add(result['conceptLabel']['value'].lower())
                
            return concepts
        except Exception as e:
            logger.error(f"Wikidata query failed for {entity.name}: {str(e)}")
            return set()
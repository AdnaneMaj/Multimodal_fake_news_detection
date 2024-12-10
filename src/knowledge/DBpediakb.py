from .KnowledgeBase import  KnowledgeBase , Entity
from typing import List, Set
import logging
from SPARQLWrapper import SPARQLWrapper, JSON


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DBpediaKB(KnowledgeBase):
    def __init__(self):
        self.endpoint = SPARQLWrapper("http://dbpedia.org/sparql")
        self.endpoint.setReturnFormat(JSON)

    def query_concepts(self, entity: Entity) -> Set[str]:
        """Query DBpedia for entity concepts using SPARQL."""
        try:
            resource_name = entity.name.replace(" ", "_")
            query = f"""
                PREFIX dbo: <http://dbpedia.org/ontology/>
                PREFIX dbr: <http://dbpedia.org/resource/>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX dct: <http://purl.org/dc/terms/>
                PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
                
                SELECT DISTINCT ?concept WHERE {{
                    {{
                        dbr:{resource_name} rdf:type ?concept .
                        FILTER(STRSTARTS(STR(?concept), "http://dbpedia.org/ontology/"))
                    }}
                    UNION
                    {{
                        dbr:{resource_name} dct:subject ?category .
                        ?category skos:prefLabel ?concept .
                        FILTER(LANG(?concept) = "en")
                    }}
                    UNION
                    {{
                        dbr:{resource_name} dbo:type ?directType .
                        ?directType rdfs:label ?concept .
                        FILTER(LANG(?concept) = "en")
                    }}
                }} 
                LIMIT 7
            """
            
            self.endpoint.setQuery(query)
            results = self.endpoint.query().convert()
            
            concepts = set()
            for result in results["results"]["bindings"]:
                concept = result["concept"]["value"]
                if concept.startswith("http://"):
                    concept_name = concept.split("/")[-1].lower()
                else:
                    concept_name = concept.lower()
                concepts.add(concept_name)
            
            return concepts
        except Exception as e:
            logger.error(f"DBpedia query failed for {entity.name}: {str(e)}")
            return set()

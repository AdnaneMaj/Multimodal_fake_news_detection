from dataclasses import dataclass
from typing import List, Set, Optional
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from urllib.parse import quote
import json
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents a named entity with its metadata."""
    name: str
    confidence: float = 1.0
    entity_type: Optional[str] = None

class EntityLinker:
    """Handles entity linking from text using spaCy."""
    
    def __init__(self, confidence_threshold: float = 0.5, model: str = "en_core_web_lg"):
        """
        Initialize the EntityLinker with spaCy.
        
        Args:
            confidence_threshold: Minimum confidence score for entity detection
            model: Name of the spaCy model to use
        """
        try:
            self.nlp = spacy.load(model)
            self.confidence_threshold = confidence_threshold
            
            # # Enable neural coref if available
            # try:
            #     # import neuralcoref
            #     # neuralcoref.add_to_pipe(self.nlp)
            #     logger.info("Neural coreference resolution enabled")
            # except ImportError:
            #     logger.warning("neuralcoref not available, skipping coreference resolution")
                
        except ImportError as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            logger.error("Please install spaCy and download the model using:")
            logger.error(f"python -m spacy download {model}")
            raise
            
    def get_entity_confidence(self, ent) -> float:
        """Calculate confidence score for an entity."""
        # Basic confidence scoring based on entity label probability
        base_score = ent._.coref_score if hasattr(ent._, 'coref_score') else 0.8
        
        # Adjust score based on entity length
        length_factor = min(len(ent.text.split()) / 3, 1.0)
        
        # Adjust score based on entity label
        label_scores = {
            'PERSON': 0.9,
            'ORG': 0.85,
            'GPE': 0.85,
            'LOC': 0.8,
            'FACILITY': 0.8,
            'PRODUCT': 0.75,
            'EVENT': 0.75,
            'WORK_OF_ART': 0.7,
            'LAW': 0.7,
            'LANGUAGE': 0.7
        }
        label_score = label_scores.get(ent.label_, 0.6)
        
        # Combine scores
        final_score = (base_score + length_factor + label_score) / 3
        return min(final_score, 1.0)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text before entity linking."""
        text = text.strip()
        text = ' '.join(text.split())
        if text and not text[-1] in '.!?':
            text += '.'
        return text
        
    def link_entities(self, text: str) -> List[Entity]:
        """Links entities in text to knowledge base entries using spaCy."""
        try:
            text = self.preprocess_text(text)
            doc = self.nlp(text)
            entities = []
            seen_entities = set()
            
            # First pass: collect all named entities
            for ent in doc.ents:
                if ent.text.lower() in seen_entities:
                    continue
                    
                confidence = self.get_entity_confidence(ent)
                
                if confidence >= self.confidence_threshold:
                    entities.append(Entity(
                        name=ent.text,
                        confidence=confidence,
                        entity_type=ent.label_
                    ))
                    seen_entities.add(ent.text.lower())
            
            # Second pass: resolve coreferences if available
            if hasattr(doc, 'coref_clusters'):
                for cluster in doc._.coref_clusters:
                    main_mention = cluster.main
                    
                    if main_mention.text.lower() in seen_entities:
                        continue
                    
                    if main_mention.ent_type_:
                        confidence = self.get_entity_confidence(main_mention)
                        
                        if confidence >= self.confidence_threshold:
                            entities.append(Entity(
                                name=main_mention.text,
                                confidence=confidence,
                                entity_type=main_mention.ent_type_
                            ))
                            seen_entities.add(main_mention.text.lower())
            
            entities.sort(key=lambda x: x.confidence, reverse=True)
            
            if entities:
                logger.info(f"Found {len(entities)} entities: {', '.join(e.name for e in entities)}")
            else:
                logger.warning("No entities found in text")
                
            return entities
            
        except Exception as e:
            logger.error(f"Entity linking failed: {str(e)}")
            return []
    
    def get_entity_context(self, text: str, entity_name: str, window_size: int = 20) -> str:
        """Get the context surrounding an entity mention."""
        try:
            start_idx = text.lower().find(entity_name.lower())
            if start_idx == -1:
                return ""
                
            context_start = max(0, start_idx - window_size)
            context_end = min(len(text), start_idx + len(entity_name) + window_size)
            
            return text[context_start:context_end]
        except Exception as e:
            logger.error(f"Failed to get entity context: {str(e)}")
            return ""

class KnowledgeBase(ABC):
    """Abstract base class for knowledge base queries."""
    
    @abstractmethod
    def query_concepts(self, entity: Entity) -> Set[str]:
        """Query concepts for an entity from the knowledge base."""
        pass
    
    @lru_cache(maxsize=100)
    def cached_query(self, entity_name: str) -> Set[str]:
        """Cached wrapper for concept queries."""
        return self.query_concepts(Entity(entity_name))
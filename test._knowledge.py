from  src.knowledge.knowledgeDistiller import KnowledgeDistiller

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Example usage
    distiller = KnowledgeDistiller()
    text = "ISIS Media account posts pic claiming to be Michael Zehaf-Bibeau, dead  OttawaShooting terrorist http://t.co/dascEeLdip via  ArmedResearch"
    
    try:
        concepts = distiller.distill_knowledge(text)
        print(f"Found concepts: {concepts}")
    except Exception as e:
        logger.error(f"Knowledge distillation failed: {str(e)}")
 
        raise
main()
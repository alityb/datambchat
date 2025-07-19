import chromadb
import os
from typing import List, Dict, Any, Optional
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB vector store for football knowledge.
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection("football_knowledge")
        except:
            self.collection = self.client.create_collection("football_knowledge")
        
        # Initialize with sample data if collection is empty
        if self.collection.count() == 0:
            self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """
        Initialize the vector store with sample football knowledge.
        """
        sample_docs = [
            {
                "text": "Poss+/- (Possession Plus/Minus) measures a player's impact on possession. It's calculated as the difference between possessions won and possessions lost per 90 minutes. A positive value indicates the player helps their team maintain possession, while a negative value suggests they lose possession more often.",
                "metadata": {"type": "stat_definition", "stat": "Poss+/-", "category": "possession"}
            },
            {
                "text": "xG (Expected Goals) measures the quality of a player's scoring chances. It assigns a probability to each shot based on factors like distance, angle, and defensive pressure. Higher xG values indicate better scoring opportunities.",
                "metadata": {"type": "stat_definition", "stat": "xG", "category": "attacking"}
            },
            {
                "text": "xA (Expected Assists) measures the quality of a player's passing to create scoring chances. It calculates the probability that a pass will result in a goal, based on the recipient's position and other factors.",
                "metadata": {"type": "stat_definition", "stat": "xA", "category": "attacking"}
            },
            {
                "text": "npxG (Non-Penalty Expected Goals) is xG excluding penalty kicks. This gives a better measure of a player's open-play scoring ability, as penalties are high-probability shots that don't reflect general attacking performance.",
                "metadata": {"type": "stat_definition", "stat": "npxG", "category": "attacking"}
            },
            {
                "text": "Goals + Assists per 90 is a combined attacking metric that measures a player's direct contribution to goals. It's useful for comparing attacking players across different positions and teams.",
                "metadata": {"type": "stat_definition", "stat": "Goals + Assists", "category": "attacking"}
            },
            {
                "text": "Exit Line refers to a defensive metric that measures how effectively a player prevents the opposition from progressing the ball into dangerous areas. It's particularly relevant for defenders and defensive midfielders.",
                "metadata": {"type": "stat_definition", "stat": "Exit Line", "category": "defensive"}
            },
            {
                "text": "Progressive actions per 90 measures how often a player moves the ball forward significantly (typically 10+ meters towards the opponent's goal). This includes both progressive passes and progressive carries.",
                "metadata": {"type": "stat_definition", "stat": "Progressive actions", "category": "possession"}
            },
            {
                "text": "A midfielder is a player who operates in the middle of the field, typically responsible for both attacking and defensive duties. They can be categorized as defensive midfielders, central midfielders, or attacking midfielders based on their primary role.",
                "metadata": {"type": "position_definition", "position": "Midfielder", "category": "position"}
            },
            {
                "text": "A striker is an attacking player whose primary role is to score goals. They typically play in the most advanced position on the field and are the main goal-scoring threat for their team.",
                "metadata": {"type": "position_definition", "position": "Striker", "category": "position"}
            },
            {
                "text": "A defender is a player whose primary role is to prevent the opposition from scoring. They can be center-backs, full-backs, or wing-backs, each with different responsibilities in the defensive system.",
                "metadata": {"type": "position_definition", "position": "Defender", "category": "position"}
            },
            {
                "text": "To analyze player performance, consider multiple metrics rather than relying on a single stat. For attacking players, look at xG, xA, and Goals + Assists. For midfielders, consider Poss+/-, Progressive actions, and passing accuracy. For defenders, focus on defensive duels, interceptions, and exit line metrics.",
                "metadata": {"type": "analysis_guide", "category": "general"}
            },
            {
                "text": "When comparing players, always consider their position, team style, and league context. A midfielder in a possession-heavy team will naturally have different stats than one in a counter-attacking system.",
                "metadata": {"type": "analysis_guide", "category": "general"}
            }
        ]
        
        # Add documents to collection
        for i, doc in enumerate(sample_docs):
            self.add_document(
                doc["text"], 
                metadata=doc["metadata"],
                doc_id=f"doc_{i}"
            )
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None, doc_id: str = None):
        """
        Add a document to the vector store.
        """
        if metadata is None:
            metadata = {}
        if doc_id is None:
            doc_id = f"doc_{self.collection.count()}"
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
    
    def query(self, query: str, n_results: int = 5, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Query the vector store with hybrid search (semantic + keyword).
        Returns a list of dictionaries with 'text', 'metadata', and 'distance' keys.
        """
        try:
            # Perform vector search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_dict
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'text': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return []
    
    def get_stat_definition(self, stat_name: str) -> Optional[str]:
        """
        Get the definition of a specific stat.
        """
        results = self.query(
            f"definition of {stat_name}",
            filter_dict={"type": "stat_definition"}
        )
        return results[0]['text'] if results else None
    
    def get_position_info(self, position: str) -> Optional[str]:
        """
        Get information about a specific position.
        """
        results = self.query(
            f"information about {position} position",
            filter_dict={"type": "position_definition"}
        )
        return results[0]['text'] if results else None 
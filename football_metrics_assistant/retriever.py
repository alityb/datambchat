from typing import Dict, Any, List, Optional
from vector_store import VectorStore

class HybridRetriever:
    def __init__(self, vector_db: VectorStore = None):
        """
        Initialize hybrid retriever with vector store and keyword search capabilities.
        """
        self.vector_db = vector_db or VectorStore()
    
    def retrieve(self, query: str, preprocessed_hints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform hybrid retrieval (vector + keyword) based on the query and preprocessed hints.
        """
        results = {
            "query": query,
            "matches": [],
            "stat_definitions": [],
            "position_info": [],
            "analysis_guides": [],
            "keywords": []
        }
        
        # Extract keywords for fallback search
        keywords = self._extract_keywords(query)
        results["keywords"] = keywords
        
        # Vector search for relevant documents
        vector_results = self._vector_search(query, preprocessed_hints)
        results["matches"].extend(vector_results)
        
        # Categorize results
        for match in vector_results:
            metadata = match.get('metadata', {})
            doc_type = metadata.get('type', '')
            
            if doc_type == 'stat_definition':
                results["stat_definitions"].append(match)
            elif doc_type == 'position_definition':
                results["position_info"].append(match)
            elif doc_type == 'analysis_guide':
                results["analysis_guides"].append(match)
        
        # Add specific stat definitions if stat is mentioned
        if preprocessed_hints and preprocessed_hints.get('stat'):
            stat_name = preprocessed_hints['stat']
            if isinstance(stat_name, list):
                stat_name = stat_name[0]  # Take the first stat if multiple
            stat_def = self.vector_db.get_stat_definition(stat_name)
            if stat_def:
                results["stat_definitions"].append({
                    'text': stat_def,
                    'metadata': {'type': 'stat_definition', 'stat': stat_name},
                    'distance': 0.0
                })
        
        # Add position information if position is mentioned
        if preprocessed_hints and preprocessed_hints.get('position'):
            position = preprocessed_hints['position']
            if isinstance(position, list):
                position = position[0]  # Take the first position if multiple
            pos_info = self.vector_db.get_position_info(position)
            if pos_info:
                results["position_info"].append({
                    'text': pos_info,
                    'metadata': {'type': 'position_definition', 'position': position},
                    'distance': 0.0
                })
        
        return results
    
    def _vector_search(self, query: str, preprocessed_hints: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform vector search with optional filtering based on preprocessed hints.
        """
        try:
            # Build filter based on preprocessed hints
            filter_dict = None
            if preprocessed_hints:
                if preprocessed_hints.get('stat'):
                    filter_dict = {"type": "stat_definition"}
                elif preprocessed_hints.get('position'):
                    filter_dict = {"type": "position_definition"}
            
            # Perform vector search
            results = self.vector_db.query(query, n_results=5, filter_dict=filter_dict)
            
            # If no filtered results, try without filter
            if not results and filter_dict:
                results = self.vector_db.query(query, n_results=5)
            
            return results
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from the query for fallback search.
        """
        # Simple keyword extraction - can be enhanced with NLP
        import re
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def get_stat_definition(self, stat_name: str) -> Optional[str]:
        """
        Get definition for a specific stat.
        """
        return self.vector_db.get_stat_definition(stat_name)
    
    def get_position_info(self, position: str) -> Optional[str]:
        """
        Get information about a specific position.
        """
        return self.vector_db.get_position_info(position) 
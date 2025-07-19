# Football Metrics Conversational Assistant

A smart, conversational football analytics system that:
- Understands football-specific metrics (e.g., "Poss+/-", "Top 5", "Exit Line")
- Answers in a clean, conversational way (not just dataframes)
- Generates charts and reports on request
- Handles fuzzy, alias-heavy queries about players, teams, and stats
- Supports definitions, reports, and examples of stats, players, filters
- Can be updated with new concepts/quirks
- Uses efficient hybrid retrieval (vector + keyword)
- Feels fast and interactive like Blitz or Statmuse

## Stack
- **Structured Data:** Pandas or DuckDB
- **Unstructured Retrieval:** ChromaDB (vector store)
- **Retriever:** Hybrid (semantic + keyword)
- **LLM:** Llama 3 via Ollama (local, free)
- **API:** FastAPI
- **Tools:** Python functions for charts, filters, summaries
- **Preprocessing:** Custom Python logic for query normalization

## Setup
1. **Install Ollama** ([instructions](https://ollama.com/)):  
   `curl -fsSL https://ollama.com/install.sh | sh`
2. **Start Llama 3 model:**  
   `ollama run llama3`
3. **Install Python dependencies:**  
   `pip install -r requirements.txt`  
   (also: `pip install fastapi uvicorn chromadb duckdb`)
4. **Run the API server:**  
   `uvicorn football_metrics_assistant.main:app --reload`

## Endpoints
- `/chat` — POST: Conversational endpoint (uses Llama 3 via Ollama)
- `/stat-definitions` — GET: Returns stat definitions (to be implemented)
- `/health` — GET: Health check

## Next Steps
- Implement hybrid retriever (ChromaDB + keyword)
- Add stat definitions and alias logic
- Build chart/report generation tools
- Connect preprocessing and retrieval to LLM pipeline 
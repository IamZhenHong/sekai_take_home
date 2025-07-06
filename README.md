# 🚀 Agents Flow - Multi-Agent Recommendation System

A sophisticated multi-agent recommendation system built with LangGraph, featuring recommendation generation, evaluation, and prompt optimization in an iterative loop. The system uses advanced semantic search, fuzzy tag matching, and LLM-powered agents to deliver personalized content recommendations.

## 🏗️ Architecture Overview

### Multi-Agent System
The system consists of three specialized agents working in a coordinated loop:

1. **🤖 Recommendation Agent** - Generates personalized content recommendations
2. **📊 Evaluation Agent** - Evaluates recommendation quality and builds taste profiles  
3. **🔧 Optimiser Agent** - Optimizes prompts based on feedback for better results

### Core Components
- **LangGraph Framework** - Orchestrates agent interactions and state management
- **ChromaDB** - Vector database for semantic similarity search
- **OpenAI Embeddings** - Powers semantic content understanding
- **FastAPI** - REST API with WebSocket support for real-time updates

## 📁 Project Structure

```
agents_flow/
├── agents/                          # Multi-agent system
│   ├── graph.py                     # Main LangGraph orchestration
│   ├── recommendation_agent.py      # Content recommendation logic
│   ├── evaluation_agent.py          # Quality evaluation & taste profiling
│   └── optimiser_agent.py          # Prompt optimization
├── datasets/                        # Data storage
│   ├── contents.csv                 # Content metadata & tags
│   ├── interactions.csv             # User-content interactions
│   └── users.csv                   # User profiles & interest tags
├── embeddings/                      # Vector database utilities
│   ├── chroma_utils.py             # ChromaDB integration
│   ├── embed_contents.py           # Content embedding scripts
│   └── embed_contents_advanced.py  # Advanced embedding pipeline
├── frontend/                        # React-based user interface
│   ├── src/                        # Source code
│   ├── public/                     # Static assets
│   ├── package.json                # Node.js dependencies
│   └── README.md                   # Frontend documentation
├── chroma_db/                      # Persistent vector database
├── main.py                         # FastAPI server
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd agents_flow

# Create virtual environment
python -m venv sekai-env
source sekai-env/bin/activate  # On Windows: sekai-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 3. Data Preparation

Ensure your datasets are in the `datasets/` folder:
- `contents.csv` - Content metadata (content_id, title, intro, generated_tags)
- `interactions.csv` - User interactions (user_id, content_id, interaction_count)
- `users.csv` - User profiles (user_id, user_interest_tags)

### 4. One-Command Development Startup

We provide convenient startup scripts that handle everything automatically:

#### Option A: Python Script (Recommended)
```bash
python start_dev.py
```

#### Option B: Shell Script
```bash
./start_dev.sh
```

These scripts will automatically:
1. 🏷️ Generate tags for content using AI
2. 🔍 Create advanced embeddings for semantic search
3. 🔗 Create combined embeddings for better matching
4. 🚀 Start the FastAPI backend server
5. 📱 Start the React frontend development server

### 5. Manual Setup (Alternative)

If you prefer to run processes manually:

```bash
# Generate tags for content
python batch_generate_tags.py

# Create embeddings
python embeddings/embed_contents_advanced.py
python embeddings/embed_contents_combined.py

# Start backend
python main.py

# In another terminal, start frontend
cd frontend
npm install
npm run dev
```

### 6. Access the Application

- **Frontend**: http://localhost:5173 (or check console for port)
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🔧 Core Features

### 🤖 Recommendation Agent
- **Fuzzy Tag Matching** - Intelligent content filtering with weighted tag scoring
- **Semantic Search** - ChromaDB-powered similarity search
- **Dynamic Prompting** - Context-aware recommendation generation
- **Performance Optimized** - Cached embeddings and pre-computed tag sets

### 📊 Evaluation Agent
- **Multi-Metric Evaluation** - Precision, tag overlap, semantic similarity
- **Parallel Processing** - Independent calculation nodes for speed
- **Taste Profile Synthesis** - LLM-powered user preference learning
- **Ground Truth Generation** - User interaction-based validation

### 🔧 Optimiser Agent
- **Prompt Optimization** - Iterative prompt improvement based on feedback
- **Performance Tracking** - Timing and quality metrics
- **Adaptive Learning** - Continuous improvement through iterations

## 📡 API Endpoints

### Core Endpoints

#### `GET /`
Health check and welcome message.

#### `GET /users`
Get list of available user IDs.

#### `POST /recommend`
Get personalized recommendations with full iteration data.

**Request:**
```json
{
  "user_id": "2376105",
  "k": 3,
  "seed": 42,
  "max_iterations": 2
}
```

**Response:**
```json
{
  "user_id": "2376105",
  "user_tags": ["romance", "adventure"],
  "iterations": [
    {
      "iteration_number": 1,
      "recommendation": {
        "recommendation_list": ["123", "456", "789"],
        "precision": 0.8,
        "recommendation_time": 2.3
      },
      "evaluation": {
        "precision": 0.8,
        "tag_overlap_ratio": 0.6,
        "semantic_overlap_ratio": 0.7,
        "evaluation_time": 1.5
      },
      "optimiser": {
        "optimised_prompt": "Improved prompt...",
        "optimiser_time": 0.8
      }
    }
  ],
  "final_state": {
    "recommendation_list": ["123", "456", "789"],
    "precision": 0.8,
    "taste_profile": "User enjoys romantic adventure...",
    "total_execution_time": 4.6
  }
}
```

### Content Endpoints

#### `GET /content/{content_id}`
Get specific content details.

#### `GET /content/batch?content_ids=123,456,789`
Get batch content information.

### WebSocket Endpoints

#### `WS /ws/recommend`
Real-time recommendation streaming with live updates.

## 🎨 Frontend Interface

The project includes a React-based frontend that provides an intuitive user interface for interacting with the recommendation system.

### Features
- **User Selection** - Choose from available users or create custom profiles
- **Real-time Streaming** - Watch agent progress in real-time via WebSocket
- **Iteration Tracking** - Visualize recommendation quality improvements
- **Content Display** - Browse recommended content with metadata
- **Performance Metrics** - Monitor timing and precision metrics

### Running the Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

The frontend will automatically open at `http://localhost:3000` and connect to the API at `http://localhost:8000`.

### Frontend Development

```bash
# Run in development mode with hot reload
npm start

# Build for production
npm run build

# Run tests
npm test
```

For detailed frontend documentation, see `frontend/README.md`.

## 🔄 Agent Flow

### Iteration Process
1. **Recommendation** → Generate content recommendations using fuzzy tag matching + semantic search
2. **Evaluation** → Calculate precision, build taste profile, generate feedback
3. **Optimisation** → Improve recommendation prompt based on feedback
4. **Loop** → Repeat until precision ≥ 0.5 or max iterations reached

### Performance Optimizations
- **LLM Response Caching** - Avoids redundant API calls
- **Pre-computed Tag Sets** - O(1) tag lookups
- **ChromaDB Collection Caching** - Persistent vector database
- **Parallel Evaluation Nodes** - Independent metric calculations
- **Batched Embedding Queries** - Reduced API calls

## 🛠️ Development

### Running Individual Agents

```bash
# Test recommendation agent
python -c "from agents.recommendation_agent import recommendation_graph; print('Ready')"

# Test evaluation agent  
python -c "from agents.evaluation_agent import evaluation_agent; print('Ready')"

# Test optimiser agent
python -c "from agents.optimiser_agent import optimiser_graph; print('Ready')"
```

### Embedding Management

```bash
# Generate embeddings for new content
python embeddings/embed_contents_advanced.py

# Check embedding status
python embeddings/chroma_utils.py
```

### Performance Profiling

```bash
# Profile evaluation performance
python -c "from agents.evaluation_agent import evaluation_agent; import time; start=time.time(); result=evaluation_agent({'user_id':'123','recommendation_list':['1','2','3']}); print(f'Time: {time.time()-start:.2f}s')"
```

## 📊 Performance Metrics

### Expected Improvements
- **LLM Calls**: 60-80% reduction through caching
- **Tag Processing**: 70-90% faster with pre-computed sets
- **ChromaDB Queries**: 50% reduction through batching
- **Overall Evaluation**: 40-60% performance improvement

### Monitoring
- Real-time timing metrics for each agent
- Precision tracking across iterations
- Memory usage optimization
- API response time monitoring

## 🔍 Advanced Usage

### Custom Recommendation Prompts

```python
# Custom prompt with specific requirements
prompt = "Find romantic adventure stories with strong female leads and emotional depth"
```

### Batch Processing

```python
# Process multiple users
for user_id in user_ids:
    result = compiled_graph.invoke({
        "user_id": user_id,
        "user_tags": sample_user_tags(user_id, k=5),
        "max_iterations": 3
    })
```

### WebSocket Streaming

```javascript
// Real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/recommend');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Agent update:', data);
};
```

## 🐛 Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**
   ```bash
   echo "OPENAI_API_KEY=your_key" > .env
   ```

2. **ChromaDB Connection Issues**
   ```bash
   rm -rf chroma_db/
   python embeddings/embed_contents_advanced.py
   ```

3. **Dataset Path Errors**
   - Ensure datasets are in `datasets/` folder
   - Check file permissions and CSV format

4. **Memory Issues**
   - Reduce batch sizes in embedding scripts
   - Monitor ChromaDB collection size

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python main.py
```

## 📈 Scaling Considerations

- **Horizontal Scaling**: Multiple API instances with shared ChromaDB
- **Caching Strategy**: Redis for LLM response caching
- **Database Optimization**: PostgreSQL for user/content metadata
- **Load Balancing**: Nginx reverse proxy for API distribution

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangGraph** - Multi-agent orchestration framework
- **ChromaDB** - Vector database for semantic search
- **OpenAI** - LLM and embedding APIs
- **FastAPI** - Modern web framework for APIs

---

**Built with ❤️ for intelligent recommendation systems** 
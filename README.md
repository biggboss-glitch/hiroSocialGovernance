# Hiro Social Governance

Hiro Social Governance is a real-time, multi-agent AI system designed for analyzing and simulating public opinion and social governance scenarios. It utilizes advanced Large Language Models (LLMs) and Knowledge Graphs to extract insights from raw text data and simulate the propagation of information and interactions among different social roles across multiple platforms (like Twitter and Reddit).

## Features

- **Automated Ontology Generation:** Upload a text document, and the system automatically analyzes its content to extract relevant entity types and relationship types, forming an ontology tailored to the specific domain.
- **Knowledge Graph Construction:** Uses the Zep API to parse document text into chunks and extract entities and relationships, constructing a detailed, real-time Knowledge Graph.
- **Role Profile Generation:** The Oasis Profile Generator creates realistic simulation profiles (personas) based on the entities found in the graph, assigning them specific traits, platforms, and activity levels.
- **Multi-Agent Simulation:** Runs a multi-agent simulation where individual AI agents (representing different personas) interact with each other. These agents have memory, context from the knowledge graph, and specific behavioral traits.
- **Cross-Platform Dynamics:** Simulates behavior across multiple social media platforms, mimicking platform-specific communication styles (e.g., short, punchy tweets vs. longer, discussion-focused Reddit posts).
- **Interactive Reporting and Q&A:** A dedicated Report Agent provides panoramic search, detailed fact-checking, and interactive Q&A capabilities, allowing users to deeply analyze the simulation results and knowledge graph.

## Architecture

The project is structured with a Python backend and a Vue.js frontend:

- **Frontend (`frontend/`):** A modern, responsive web interface built with Vue 3 and Vite. It guides the user through a step-by-step process: Graph Build, Environment Setup, Simulation Execution, Reporting, and Interaction.
- **Backend (`backend/`):** A robust Flask application serving the API.
  - **LLM Integration:** Powered by Google Gemini (`gemini-2.5-pro` and `gemini-2.5-flash`).
  - **Memory & Graph:** Integrates with Zep for long-term memory and knowledge graph storage.
  - **Simulation Engine:** Custom-built manager for orchestrating parallel AI agents, managing their state, and logging their interactions.
  - **Static Serving:** The backend is configured to serve the built Vue frontend files directly from the root path, simplifying deployment.

## Prerequisites

- **Python:** 3.10 or higher.
- **Node.js:** 18 or higher.
- **`uv`:** Fast Python package installer and resolver.
- **API Keys:**
  - `GEMINI_API_KEY`: For LLM inference.
  - `ZEP_API_URL` and `ZEP_API_KEY`: For knowledge graph and memory storage.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/biggboss-glitch/hiroSocialGovernance.git
cd hiroSocialGovernance
```

### 2. Environment Variables

Create a `.env` file in the `backend/` directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
ZEP_API_URL=your_zep_url_here
ZEP_API_KEY=your_zep_key_here
```

### 3. Build the Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

The built frontend files will be placed in `frontend/dist`. The Flask backend is configured to serve these files automatically.

### 4. Run the Backend

It is recommended to use `uv` for managing the Python environment and running the app:

```bash
cd backend
uv run python run.py
```

The application will be available at `http://localhost:7860`.

## Usage Workflow

1. **Step 1: Graph Build:** Upload a source text document. The system will extract the ontology and build the knowledge graph.
2. **Step 2: Environment Setup:** Configure the simulation parameters, such as the total duration and the scenario context.
3. **Step 3: Simulation Execution:** Start the simulation. Watch as agents interact and generate posts/comments in real-time.
4. **Step 4: Report Generation:** Once the simulation completes, the Report Agent generates a comprehensive summary, including key facts, interview logs, and structural analysis.
5. **Step 5: Interactive Q&A:** Chat directly with the Report Agent to ask specific questions about the data, the agents, or the predicted outcomes.

## Deployment

The application is designed to be easily deployable using Docker, specifically targeting environments like Hugging Face Spaces.

### Docker Deployment

A `Dockerfile` is provided at the root of the repository. It handles:
1. Setting up the Node.js environment and building the Vue frontend.
2. Setting up the Python environment using `uv` and installing dependencies.
3. Copying the necessary files and starting the Flask server on port `7860`.

To build and run locally with Docker:

```bash
docker build -t hiro-social-governance .
docker run -p 7860:7860 --env-file backend/.env hiro-social-governance
```

## License

MIT License
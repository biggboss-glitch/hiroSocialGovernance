# HIRO Social Governance Simulator 🐙

Welcome to the **HIRO Social Governance Simulator**! This document serves as a complete, step-by-step guide to understanding, installing, and explaining every feature of the project. It is perfectly tailored for your hackathon presentation or demo video.

---

## 🌟 1. Project Overview

**What is HIRO?**
HIRO is an AI-powered social governance simulator designed to model how public policies, events, or decisions might impact a society. By inputting a scenario, the system uses Large Language Models (LLMs) to generate diverse citizen personas and simulates their interactions (like a virtual Reddit or Twitter). It helps policymakers, sociologists, and community managers visualize the "butterfly effect" of governance decisions before they are implemented in the real world.

**Key Technologies:**
- **Frontend:** Vue.js + Vite (for a responsive, interactive UI)
- **Backend:** Python + Flask (handling API requests and orchestration)
- **Simulation Engine:** CAMEL-AI & OASIS (for multi-agent roleplaying and social network simulation)
- **Memory & Graph:** Mem0 (for long-term agent memory) & Local JSON Graph Storage (for relationship mapping)
- **Deployment:** Docker & Hugging Face Spaces

---

## 🚀 2. Getting Started & Installation

If someone downloads the project from GitHub, here is how they can get it running from start to finish.

### Prerequisites
- Install **Node.js** (v18+)
- Install **Python** (v3.10 or v3.11)
- Get API Keys: You will need an OpenAI API Key (or a compatible DeepSeek/Meta API key) and a Jina API Key (for search/embeddings).

### Step-by-Step Local Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/arnold2309/HIRO_SOCIAL_GOVERNANCE.git
   cd HIRO_SOCIAL_GOVERNANCE
   ```

2. **Configure Environment Variables:**
   - Open the `.env` file in the root directory.
   - Insert your API keys:
     ```env
     OPENAI_API_KEY=your_api_key_here
     JINA_API_KEY=your_jina_key_here
     ```

3. **Start the Project using Docker (Recommended):**
   ```bash
   docker-compose up --build
   ```
   *This automatically builds both the frontend and backend, serving the application on `http://localhost:7860`.*

---

## 🗺️ 3. Feature Walkthrough (The User Journey)

When you open the application, you are guided through a **5-Step Process**. Here is how to explain each feature during your demo:

### Step 1: Knowledge Graph Building 🧠
- **What you do:** You start by uploading a document (PDF, TXT) or pasting text describing a specific scenario or policy (e.g., "Implementing a 4-day work week in a tech hub").
- **What the AI does:** The backend parses the text and uses the LLM to extract key "Entities" (e.g., Employees, Managers, Tech Companies) and the "Relationships" between them. 
- **The Result:** It visually constructs a **Knowledge Graph**. You can actually see the nodes and connections on the screen. This graph serves as the foundational worldview for the AI agents.

### Step 2: Environment Configuration ⚙️
- **What you do:** Here, you fine-tune the parameters of the simulation.
- **Features:** 
  - **Sentiment Intensity:** How emotionally charged the agents should be (e.g., highly reactive vs. calm).
  - **Simulation Steps:** How long the simulation should run.
  - **Platform Choice:** You can choose to simulate the interactions on a virtual "Reddit" (long-form discussions) or "Twitter" (fast-paced, short reactions).
- **Why it matters:** This allows you to test the exact same policy under different societal conditions (e.g., during a period of high social tension vs. stability).

### Step 3: The Simulation Engine 🔄
- **What you do:** You hit "Start Simulation" and watch the magic happen.
- **What the AI does:** 
  1. **Persona Generation:** Using the Knowledge Graph, the system generates diverse AI agents (Personas) with unique backgrounds, biases, and goals.
  2. **Interaction:** These agents are dropped into the chosen platform (Reddit/Twitter) using **CAMEL-OASIS**. They begin posting, replying, agreeing, and arguing with each other based on their programmed personas.
  3. **Memory:** Agents use **Mem0** to remember past interactions, meaning a user who got argued with in step 1 might hold a grudge in step 10!
- **The Result:** A live feed of synthetic social media demonstrating how the public would genuinely react to the policy.

### Step 4: Analytical Report 📊
- **What you do:** Once the simulation ends, you review the AI-generated report.
- **What the AI does:** An analytical "Report Agent" reads through the entire history of the simulated social media feed. It identifies key trends, major points of friction, overall sentiment (Positive/Negative), and potential risks.
- **Why it matters:** Policymakers don't have to read thousands of simulated tweets; they get a clean, actionable executive summary of the societal impact.

### Step 5: Interactive Chat 💬
- **What you do:** You can literally chat with the simulation.
- **What the AI does:** You can ask the system questions like, *"Why were the managers so angry about the 4-day work week?"* The AI will query the generated knowledge graph and the simulation memory to give you an accurate answer based *only* on what happened in your specific simulation.

---

## 🌍 4. Deployment Architecture

When explaining how robust your project is, you can highlight your Hugging Face Space deployment:

- **Hugging Face Spaces:** The entire application (Frontend + Backend) is containerized using Docker and hosted live on Hugging Face.
- **Seamless Integration:** By using a multi-stage `Dockerfile`, the Vue.js frontend is compiled into static files and served directly by the Python Flask backend, requiring only a single port (`7860`) to run flawlessly in the cloud.
- **Git LFS (Large File Storage):** We utilized Git LFS to seamlessly handle large database files (`.db`, `.sqlite`) and static image assets, bypassing standard repository limits.

---

## 🎯 5. Conclusion for the Judges
*"HIRO is not just a chatbot; it is a complex, multi-agent society in a box. By combining Knowledge Graphs for world-building, CAMEL-AI for dynamic roleplaying, and Mem0 for persistent memory, we've created a tool that allows leaders to test the social consequences of their decisions before they make them. HIRO turns unpredictable public reaction into quantifiable, analyzable data."*

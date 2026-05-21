"""
Practical RAG Implementation Guide

Real production patterns for using RAG effectively.
"""

# =============================================================================
# REAL-WORLD RAG PATTERNS
# =============================================================================

"""
PATTERN 1: User-Specific RAG
─────────────────────────────────

For each user, maintain their own RAG context.

┌─────────────────────────────────────────────────────────────────┐
│                        USER-SPECIFIC RAG                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User A (Contractor)              User B (Customer)             │
│  ┌─────────────────────┐          ┌─────────────────────┐       │
│  │ • Their contract     │          │ • Their order       │       │
│  │ • Their invoices    │          │ • Their tickets     │       │
│  │ • Their agreements   │          │ • Their profile     │       │
│  │                     │          │                     │       │
│  │ Total: ~10 docs    │          │ Total: ~5 docs    │       │
│  └─────────────────────┘          └─────────────────────┘       │
│           │                                │                       │
│           ▼                                ▼                       │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                    RAG QUERY                            │     │
│  │                                                          │     │
│  │  "What are my delivery terms?"                         │     │
│  │                                                          │     │
│  │  → Retrieves from User A's contract ONLY               │     │
│  │                                                          │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Implementation:
"""

class UserSpecificRAG:
    """RAG scoped to individual user data."""
    
    def __init__(self):
        self.user_rags: dict[str, ProductionRAG] = {}
    
    def add_user_document(self, user_id: str, document: str, metadata: dict):
        """Add document to user's RAG."""
        if user_id not in self.user_rags:
            self.user_rags[user_id] = ProductionRAG()
        
        self.user_rags[user_id].add_document(
            content=document,
            metadata=metadata,
        )
    
    def query_user(self, user_id: str, question: str) -> dict:
        """Query only user's documents."""
        if user_id not in self.user_rags:
            return {"error": "No documents for user"}
        
        return self.user_rags[user_id].query(question)


# =============================================================================
# PATTERN 2: TEMPORAL RAG (Recent First)
# =============================================================================

"""
PATTERN 2: Temporal Relevance
─────────────────────────────────

Filter by recency - most recent docs are most relevant.

┌─────────────────────────────────────────────────────────────────┐
│                    TEMPORAL RAG                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query: "What's our policy on remote work?"                   │
│                                                                  │
│  Time Filter: Last 6 months                                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. HR Policy 2024-03.pdf    (March 2024)  ← RECENT │  │
│  │     "Remote work is allowed 3 days per week..."        │  │
│  │                                                          │  │
│  │  2. Company Handbook 2024-01.pdf  (Jan 2024)          │  │
│  │     "Standard office hours are 9-5..."                 │  │
│  │                                                          │  │
│  │  3. Old Policy 2023.pdf        (2023)     ← SKIP     │  │
│  │     "All employees must work on-site..."               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Result: "Remote work is allowed 3 days per week..."       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Implementation:
"""

class TemporalRAG:
    """RAG with temporal filtering."""
    
    def __init__(self):
        self.documents: list[tuple[str, str, float]] = []  # (content, metadata, timestamp)
    
    def add_document(self, document: str, metadata: dict, timestamp: float):
        """Add document with timestamp."""
        self.documents.append((document, metadata, timestamp))
    
    def query_recent(self, question: str, days_back: int = 180) -> list:
        """Query only recent documents."""
        import time
        cutoff = time.time() - (days_back * 86400)
        
        recent = [
            (doc, meta, ts) 
            for doc, meta, ts in self.documents 
            if ts >= cutoff
        ]
        
        # Sort by recency
        recent.sort(key=lambda x: -x[2])
        
        return recent[:5]  # Top 5 recent


# =============================================================================
# PATTERN 3: MULTI-HOP RAG
# =============================================================================

"""
PATTERN 3: Multi-Hop Reasoning
─────────────────────────────────

RAG that chains through multiple documents.

┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-HOP RAG                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query: "Can the contractor charge for changes?"                │
│                                                                  │
│  HOP 1: Find relevant contract                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Contract: "Contractor may charge for changes with      │  │
│  │  written approval from project manager."                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  HOP 2: Find approval policy                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Policy: "Project manager approval requires:            │  │
│  │  1. Written request                                     │  │
│  │  2. Cost estimate                                       │  │
│  │  3. Timeline impact assessment"                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  HOP 3: Synthesize answer                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Answer: "Yes, the contractor can charge for changes    │  │
│  │  if they have written approval from the project manager,  │  │
│  │  which requires a written request, cost estimate, and     │  │
│  │  timeline impact assessment."                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
"""

class MultiHopRAG:
    """RAG that chains through multiple document types."""
    
    def __init__(self):
        self.collections: dict[str, ProductionRAG] = {}
    
    def add_collection(self, name: str, documents: list[str]):
        """Add a collection of related documents."""
        self.collections[name] = ProductionRAG()
        for doc in documents:
            self.collections[name].add_document(doc)
    
    def multi_hop_query(self, query: str) -> dict:
        """Query across multiple hops."""
        # Hop 1: Find primary context
        primary = self.query_primary(query)
        
        # Hop 2: Use result to find secondary context
        secondary = self.query_secondary(primary["answer"])
        
        return {
            "primary_context": primary["context"],
            "secondary_context": secondary["context"],
            "answer": self.synthesize(primary, secondary),
        }


# =============================================================================
# PATTERN 4: AGENTIC RAG
# =============================================================================

"""
PATTERN 4: Agentic RAG (Self-Deciding Retrieval)
────────────────────────────────────────────────────

The LLM decides WHEN and WHAT to retrieve.

┌─────────────────────────────────────────────────────────────────┐
│                      AGENTIC RAG                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User: "Book me a flight to NYC"                               │
│                                                                  │
│  LLM thinks: "I need date info → I'll check calendar first"   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ACTION: retrieve_documents                            │  │
│  │  QUERY: "user calendar availability next week"         │  │
│  │  RESULT: "You're free Mon-Fri next week"              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                    │                                              │
│                    ▼                                              │
│  LLM: "I see you're free next week. Which day for NYC?"     │
│                                                                  │
│  User: "Tuesday"                                              │
│                                                                  │
│  LLM thinks: "Now I need flight options"                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ACTION: browse_web                                    │  │
│  │  QUERY: "flights to NYC on Tuesday"                   │  │
│  │  RESULT: [Flight options with prices]                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
"""

class AgenticRAG:
    """RAG with agentic decision making."""
    
    # Tool definitions for the agent
    TOOLS = [
        {
            "name": "retrieve_documents",
            "description": "Search internal documents for relevant information",
            "parameters": {"query": "string"},
        },
        {
            "name": "browse_web", 
            "description": "Search the web for current information",
            "parameters": {"query": "string"},
        },
        {
            "name": "query_database",
            "description": "Query structured database for specific data",
            "parameters": {"query": "string"},
        },
    ]
    
    def decide_and_retrieve(self, user_message: str, history: list) -> dict:
        """LLM decides what to retrieve."""
        # In production: Use LLM to decide tool + query
        # Here: simple rule-based demo
        
        if "calendar" in user_message.lower() or "available" in user_message.lower():
            return {
                "tool": "retrieve_documents",
                "query": "calendar availability",
                "result": "User is free Mon-Fri next week",
            }
        
        if "flight" in user_message.lower() or "fly" in user_message.lower():
            return {
                "tool": "browse_web",
                "query": user_message,
                "result": "[Flight options would go here]",
            }
        
        return {
            "tool": None,
            "query": None,
            "result": "No retrieval needed",
        }


# =============================================================================
# PATTERN 5: RAG WITH MEMORY
# =============================================================================

"""
PATTERN 5: Conversational RAG
───────────────────────────────

RAG that remembers conversation context.

┌─────────────────────────────────────────────────────────────────┐
│                    CONVERSATIONAL RAG                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Turn 1:                                                        │
│  User: "What's in my contract?"                                 │
│  RAG: Retrieves contract → "You have a fixed-price contract"   │
│  Memory: [contract_context]                                    │
│                                                                  │
│  Turn 2:                                                        │
│  User: "What about overtime?"                                  │
│  RAG: [contract_context] + "overtime" → "Overtime is..."     │
│  Memory: [contract_context, overtime_clause]                   │
│                                                                  │
│  Turn 3:                                                        │
│  User: "Can I expense meals?"                                  │
│  RAG: [all_context] + "expense meals" → "Meal expenses..."    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
"""

class ConversationalRAG:
    """RAG with conversation memory."""
    
    def __init__(self, rag: ProductionRAG, memory_limit: int = 5):
        self.rag = rag
        self.memory: list[str] = []
        self.memory_limit = memory_limit
    
    def query_with_memory(self, question: str) -> dict:
        """Query with conversation context."""
        # Build context from memory
        context = "\n".join(self.memory)
        
        # Enhance query with memory context
        enhanced_query = f"{context}\n\nQuestion: {question}"
        
        # Retrieve
        result = self.rag.query(enhanced_query)
        
        # Update memory
        self.memory.append(result.get("context", "")[:500])
        if len(self.memory) > self.memory_limit:
            self.memory = self.memory[-self.memory_limit:]
        
        return result


# =============================================================================
# REAL PRODUCTION EXAMPLE
# =============================================================================

"""
COMPLETE PRODUCTION EXAMPLE:
Law Firm Contract Assistant
"""

class ContractAssistant:
    """
    Production RAG for contract review.
    
    Real data:
    - ~5-20 contracts per client
    - ~10-50 pages per contract
    - Specific queries like "What are termination terms?"
    
    NOT: 100,000 random documents.
    """
    
    def __init__(self):
        self.client_rags: dict[str, ProductionRAG] = {}
        self.global_knowledge = ProductionRAG()  # Legal glossary, templates
    
    def setup_client(self, client_id: str, contracts: list[dict]):
        """Setup RAG for a client with their contracts."""
        self.client_rags[client_id] = ProductionRAG()
        
        for contract in contracts:
            self.client_rags[client_id].add_document(
                content=contract["text"],
                metadata={
                    "source": contract["name"],
                    "type": "contract",
                    "date": contract.get("date"),
                },
            )
        
        # Add global legal knowledge
        self.global_knowledge.add_document(
            content="Legal glossary and standard terms...",
            metadata={"type": "glossary"},
        )
    
    def answer_question(self, client_id: str, question: str) -> dict:
        """Answer question about client's contracts."""
        # 1. Search client's contracts
        client_results = self.client_rags[client_id].query(question)
        
        # 2. Enrich with legal knowledge
        legal_results = self.global_knowledge.query(question)
        
        # 3. Synthesize answer
        answer = f"""
Based on {client_id}'s contract(s):

{client_results['context'][:1000]}

Relevant legal definitions:
{legal_results['context'][:500]}

Please review the full documents for complete details.
"""
        
        # 4. Verify
        verification = self.client_rags[client_id].verify_and_ground(
            answer, question
        )
        
        return {
            "answer": answer,
            "confidence": verification.get("confidence", 0.5),
            "sources": client_results.get("results", [])[:3],
            "verified": verification.get("is_verified", False),
        }


# =============================================================================
# SUMMARY: WHEN TO USE RAG
# =============================================================================

"""
┌─────────────────────────────────────────────────────────────────┐
│                     RAG DECISION GUIDE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  USE RAG WHEN:                                                  │
│  ✓ Private/specific data (your documents, not internet)        │
│  ✓ Frequently changing data (daily updates)                      │
│  ✓ User-specific context (their data, not everyone's)          │
│  ✓ Need citations/provenance                                    │
│  ✓ Regulatory compliance (must show source)                    │
│                                                                  │
│  DON'T USE RAG WHEN:                                           │
│  ✗ General knowledge (LLM already knows)                      │
│  ✗ Static facts that never change                              │
│  ✗ Code generation (use training/fine-tuning)                  │
│  ✗ Simple Q&A with obvious answers                              │
│                                                                  │
│  RAG DATA SCALE:                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Use Case              | Recommended Docs | Update      │  │
│  │  ─────────────────────────────────────────────────────│  │
│  │  Personal assistant    | 1-10 docs     | Real-time   │  │
│  │  Business contracts    | 10-100 docs   | Per deal    │  │
│  │  Product docs          | 100-1000 docs | Per release  │  │
│  │  Company knowledge base| 1000-10000    | Weekly       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  BOTTOM LINE:                                                  │
│  RAG = Selective context injection, NOT database dump.          │
│  Typical: 1-100 docs per query, not 100,000.                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
"""


__all__ = [
    "UserSpecificRAG",
    "TemporalRAG",
    "MultiHopRAG",
    "AgenticRAG",
    "ConversationalRAG",
    "ContractAssistant",
]

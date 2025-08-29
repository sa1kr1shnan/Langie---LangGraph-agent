!pip install langgraph langchain langchain-openai
import os
import os
import uuid
from typing import Annotated, TypedDict, List
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# Mock MCP tools (replacing with langchain-mcp-adapters for real Atlas/Common servers)
@tool
def parse_request_text(text: str) -> dict:
    """Common MCP: Parse unstructured text to structured data."""
    print("LOG: Calling parse_request_text (COMMON MCP)")
    return {"query": text, "product": "app", "account": "user123"}

@tool
def extract_entities(data: dict) -> dict:
    """Atlas MCP: Extract product, account, dates."""
    print("LOG: Calling extract_entities (ATLAS MCP)")
    return {"product": data.get("product", "unknown"), "account": data.get("account", "unknown"), "dates": "2025-08-28"}

@tool
def normalize_fields(data: dict) -> dict:
    """Common MCP: Standardize dates, codes, IDs."""
    print("LOG: Calling normalize_fields (COMMON MCP)")
    return {"normalized_date": "2025-08-28", "ticket_id": data.get("ticket_id", "T123")}

@tool
def enrich_records(ticket_id: str) -> dict:
    """Atlas MCP: Add SLA, historical ticket info."""
    print("LOG: Calling enrich_records (ATLAS MCP)")
    return {"sla": "24h", "history": f"Ticket {ticket_id}: 1 prior issue"}

@tool
def add_flags_calculations(data: dict) -> dict:
    """Common MCP: Compute priority or SLA risk."""
    print("LOG: Calling add_flags_calculations (COMMON MCP)")
    return {"priority": "high" if "urgent" in data.get("query", "") else data.get("priority", "medium")}

@tool
def clarify_question(data: dict) -> str:
    """Atlas MCP: Request missing info."""
    print("LOG: Calling clarify_question (ATLAS MCP)")
    return "Please provide error code or billing details."

@tool
def extract_answer(response: str) -> str:
    """Atlas MCP: Capture concise user response."""
    print("LOG: Calling extract_answer (ATLAS MCP)")
    return response[:100]

@tool
def knowledge_base_search(query: str) -> str:
    """Atlas MCP: Lookup KB or FAQ."""
    print("LOG: Calling knowledge_base_search (ATLAS MCP)")
    return f"KB result for '{query}': Restart app."

@tool
def solution_evaluation(data: dict) -> int:
    """Common MCP: Score potential solutions (1-100)."""
    print("LOG: Calling solution_evaluation (COMMON MCP)")
    return 85 if "bug" in data.get("query", "") else 95

@tool
def escalation_decision(score: int) -> str:
    """Atlas MCP: Decide escalation if score <90."""
    print("LOG: Calling escalation_decision (ATLAS MCP)")
    return "escalate" if score < 90 else "resolve"

@tool
def update_ticket(ticket_id: str, status: str) -> str:
    """Atlas MCP: Update ticket status."""
    print("LOG: Calling update_ticket (ATLAS MCP)")
    return f"Ticket {ticket_id} updated to {status}."

@tool
def close_ticket(ticket_id: str) -> str:
    """Atlas MCP: Close ticket."""
    print("LOG: Calling close_ticket (ATLAS MCP)")
    return f"Ticket {ticket_id} closed."

@tool
def response_generation(data: dict) -> str:
    """Common MCP: Draft customer reply."""
    print("LOG: Calling response_generation (COMMON MCP)")
    if data.get("decision") == "escalate":
        return f"Dear {data.get('customer_name', 'Customer')}, your issue has been escalated."
    return f"Dear {data.get('customer_name', 'Customer')}, your issue is resolved."

@tool
def execute_api_calls(data: dict) -> str:
    """Atlas MCP: Trigger CRM/order system actions."""
    print("LOG: Calling execute_api_calls (ATLAS MCP)")
    return "CRM updated."

@tool
def trigger_notifications(data: dict) -> str:
    """Atlas MCP: Notify customer."""
    print("LOG: Calling trigger_notifications (ATLAS MCP)")
    return "Notification sent."

# Initializing LLM and bind tools
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
tools = [
    parse_request_text, extract_entities, normalize_fields, enrich_records,
    add_flags_calculations, clarify_question, extract_answer, knowledge_base_search,
    solution_evaluation, escalation_decision, update_ticket, close_ticket,
    response_generation, execute_api_calls, trigger_notifications
]
llm_with_tools = llm.bind_tools(tools)

# Stating definition for persistence
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    customer_name: str
    email: str
    query: str
    priority: str
    ticket_id: str
    product: str
    account: str
    dates: str
    normalized_date: str
    sla: str
    history: str
    user_response: str
    kb_result: str
    solution_score: int
    decision: str
    ticket_status: str
    customer_reply: str
    api_result: str
    notification: str

# Node functions for each stage
def intake_node(state: State):
    """Stage 1: INTAKE - Capture initial payload"""
    print("LOG: Executing INTAKE (deterministic) - accept_payload")
    return {
        "customer_name": state.get("customer_name", "John Doe"),
        "email": state.get("email", "john@example.com"),
        "query": state.get("query", state["messages"][-1].content),
        "priority": state.get("priority", "medium"),  # Use input priority
        "ticket_id": state.get("ticket_id", "T123")
    }

def understand_node(state: State):
    """Stage 2: UNDERSTAND - Parse query, extract entities via MCP"""
    print("LOG: Executing UNDERSTAND (deterministic)")
    parsed = parse_request_text.invoke({"text": state["query"]})
    entities = extract_entities.invoke({"data": parsed})
    return {**parsed, **entities}

def prepare_node(state: State):
    """Stage 3: PREPARE - Normalize and enrich data"""
    print("LOG: Executing PREPARE (deterministic)")
    normalized = normalize_fields.invoke({"data": {"ticket_id": state["ticket_id"]}})
    enriched = enrich_records.invoke({"ticket_id": state["ticket_id"]})
    flags = add_flags_calculations.invoke({"data": {"query": state["query"], "priority": state["priority"]}})
    return {**normalized, **enriched, **flags}

def ask_node(state: State):
    """Stage 4: ASK - Request clarification"""
    print("LOG: Executing ASK (deterministic)")
    question = clarify_question.invoke({"data": state})
    print(f"LOG: Clarification question: {question}")
    return {"messages": [SystemMessage(content=question)]}

def wait_node(state: State):
    """Stage 5: WAIT - Capture and store user response"""
    print("LOG: Executing WAIT (deterministic)")
    response = "Error code: 404"  # Mock for demo; swap with input() for interactive
    # response = input("Enter response to clarification (e.g., 'Error code: 404'): ")
    extracted = extract_answer.invoke({"response": response})
    return {"user_response": extracted}

def retrieve_node(state: State):
    """Stage 6: RETRIEVE - Search KB and store results"""
    print("LOG: Executing RETRIEVE (deterministic)")
    kb_result = knowledge_base_search.invoke({"query": state["query"]})
    return {"kb_result": kb_result}

def decide_node(state: State):
    """Stage 7: DECIDE - Evaluate solutions and decide escalation"""
    print("LOG: Executing DECIDE (non-deterministic)")
    score = solution_evaluation.invoke({"data": state})
    decision = escalation_decision.invoke({"score": score})
    print(f"LOG: Solution score: {score}, Decision: {decision}")
    return {"solution_score": score, "decision": decision}

def update_node(state: State):
    """Stage 8: UPDATE - Update ticket status"""
    print("LOG: Executing UPDATE (deterministic)")
    status = update_ticket.invoke({"ticket_id": state["ticket_id"], "status": state["decision"]})
    if state["decision"] == "resolve":
        close_msg = close_ticket.invoke({"ticket_id": state["ticket_id"]})
        return {"ticket_status": f"{status}; {close_msg}"}
    return {"ticket_status": status}

def create_node(state: State):
    """Stage 9: CREATE - Draft customer reply"""
    print("LOG: Executing CREATE (deterministic)")
    reply = response_generation.invoke({"data": state})
    return {"customer_reply": reply}

def do_node(state: State):
    """Stage 10: DO - Execute API calls and notify"""
    print("LOG: Executing DO (deterministic)")
    api_result = execute_api_calls.invoke({"data": state})
    notif = trigger_notifications.invoke({"data": state})
    return {"api_result": api_result, "notification": notif}

def complete_node(state: State):
    """Stage 11: COMPLETE - Output final payload"""
    print("LOG: Executing COMPLETE (deterministic) - output_payload")
    final_payload = {
        "customer_name": state["customer_name"],
        "email": state["email"],
        "query": state["query"],
        "priority": state["priority"],
        "ticket_id": state["ticket_id"],
        "product": state["product"],
        "account": state["account"],
        "dates": state["dates"],
        "normalized_date": state["normalized_date"],
        "sla": state["sla"],
        "history": state["history"],
        "user_response": state["user_response"],
        "kb_result": state["kb_result"],
        "solution_score": state["solution_score"],
        "decision": state["decision"],
        "ticket_status": state["ticket_status"],
        "customer_reply": state["customer_reply"],
        "api_result": state["api_result"],
        "notification": state["notification"]
    }
    print(f"LOG: Final Payload: {final_payload}")
    return {"messages": [SystemMessage(content=str(final_payload))]}

# Building the graph
workflow = StateGraph(State)
workflow.add_node("intake", intake_node)
workflow.add_node("understand", understand_node)
workflow.add_node("prepare", prepare_node)
workflow.add_node("ask", ask_node)
workflow.add_node("wait", wait_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("decide", decide_node)
workflow.add_node("update", update_node)
workflow.add_node("create", create_node)
workflow.add_node("do", do_node)
workflow.add_node("complete", complete_node)

# Defining edges
workflow.add_edge(START, "intake")
workflow.add_edge("intake", "understand")
workflow.add_edge("understand", "prepare")
workflow.add_edge("prepare", "ask")
workflow.add_edge("ask", "wait")
workflow.add_edge("wait", "retrieve")
workflow.add_edge("retrieve", "decide")
workflow.add_conditional_edges("decide", lambda state: state["decision"], {"escalate": "update", "resolve": "update"})
workflow.add_edge("update", "create")
workflow.add_edge("create", "do")
workflow.add_edge("do", "complete")
workflow.add_edge("complete", END)

# Compile with MemorySaver for state persistence
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# Demo run with sample input
initial_state = {
    "messages": [HumanMessage(content="My dig app crashed with a bug")],
    "customer_name": "Alan Grant",
    "email": "grant@paleo.org",
    "query": "My dig app crashed with a bug",
    "priority": "high",
    "ticket_id": "DIG-001"
}
thread = {"configurable": {"thread_id": str(uuid.uuid4())}}  # Unique thread ID
for event in app.stream(initial_state, thread, stream_mode="values"):
    print(event)

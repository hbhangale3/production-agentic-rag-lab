"""Minimal LangGraph topology and future dependency-injection boundary."""

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from src.services.agent.config import AgentGraphConfig
from src.services.agent.events import AgentExecutionEvent, AgentExecutionStatus
from src.services.agent.state import AgentState
from src.services.agent.types import TerminalReason


@dataclass(frozen=True)
class AgentGraphDependencies:
    """Foundation dependency boundary; future milestones add typed services."""


def _initialize_foundation(state: AgentState) -> dict[str, Any]:
    sequence = len(state["execution_events"])
    return {
        "execution_events": [
            AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_STARTED, sequence=sequence),
            AgentExecutionEvent(status=AgentExecutionStatus.GRAPH_COMPLETED, sequence=sequence + 1),
        ],
        "terminal_reason": TerminalReason.COMPLETED,
    }


def build_agent_graph(
    *,
    config: AgentGraphConfig | None = None,
    dependencies: AgentGraphDependencies | None = None,
) -> CompiledStateGraph:
    """Compile the foundation-only graph without creating external services."""

    graph_config = config or AgentGraphConfig()
    graph_dependencies = dependencies or AgentGraphDependencies()
    _ = (graph_config, graph_dependencies)

    builder = StateGraph(AgentState)
    builder.add_node("initialize_foundation", _initialize_foundation)
    builder.add_edge(START, "initialize_foundation")
    builder.add_edge("initialize_foundation", END)
    return builder.compile()

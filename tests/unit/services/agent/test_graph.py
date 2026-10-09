from src.services.agent import (
    AgentExecutionEvent,
    AgentExecutionStatus,
    TerminalReason,
    build_agent_graph,
    create_initial_agent_state,
)


def test_foundation_graph_compiles_and_invokes_without_services() -> None:
    graph = build_agent_graph()
    initial = create_initial_agent_state("How can AI improve healthcare access?")

    result = graph.invoke(initial)

    assert result["original_question"] == initial["original_question"]
    assert result["current_query"] == initial["current_query"]
    assert result["retrieval_attempts"] == 0
    assert result["local_retrieval_result"] is None
    assert result["generated_answer"] is None
    assert result["terminal_reason"] == TerminalReason.COMPLETED
    assert [(event.status, event.sequence) for event in result["execution_events"]] == [
        (AgentExecutionStatus.GRAPH_STARTED, 0),
        (AgentExecutionStatus.GRAPH_COMPLETED, 1),
    ]


def test_execution_event_reducer_accumulates_in_graph_order() -> None:
    graph = build_agent_graph()
    initial = create_initial_agent_state("Question")
    initial["execution_events"] = [
        AgentExecutionEvent(status=AgentExecutionStatus.GUARDRAIL_STARTED, sequence=0)
    ]

    result = graph.invoke(initial)

    assert [event.status for event in result["execution_events"]] == [
        AgentExecutionStatus.GUARDRAIL_STARTED,
        AgentExecutionStatus.GRAPH_STARTED,
        AgentExecutionStatus.GRAPH_COMPLETED,
    ]
    assert [event.sequence for event in result["execution_events"]] == [0, 1, 2]


def test_graph_compilation_is_deterministic_and_has_foundation_topology() -> None:
    first = build_agent_graph().get_graph()
    second = build_agent_graph().get_graph()

    assert set(first.nodes) == {"__start__", "initialize_foundation", "__end__"}
    assert set(first.nodes) == set(second.nodes)
    assert {(edge.source, edge.target) for edge in first.edges} == {
        ("__start__", "initialize_foundation"),
        ("initialize_foundation", "__end__"),
    }

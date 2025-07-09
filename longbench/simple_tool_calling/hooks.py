import json
import logging
from typing import Any, Generic, TypeVar

from agents import Agent, FunctionTool, RunContextWrapper, Tool

T = TypeVar("T")

logger = logging.getLogger(__name__)


class DefaultHooks(Generic[T]):
    """Base hook implementation for emitting status messages during agent lifecycle events."""

    async def on_agent_start(
        self,
        ctx: RunContextWrapper[T],
        agent: Agent[T],
    ) -> None:
        """Called when the agent is being started."""
        logger.info(f"Starting {agent.name} (on_agent_start hook)")

    async def on_agent_end(
        self,
        ctx: RunContextWrapper[T],
        agent: Agent[T],
        output: Any,
    ) -> None:
        """Called when the agent is being ended."""
        logger.info(f"{agent.name} completed (on_agent_end hook)")

    async def on_start(
        self,
        ctx: RunContextWrapper[T],
        agent: Agent[T],
    ) -> None:
        """Called before the agent is invoked."""
        logger.info(f"Starting {agent.name} (on_start hook)")

    async def on_end(
        self,
        ctx: RunContextWrapper[T],
        agent: Agent[T],
        output: Any,
    ) -> None:
        """Called when the agent produces a final output."""
        logger.info(f"{agent.name} completed (on_end hook)")

    async def on_handoff(
        self,
        ctx: RunContextWrapper[T],
        agent: Agent[T],
        source: Agent[T],
    ) -> None:
        """Called when the agent is being handed off to."""
        logger.info(f"Handing off from {source.name} to {agent.name} (on_handoff hook)")

    async def on_tool_start(
        self,
        ctx: RunContextWrapper[T],
        agent: Agent[T],
        tool: Tool,
    ) -> None:
        """Called before a tool is invoked."""
        logger.info(f"{agent.name} using tool: {tool.name} (on_tool_start hook)")

    async def on_tool_end(
        self,
        ctx: RunContextWrapper[T],
        agent: Agent[T],
        tool: Tool,
        result: str,
    ) -> None:
        """Called after a tool is invoked."""
        logger.info(f"{agent.name} completed tool: {tool.name} (on_tool_end hook)")
        logger.info("-" * 100)
        logger.info(f"{result[:1000]}")
        logger.info("-" * 100)


def create_logging_tool_wrapper(original_tool: FunctionTool) -> FunctionTool:
    """Create a wrapper around a FunctionTool that logs the arguments."""

    original_on_invoke = original_tool.on_invoke_tool

    async def logging_on_invoke(ctx, input_args: str) -> Any:
        # Log the arguments before calling the original tool
        logger.info(f"Tool '{original_tool.name}' called with arguments: {input_args}")

        # Parse arguments for prettier logging
        try:
            parsed_args = json.loads(input_args) if input_args else {}
            logger.info(f"Parsed arguments: {parsed_args}")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse arguments as JSON: {input_args}")

        # Call the original tool
        result = await original_on_invoke(ctx, input_args)

        # Log the result
        # logger.info(f"Tool '{original_tool.name}' returned: {result}")

        return result

    # Create a new FunctionTool with the logging wrapper
    return FunctionTool(
        name=original_tool.name,
        description=original_tool.description,
        params_json_schema=original_tool.params_json_schema,
        on_invoke_tool=logging_on_invoke,
        strict_json_schema=original_tool.strict_json_schema,
        is_enabled=original_tool.is_enabled,
    )

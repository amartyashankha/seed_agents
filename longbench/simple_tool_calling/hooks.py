import logging
from typing import Any, Generic, TypeVar

from agents import Agent, RunContextWrapper, Tool

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
        logger.info(f"{result}")
        logger.info("-" * 100)

import asyncio
import re
import types
import uuid
from typing import Any, ClassVar

import httpx
from decouple import config

from src.utils.logger import create_logger

logger = create_logger(level="DEBUG")

HTTP_STATUS_NOT_FOUND = 404


class Config:
    """Centralized configuration for the application."""

    BACKEND_BASE_URL = config("BACKEND_BASE_URL")
    CORE_INTERNAL_API_TOKEN: ClassVar[str] = config("CORE_INTERNAL_API_TOKEN")

    HEADERS: ClassVar[dict[str, str]] = {
        "X-API-KEY": CORE_INTERNAL_API_TOKEN,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


# --- API Client ---
class APIClient:
    """Handles all API interactions with the backend."""

    def __init__(self) -> None:
        # TODO: Change the verify to True for enabling SSL Certificate verification
        # nosec B501
        self._client = httpx.AsyncClient(verify=False, follow_redirects=True, timeout=30.0, headers=Config.HEADERS)

    async def __aenter__(self) -> httpx.AsyncClient:
        return self._client

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """
        Exit the async context manager, closing the HTTP client.

        Args:
            exc_type: The exception type if raised, otherwise None.
            exc_val: The exception instance if raised, otherwise None.
            exc_tb: The traceback if an exception was raised, otherwise None.
        """
        await self._client.aclose()


# --- Memory Operations ---
class MemoryService:
    """Manages CRUD operations for memories."""

    @staticmethod
    async def fetch_memories(agent_id: str, user_id: str) -> list[dict[str, Any]]:
        """Fetches all memories for a given user and agent."""
        url = f"{Config.BACKEND_BASE_URL}/{agent_id}/users/{user_id}/memory/"
        async with httpx.AsyncClient(
            # TODO: Change the verify to True for enabling SSL Certificate verification
            # nosec B501
            verify=False,
            headers={"X-API-KEY": Config.CORE_INTERNAL_API_TOKEN, "Accept": "application/json"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json().get("data", [])

    @staticmethod
    async def delete_memory_by_id(agent_id: str, user_id: str, mid: str) -> bool:
        """Deletes a single memory by its ID."""
        url = f"{Config.BACKEND_BASE_URL}/{agent_id}/users/{user_id}/memory/{mid}"
        try:
            async with APIClient() as client:
                response = await client.delete(url)
                if response.status_code in {200, 204, 202}:
                    logger.info(f"Successfully deleted memory: {mid}")
                    return True
                if response.status_code == HTTP_STATUS_NOT_FOUND:
                    logger.warning(f"Memory {mid} not found, may have been deleted already.")
                    return True
                logger.error(f"Failed to delete memory {mid}: {response.status_code} - {response.text}")
                return False
        except httpx.TimeoutException:
            logger.error(f"Timeout while deleting memory {mid}")
            return False
        except Exception as e:
            logger.error(f"Error deleting memory {mid}: {e}")
            return False

    @staticmethod
    async def create_memory(agent_id: str, user_id: str, mid: str, data: dict[str, str]) -> dict[str, Any]:
        """Creates a new memory."""
        url = f"{Config.BACKEND_BASE_URL}/{agent_id}/users/{user_id}/memory/"
        payload = {"memory_id": mid, "personalized_memory_context": data}
        async with APIClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            logger.info(f"Memory created: {response.json()}")
            return response.json()

    @staticmethod
    async def update_memory(agent_id: str, user_id: str, mid: str, data: dict[str, str]) -> dict[str, Any]:
        """Updates an existing memory."""
        url = f"{Config.BACKEND_BASE_URL}/{agent_id}/users/{user_id}/memory/{mid}/"
        payload = {"memory_id": mid, "personalized_memory_context": data}
        async with APIClient() as client:
            response = await client.put(url, json=payload)
            response.raise_for_status()
            logger.info(f"Memory updated: {response.json()}")
            return response.json()

    @staticmethod
    async def delete_all_memories(agent_id: str, user_id: str) -> bool:
        """Deletes all memories for a user, using a bulk endpoint if available, otherwise individually."""
        bulk_url = f"{Config.BACKEND_BASE_URL}/{agent_id}/users/{user_id}/memory/all"
        try:
            async with APIClient() as client:
                response = await client.delete(bulk_url)
                if response.status_code in {200, 204, 202}:
                    logger.info("Successfully deleted all memories using bulk endpoint.")
                    return True
                logger.warning(f"Bulk delete failed ({response.status_code}), falling back to individual deletion.")
                return await MemoryService._delete_all_individually(agent_id, user_id)
        except Exception as e:
            logger.error(f"Error with bulk delete, falling back to individual deletion: {e}")
            return await MemoryService._delete_all_individually(agent_id, user_id)

    @staticmethod
    async def _delete_all_individually(agent_id: str, user_id: str) -> bool:
        """Helper to delete memories one by one in batches."""
        logger.info(f"Starting to delete all memories for user {user_id}, agent {agent_id} individually.")
        memories = await MemoryService.fetch_memories(agent_id, user_id)
        if not memories:
            logger.info("No memories found to delete.")
            return True

        memory_ids = [
            str(mem.get("memory_id") or mem.get("id")) for mem in memories if mem.get("memory_id") or mem.get("id")
        ]
        if not memory_ids:
            logger.warning("No valid memory IDs found.")
            return False

        deleted_count = 0
        tasks = [MemoryService.delete_memory_by_id(agent_id, user_id, mid) for mid in memory_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if result is True:
                deleted_count += 1
            elif isinstance(result, Exception):
                logger.error(f"Failed to delete a memory due to an exception: {result}")
            else:
                logger.error("Failed to delete a memory.")

        return deleted_count == len(memory_ids)


# --- Command Handlers and Utilities ---
class MemoryManager:
    """
    A unified utility class for detecting, validating, parsing, and executing
    memory-related commands. This class is designed to be a single, stateless
    entry point for all command-driven memory operations.
    """

    _COMMANDS: ClassVar[dict[str, re.Pattern[str]]] = {
        "delete": re.compile(r"^/memory\.delete\s+[\w\s,.]+$"),
        "update": re.compile(r"^/memory\.update\s+([\w\s.]+:\s*[^,]+(\s*,\s*[\w\s.]+:\s*[^,]+)*)?$"),
        "create": re.compile(r"^/memory\.create\s+([\w\s.]+:\s*[^,]+(\s*,\s*[\w\s.]+:\s*[^,]+)*)?$"),
        "get": re.compile(r"^/memory\.get(\s+[\w\s.]+)?$"),
        "clear": re.compile(r"^/memory\.clear$"),
    }

    _HANDLER_MAP: ClassVar[dict[str, str]] = {
        "delete": "_execute_delete_command",
        "update": "_execute_update_command",
        "create": "_execute_create_command",
        "get": "_execute_get_command",
        "clear": "_execute_clear_command",
    }

    @staticmethod
    def _detect_command(input_string: str) -> dict[str, Any] | None:
        """Detects a command and validates its format."""
        input_string = input_string.strip().lower()
        for cmd_type, pattern in MemoryManager._COMMANDS.items():
            if pattern.match(input_string):
                return {"type": cmd_type, "input": input_string}
        return None

    @staticmethod
    def _parse_kv_command(command_string: str, prefix: str) -> dict[str, str]:
        """Parses key-value pairs from a command string."""
        command_body = command_string[len(prefix) :].strip()
        if not command_body:
            return {}

        result = {}
        for part in command_body.split(","):
            if ":" not in part:
                raise ValueError(f"Invalid format '{part}'. Expected 'key:value'.")
            key, value = part.split(":", 1)
            result[key.strip()] = value.strip()
        return result

    @staticmethod
    def _extract_delete_info(user_query: str, summary: list[dict[str, Any]]) -> dict[str, Any]:
        """Extracts memory IDs and key-value pairs to delete from user query."""
        raw_terms = re.sub(r"^/memory\.delete\s*", "", user_query, flags=re.IGNORECASE)
        delete_keys = {term.strip().lower() for term in raw_terms.split(",") if term.strip()}

        matched_ids = []
        matched_pairs = []
        found_keys = set()

        for item in summary:
            context = item.get("personalized_memory_context", {})
            key_original = context.get("Key", "").strip()
            if key_original.lower() in delete_keys:
                matched_ids.append(item.get("memory_id"))
                matched_pairs.append({key_original: context.get("Value", "")})
                found_keys.add(key_original.lower())

        missing_keys = list(delete_keys - found_keys)
        return {"memory_ids": matched_ids, "key_value_pairs": matched_pairs, "missing": missing_keys}

    @staticmethod
    def _format_summary(summary: list[dict[str, Any]]) -> str:
        """Formats memory entries into a readable Markdown table."""
        if not summary:
            return "No summary memory found."

        output = ["## Summary Memory Entries", "| Key | Value |", "|---|---|"]
        for item in summary:
            context = item.get("personalized_memory_context", {})
            key = str(context.get("Key", "N/A")).replace("|", "\\|")
            value = str(context.get("Value", "N/A")).replace("|", "\\|")
            output.append(f"| {key} | {value} |")

        return "\n".join(output)

    @staticmethod
    def _summarize_deletion(result: dict[str, Any]) -> str:
        """Generates a human-readable summary of a deletion operation."""
        deleted_pairs = result.get("key_value_pairs", [])
        missing_keys = result.get("missing", [])

        messages = []
        if deleted_pairs:
            items = [f"'{k}':'{v}'" for pair in deleted_pairs for k, v in pair.items()]
            messages.append(f"The following key-value pairs were deleted: {', '.join(items)}.")
        if missing_keys:
            messages.append(
                "The following keys were missing from the summary: " + ", ".join(f"'{k}'" for k in missing_keys) + "."
            )

        return " ".join(messages).strip() if messages else "No matching memories found to delete."

    @staticmethod
    def _key_exists(summary: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
        """Checks if a key exists in the summary memory and returns its details."""
        key_lower = key.strip().lower()
        for item in summary:
            context = item.get("personalized_memory_context", {})
            if isinstance(context, dict) and context.get("Key", "").strip().lower() == key_lower:
                return {"memory_id": item.get("memory_id"), "value": context.get("Value")}
        return None

    @staticmethod
    def _key_value_exists(summary: list[dict[str, Any]], key: str, value: str) -> bool:
        """Checks if a specific key-value pair exists in the summary memory."""
        key_lower = key.strip().lower()
        value_lower = value.strip().lower()
        for item in summary:
            context = item.get("personalized_memory_context", {})
            if (
                isinstance(context, dict)
                and context.get("Key", "").strip().lower() == key_lower
                and context.get("Value", "").strip().lower() == value_lower
            ):
                return True
        return False

    @staticmethod
    async def _execute_command(
        agent_id: str,
        user_id: str,
        command_info: dict[str, Any],
        summary_memory: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Dispatches the parsed command to the correct handler method.
        This method acts as the central entry point and handles
        generic error wrapping.
        """
        cmd_type = command_info["type"]
        handler_name = MemoryManager._HANDLER_MAP.get(cmd_type)

        if not handler_name:
            return {"command_handled": True, "final_answer": f"Unknown command type: {cmd_type}", "status": "error"}

        try:
            handler_method = getattr(MemoryManager, handler_name)
            return await handler_method(agent_id, user_id, command_info, summary_memory)
        except ValueError as ve:
            return {"command_handled": True, "final_answer": str(ve), "status": "error"}
        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            return {"command_handled": True, "final_answer": f"An unexpected error occurred: {e}", "status": "error"}

    @staticmethod
    async def _execute_delete_command(
        agent_id: str, user_id: str, command_info: dict[str, Any], summary_memory: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Handles the logic for the 'delete' command."""
        command_data = MemoryManager._extract_delete_info(command_info["input"], summary_memory)

        if not command_data["memory_ids"]:
            return {
                "command_handled": True,
                "final_answer": MemoryManager._summarize_deletion(command_data),
                "status": "success",
            }

        tasks = [MemoryService.delete_memory_by_id(agent_id, user_id, mid) for mid in command_data["memory_ids"]]
        await asyncio.gather(*tasks)

        return {
            "command_handled": True,
            "final_answer": MemoryManager._summarize_deletion(command_data),
            "status": "success",
        }

    @staticmethod
    async def _execute_create_command(
        agent_id: str, user_id: str, command_info: dict[str, Any], summary_memory: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Handles the logic for the 'create' command."""
        data = MemoryManager._parse_kv_command(command_info["input"], "/memory.create")
        messages = []

        for key, value in data.items():
            if MemoryManager._key_value_exists(summary_memory, key, value):
                messages.append(f"Key-value pair '{key}':'{value}' already exists.")
            elif MemoryManager._key_exists(summary_memory, key):
                messages.append(f"Key '{key}' already exists. Use /memory.update to change it.")
            else:
                await MemoryService.create_memory(agent_id, user_id, str(uuid.uuid4()), {"Key": key, "Value": value})
                messages.append(f"Memory created for: '{key}':'{value}'.")

        final_answer = " ".join(messages).strip()
        return {"command_handled": True, "final_answer": final_answer, "status": "success"}

    @staticmethod
    async def _execute_update_command(
        agent_id: str,
        user_id: str,
        command_info: dict[str, Any],
        summary_memory: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Handles the logic for the 'update' command."""
        data = MemoryManager._parse_kv_command(command_info["input"], "/memory.update")
        messages = []

        for key, value in data.items():
            key_info = MemoryManager._key_exists(summary_memory, key)
            if key_info:
                await MemoryService.update_memory(
                    agent_id, user_id, key_info["memory_id"], {"Key": key, "Value": value}
                )
                messages.append(f"Memory updated for: '{key}':'{value}'.")
            else:
                messages.append(f"Key '{key}' does not exist to update.")

        final_answer = " ".join(messages).strip()
        return {"command_handled": True, "final_answer": final_answer, "status": "success"}

    @staticmethod
    async def _execute_get_command(
        _agent_id: str,
        _user_id: str,
        command_info: dict[str, Any],
        summary_memory: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Handles the logic for the 'get' command."""
        key_to_get = command_info["input"][len("/memory.get") :].strip() or "all"

        if key_to_get == "all":
            response_text = MemoryManager._format_summary(summary_memory)
        else:
            found_item = MemoryManager._key_exists(summary_memory, key_to_get)
            if found_item:
                response_text = MemoryManager._format_summary([
                    {"personalized_memory_context": {"Key": key_to_get, "Value": found_item["value"]}}
                ])
            else:
                response_text = f"No memory found for key: '{key_to_get}'."

        return {"command_handled": True, "final_answer": response_text, "status": "success"}

    @staticmethod
    async def _execute_clear_command(
        agent_id: str, user_id: str, user_query: str, summary_memory: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Handles the logic for the 'clear' command."""
        logger.info(
            f"{'*' * 20} agent id :{agent_id} user id: {user_id} summary memory: {summary_memory} user query: {user_query} {'*' * 20}"
        )
        await MemoryService.delete_all_memories(agent_id, user_id)
        return {
            "command_handled": True,
            "final_answer": "All summary memories have been cleared.",
            "status": "success",
        }

    @staticmethod
    async def invoke(
        user_query: str, agent_id: str, user_id: str, summary_memory: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        The main entry point for command processing. It handles the entire
        lifecycle from detection to execution.
        """
        command_info = MemoryManager._detect_command(user_query)

        if not command_info:
            return {"command_handled": False, "message": "No command detected."}

        if command_info["type"] == "continue":
            return {
                "command_handled": True,
                "continue_to_nlp": True,
                "cleaned_query": user_query.replace("--", "", 1).strip(),
            }

        if summary_memory is None:
            try:
                summary_memory = await MemoryService.fetch_memories(agent_id, user_id)
            except Exception as e:
                return {"command_handled": True, "final_answer": f"Error accessing memory: {e}", "status": "error"}

        try:
            return await MemoryManager._execute_command(agent_id, user_id, command_info, summary_memory)
        except Exception as e:
            return {"command_handled": True, "final_answer": f"An unexpected error occurred: {e}", "status": "error"}

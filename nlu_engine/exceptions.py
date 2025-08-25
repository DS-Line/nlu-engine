class UserQueryNotFoundError(Exception):
    """Raised when a user query is not found in the system."""


class AgentIDNotFoundError(Exception):
    """Raised when the specified agent ID does not exist."""


class DatabaseConfigNotFoundError(Exception):
    """Raised when the database configuration is missing or not found."""

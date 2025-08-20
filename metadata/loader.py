import itertools
import pathlib
import re
from typing import Any, ClassVar, Union

import yaml

from src.utils.logger import create_logger

logger = create_logger(level="DEBUG")


class MetadataError(Exception):
    """Base exception for metadata processing errors."""


class ModelNotFoundError(MetadataError):
    """Raised when a model's file cannot be found."""


class ValidationError(MetadataError):
    """Raised on validation failures (e.g., in calculations or imports)."""


def read_yaml_file(path: str) -> dict[str, Any]:
    """Reads a YAML file from the given path."""
    try:
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ModelNotFoundError(f"File not found at path: {path}") from e
    except yaml.YAMLError as e:
        raise MetadataError(f"Error parsing YAML file {path}: {e}") from e


class BaseModel:
    """A base class to hold shared validation logic for metadata models."""

    @staticmethod
    def _validate_structure(data: dict, valid_labels: list[str], model_type: str) -> None:
        """
        Validates the top-level keys of the loaded data.

        :param data: The dictionary of data loaded from a YAML file.
        :param valid_labels: A list of allowed top-level keys.
        :param model_type: A string representing the type of model (e.g., 'schema') for error messages.
        :raises ValidationError: If any invalid keys are found.
        """
        if not data:
            return
        invalid_keys = set(data.keys()) - set(valid_labels)
        if invalid_keys:
            raise ValidationError(f"Invalid top-level {model_type} key(s) found: {', '.join(invalid_keys)}")


class SchemaModel(BaseModel):
    """
    Represents a single, validated schema definition, including its columns and table structure.
    """

    valid_labels: ClassVar[list[str]] = ["subject_area", "source", "columns", "functions", "table_info"]

    def __init__(self, name: str, data: dict) -> None:
        """
        Initializes a SchemaModel.

        :param name: The name of the schema (e.g., 'GAMES').
        :param data: The raw dictionary content loaded from the schema's YAML file.
        """
        self._validate_structure(data, self.valid_labels, "schema")
        self.name = name
        self.data = data
        self.columns: dict[str, Any] = data.get("columns", {})
        self.table_info: list[dict[str, Any]] = data.get("table_info", [])


class SemanticModel(BaseModel):
    """
    Represents a single, validated semantic model, handling the resolution of its own
    dependencies and the validation of its business logic.
    """

    valid_labels: ClassVar[list[str]] = ["folder", "type", "source", "metrics", "attributes", "hierarchies"]

    def __init__(self, name: str, raw_data: dict) -> None:
        """
        Initializes a SemanticModel.

        :param name: The name of the semantic model (e.g., 'game_stats').
        :param raw_data: The raw dictionary content loaded from the model's YAML file.
        """
        self._validate_structure(raw_data, self.valid_labels, "semantic")
        self.name = name
        self._raw_data = raw_data
        self.sources: dict[str, Any] = self._raw_data.get("source", {})
        self.attributes: dict[str, Any] = self._raw_data.get("attributes", {}).copy()
        self.metrics: dict[str, Any] = self._raw_data.get("metrics", {}).copy()
        self.hierarchies: dict[str, Any] = self._raw_data.get("hierarchies", {}).copy()
        self.data_context: dict[str, dict[str, Any]] = {}
        self.base_source_tables: list[str] = []

    def _get_available_scoped_columns(self) -> set[str]:
        """
        Helper to create a set of all available 'TABLE.COLUMN' strings from the data context.

        :return: A set of all fully-qualified column names.
        """
        return {
            f"{table_name}.{column_name}"
            for table_name, columns in self.data_context.items()
            for column_name in columns
        }

    def _get_common_unscoped_columns(self) -> set[str]:
        """
        Helper method to identify column names that are common across ALL directly sourced base schemas.

        :return: A set of column names that can be used without a table scope.
        """
        if not self.base_source_tables:
            return set()

        column_sets = [set(self.data_context[table_name].keys()) for table_name in self.base_source_tables]
        return set.intersection(*column_sets) if column_sets else set()

    def _add_source_to_context(self, base_schema: "SchemaModel") -> list[dict[str, Any]]:
        """Helper method to add a base schema and its columns to the data context.

        :param base_schema: The base schema to add columns to.
        """
        defining_table_info = base_schema.table_info
        primary_table_name = defining_table_info[0].get("table")
        if not primary_table_name:
            raise ValidationError(f"Primary table name not defined in schema '{base_schema.name}'")

        self.base_source_tables.append(primary_table_name)
        self.data_context[primary_table_name] = {
            name: {**details, "table_source": defining_table_info} for name, details in base_schema.columns.items()
        }
        return defining_table_info

    def _add_joins_to_context(
        self, loader: "MetadataLoader", defining_table_info: list[dict], source_model_name: str
    ) -> None:
        """Helper method to process joins and add aliased tables to the data context.

        :param loader: The metadata loader used to load data from this model.
        :param defining_table_info: A list of dictionaries defining table names and aliases.
        :param source_model_name: The model name used to load data from this model.
        """
        for join_info in defining_table_info[0].get("joins", []):
            join_def = join_info.get("join")
            if not isinstance(join_def, dict) or "table" not in join_def or "as" not in join_def:
                raise ValidationError(
                    f"Join in schema '{source_model_name}' is not properly formed with 'table' and 'as' keys."
                )

            joined_table_schema = loader.get_schema_model(join_def["table"])
            alias = join_def["as"]
            self.data_context[alias] = {
                name: {**details, "table_source": defining_table_info}
                for name, details in joined_table_schema.columns.items()
            }

    def resolve_dependencies(self, loader: "MetadataLoader") -> None:
        """
        Builds a scoped data_context by processing all sources and joins.
        """
        for source_name, _ in self.sources:
            source_type, source_model_name = source_name.split(".")
            if source_type != "schema":
                raise NotImplementedError("Source imports from other semantic models are not yet supported.")

            base_schema = loader.get_schema_model(source_model_name)
            defining_table_info = self._add_source_to_context(base_schema)
            self._add_joins_to_context(loader, defining_table_info, source_model_name)

    @staticmethod
    def _get_all_dependencies(item_content: dict) -> set[str]:
        """Helper method to extract all dependencies from an item's calculation, include, and filter clauses."""
        dependencies = set(item_content.get("include", []))

        strings_with_deps = []
        if item_content.get("calculation"):
            strings_with_deps.append(item_content["calculation"])

        filters = item_content.get("filters", item_content.get("filter", []))
        strings_with_deps.extend(filters)

        for text in strings_with_deps:
            dependencies.update(re.findall(r"\[([^\]]+)\]", text))

        return dependencies

    def validate(self) -> None:
        """
        Validates all dependencies for all metrics and attributes in the model.
        """
        valid_local_items = set(self.attributes.keys()) | set(self.metrics.keys())
        valid_scoped_columns = self._get_available_scoped_columns()
        valid_common_unscoped_cols = self._get_common_unscoped_columns()

        items_to_check = {**self.attributes, **self.metrics}

        for item_name, item_content in items_to_check.items():
            dependencies = self._get_all_dependencies(item_content)
            for dep in dependencies:
                if not (dep in valid_local_items or dep in valid_scoped_columns or dep in valid_common_unscoped_cols):
                    raise ValidationError(
                        f"In model '{self.name}', item '{item_name}' has an undefined or ambiguous dependency: '{dep}'"
                    )

    def _merge_item_properties(self, item_collection: dict) -> None:
        """Helper method to merge column properties, handling both scoped and unscoped includes.

        :param item_collection: The collection of items to merge.
        """
        for item_name, item_details in item_collection.items():
            for include_item in item_details.get("include", []):
                if "." in include_item:
                    table, column = include_item.split(".")
                    if table in self.data_context and column in self.data_context[table]:
                        item_collection[item_name] = {**self.data_context[table][column], **item_details}
                else:
                    for table in self.base_source_tables:
                        if include_item in self.data_context.get(table, {}):
                            item_collection[item_name] = {**self.data_context[table][include_item], **item_details}
                            break

    def merge_components(self) -> None:
        """Merges properties from included columns into both attributes and metrics."""
        self._merge_item_properties(self.attributes)
        self._merge_item_properties(self.metrics)

    @staticmethod
    def _extract_drill_across(source_hierarchy: str, source_level: str, drill_data: list[dict]) -> list[dict]:
        """Helper to parse drill_across definitions from a hierarchy level.

        :param source_hierarchy: The hierarchy level to extract data from.
        :param source_level: The hierarchy level to extract data from.
        :param drill_data: A list of dictionaries defining table names and aliases.

        :return: A list of dictionaries defining table names and aliases.
        """
        return [
            {
                "source_hierarchy": source_hierarchy,
                "source_level_name": source_level,
                "target_hierarchy": drill.get("target_hierarchy"),
                "target_level_name": drill.get("target_level_name"),
            }
            for drill in drill_data
        ]

    def _parse_single_level(self, hierarchy_key: str, level_def: dict) -> tuple[dict, list]:
        """Helper to parse a single level within a hierarchy.

        :param hierarchy_key: The hierarchy level to extract data from.
        :param level_def: The hierarchy level to extract data from.
        """
        if isinstance(level_def, dict):
            level_name, attributes = level_def.get("level_name"), level_def.get("attributes", [])
            drills = self._extract_drill_across(hierarchy_key, level_name, level_def.get("drill_across", []))
            return {"level_name": level_name, "attributes": attributes}, drills
        return {"level_name": level_def, "attributes": []}, []

    def _parse_hierarchy_levels(self, hierarchy_key: str, levels_data: list) -> tuple[list, list]:
        """Helper to parse all levels for a given hierarchy.

        :param hierarchy_key: The hierarchy level to extract data from.
        :param levels_data: A list of dictionaries defining table names and aliases.
        :return: A list of dictionaries defining table names and aliases.
        """
        parsed_levels, drill_across_list = [], []
        for level_def in levels_data:
            parsed_level, drills = self._parse_single_level(hierarchy_key, level_def)
            parsed_levels.append(parsed_level)
            drill_across_list.extend(drills)
        return parsed_levels, drill_across_list

    def get_parsed_hierarchies(self) -> tuple[dict, list]:
        """
        Parses all hierarchy definitions for this model.

        :return: A tuple containing a dictionary of parsed hierarchies and a list of all drill_across definitions.
        """
        parsed_hierarchies, drill_across_list = {}, []
        for key, value in self.hierarchies.items():
            levels, drills = self._parse_hierarchy_levels(key, value)
            parsed_hierarchies[key], drill_across_list = levels, drill_across_list + drills
        return parsed_hierarchies, drill_across_list


class SemanticParser:
    """
    A dedicated class to handle the parsing, validation, and resolution
    of a raw SemanticModel using a provided set of schema dependencies.
    """

    def __init__(
        self,
        semantic_model: SemanticModel,
        semantic_dependencies: dict[str, SemanticModel],
        schema_dependencies: dict[str, SchemaModel],
    ) -> None:
        """
        Initializes the parser.

        :param semantic_model: The raw SemanticModel instance to be processed.
        :param schema_dependencies: A dictionary of all SchemaModel objects required by the semantic model.
        """
        self.model = semantic_model
        self.semantic_dependencies = semantic_dependencies
        self.schema_dependencies = schema_dependencies

    def process(self) -> SemanticModel:
        """
        Orchestrates the full processing pipeline for the semantic model.

        :return: The fully resolved and validated SemanticModel.
        """
        self._resolve_dependencies()
        self._validate()
        self._merge_components()
        self._parse_hierarchies()
        return self.model

    @staticmethod
    def _fix_join_on_keys(table_info: list[dict]) -> list[dict]:
        """
        Corrects a common YAML parsing issue where an unquoted 'on:' key is read as True.

        :param table_info: The raw table_info list from a schema.
        :return: The corrected table_info list with 'on' as a string key.
        """
        for info in table_info:
            if "joins" in info and isinstance(info["joins"], list):
                for join_item in info["joins"]:
                    if isinstance(join_item, dict) and True in join_item:
                        join_item["on"] = join_item.pop(True)
        return table_info

    def _resolve_dependencies(self) -> None:
        """
        Builds the model's data_context by processing its explicit sources.
        Schema joins are not processed for dependencies.
        """
        for source_name in self.model.sources:
            source_type, source_model_name = source_name.split(".")

            if source_type == "schema":
                base_schema = self.schema_dependencies.get(source_model_name)
                if not base_schema:
                    raise ValidationError(f"Schema dependency '{source_model_name}' not found.")

                defining_table_info = self._fix_join_on_keys(base_schema.table_info)
                self._add_source_to_context(base_schema, defining_table_info)
            elif source_type == "semantic":
                dependency_model = self.semantic_dependencies.get(source_model_name)
                if not dependency_model:
                    raise ValidationError(f"Semantic dependency '{source_model_name}' not found.")

                self.model.attributes.update(dependency_model.attributes)
                self.model.metrics.update(dependency_model.metrics)
            else:
                raise NotImplementedError(f"Unsupported source type: {source_type}")

    def _add_source_to_context(self, base_schema: "SchemaModel", defining_table_info: list[dict]) -> list[dict]:
        """Helper to add a base schema and its columns to the data context."""
        primary_table_name = defining_table_info[0].get("table")
        if not primary_table_name:
            raise ValidationError(f"Primary table name not defined in schema '{base_schema.name}'")

        self.model.base_source_tables.append(primary_table_name)
        self.model.data_context[primary_table_name] = {
            name: {**details, "table_source": defining_table_info} for name, details in base_schema.columns.items()
        }
        return defining_table_info

    def _add_joins_to_context(self, defining_table_info: list[dict], source_model_name: str) -> None:
        """Helper to process joins and add aliased tables to the data context."""
        for join_info in defining_table_info[0].get("joins", []):
            join_def = join_info.get("join")

            table_name = None
            alias = None

            if isinstance(join_def, dict):
                table_name = join_def.get("table")
                alias = join_def.get("as")
                if not table_name or not alias:
                    raise ValidationError(
                        f"Aliased join in schema '{source_model_name}' must have 'table' and 'as' keys."
                    )
            elif isinstance(join_def, str):
                table_name = join_def
                alias = join_def
            else:
                raise ValidationError(f"Join definition in schema '{source_model_name}' is malformed.")

            joined_table_schema = self.schema_dependencies.get(table_name)
            if not isinstance(joined_table_schema, SchemaModel):
                raise ValidationError(
                    f"Joined table '{table_name}' must be a schema, but its dependency was not found or is of the wrong type."
                )

            self.model.data_context[alias] = {
                name: {**details, "table_source": defining_table_info}
                for name, details in joined_table_schema.columns.items()
            }

    def _validate(self) -> None:
        """Validates all dependencies for all metrics and attributes in the model."""
        valid_local_items = set(self.model.attributes.keys()) | set(self.model.metrics.keys())
        valid_scoped_columns = self.model._get_available_scoped_columns()
        valid_common_unscoped_cols = self.model._get_common_unscoped_columns()

        items_to_check = {**self.model.attributes, **self.model.metrics}

        for item_name, item_content in items_to_check.items():
            dependencies = self.model._get_all_dependencies(item_content)
            for dep in dependencies:
                dep_valid = dep in valid_local_items or dep in valid_scoped_columns or dep in valid_common_unscoped_cols
                if not dep_valid:
                    for table_name in self.model.base_source_tables:
                        qualified_dep = f"{table_name}.{dep}"
                        if qualified_dep in valid_scoped_columns:
                            dep_valid = True
                            break
                if not dep_valid:
                    raise ValidationError(
                        f"In model '{self.model.name}', item '{item_name}' has an undefined or ambiguous dependency: '{dep}'"
                    )

    def _merge_components(self) -> None:
        """Merges properties from included columns into both attributes and metrics."""
        self.model._merge_item_properties(self.model.attributes)
        self.model._merge_item_properties(self.model.metrics)

    def _parse_hierarchies(self) -> None:
        """Parses all hierarchy definitions for the model and stores them."""
        parsed_hierarchies, drill_across_list = {}, []
        for key, value in self.model.hierarchies.items():
            levels, drills = self._parse_hierarchy_levels(key, value)
            parsed_hierarchies[key] = levels
            drill_across_list.extend(drills)
        self.model.parsed_hierarchies = (parsed_hierarchies, drill_across_list)

    def _parse_hierarchy_levels(self, hierarchy_key: str, levels_data: list) -> tuple[list, list]:
        """Helper to parse all levels for a given hierarchy."""
        parsed_levels, drill_across_list = [], []
        for level_def in levels_data:
            parsed_level, drills = self._parse_single_level(hierarchy_key, level_def)
            parsed_levels.append(parsed_level)
            drill_across_list.extend(drills)
        return parsed_levels, drill_across_list

    def _parse_single_level(self, hierarchy_key: str, level_def: dict) -> tuple[dict, list]:
        """Helper to parse a single level within a hierarchy."""
        if isinstance(level_def, dict):
            level_name, attributes = level_def.get("level_name"), level_def.get("attributes", [])
            drills = self._extract_drill_across(hierarchy_key, level_name, level_def.get("drill_across", []))
            return {"level_name": level_name, "attributes": attributes}, drills
        return {"level_name": level_def, "attributes": []}, []

    @staticmethod
    def _extract_drill_across(source_hierarchy: str, source_level: str, drill_data: list[dict]) -> list[dict]:
        """Helper to parse drill_across definitions from a hierarchy level."""
        return [
            {
                "source_hierarchy": source_hierarchy,
                "source_level_name": source_level,
                "target_hierarchy": drill.get("target_hierarchy"),
                "target_level_name": drill.get("target_level_name"),
            }
            for drill in drill_data
        ]


class MetadataLoader:
    """
    Manages the loading, resolution, and caching of all metadata models.
    This is the primary entry point for the system.
    """

    def __init__(self, schema_input: str | dict, semantics_input: str | dict) -> None:
        """
        Initializes the MetadataLoader. Can accept either directory paths or preloaded dictionaries.

        :param schema_input: The directory path for schemas, or a dict of preloaded schema assets.
        :param semantics_input: The directory path for semantics, or a dict of preloaded semantic assets.
        :raises FileNotFoundError: If a directory path is provided and does not exist.
        :raises TypeError: If the inputs are not of type str or dict.
        """
        self._cache: dict[str, Any] = {}

        if isinstance(schema_input, str):
            if not pathlib.Path(schema_input).is_dir():
                raise FileNotFoundError(f"Schema directory not found: '{schema_input}'")
            self._schema_mode = "dir"
            self.schema_dir = schema_input
        elif isinstance(schema_input, dict):
            self._schema_mode = "dict"
            self._schema_data = {k.lower(): v for k, v in schema_input.items()}
        else:
            raise TypeError("schema_input must be a directory path (str) or a dictionary.")

        if isinstance(semantics_input, str):
            if not pathlib.Path(semantics_input).is_dir():
                raise FileNotFoundError(f"Semantics directory not found: '{semantics_input}'")
            self._semantics_mode = "dir"
            self.semantics_dir = semantics_input
        elif isinstance(semantics_input, dict):
            self._semantics_mode = "dict"
            self._semantics_data = semantics_input
        else:
            raise TypeError("semantics_input must be a directory path (str) or a dictionary.")

    def _get_model_type(self, model_name: str) -> str:
        """
        Determines if a model is a 'schema' or 'semantic' based on file existence.

        Args:
            model_name: Name of the model (without extension).

        Returns:
            "schema" if a schema YAML file exists,
            "semantic" if a semantic YAML file exists,
            "unknown" otherwise.
        """
        schema_path = pathlib.Path(self.schema_dir) / f"{model_name}.yaml"
        semantics_path = pathlib.Path(self.semantics_dir) / f"{model_name}.yaml"

        if self._schema_mode == "dir" and schema_path.exists():
            return "schema"

        if self._semantics_mode == "dir" and semantics_path.exists():
            return "semantic"

        # Default if no specific file is found
        return "unknown"

    def get_model(self, model_name: str) -> BaseModel:
        """
        Returns a fully resolved model, dynamically determining if it's a schema or semantic model.
        """
        model_type = self._get_model_type(model_name)

        if model_type == "schema":
            return self.get_schema_model(model_name)
        if model_type == "semantic":
            return self.get_semantic_model(model_name)
        raise ModelNotFoundError(f"Model '{model_name}' not found in any metadata directory.")

    def get_semantic_model(self, model_name: str) -> "SemanticModel":
        """
        Returns a fully resolved semantic model. Uses a cache to avoid reprocessing.
        """
        if model_name in self._cache and isinstance(self._cache[model_name], SemanticModel):
            return self._cache[model_name]

        if self._semantics_mode == "dir":
            path = pathlib.Path(self.semantics_dir) / f"{model_name}.yaml"
            raw_data = read_yaml_file(path)
        else:
            raw_data = self._semantics_data.get(model_name)
            if raw_data is None:
                raise ModelNotFoundError(f"Semantic model '{model_name}' not found in the provided dictionary.")

        raw_model = SemanticModel(model_name, raw_data)

        # Gather dependencies, but with the new non-traversing logic
        semantic_deps, schema_deps = self._gather_all_dependencies(raw_model)

        parser = SemanticParser(raw_model, semantic_deps, schema_deps)
        processed_model = parser.process()

        self._cache[model_name] = processed_model
        return processed_model

    def _gather_all_dependencies(
        self, model: Union["SemanticModel", "SchemaModel"], visited: set[str] | None = None
    ) -> tuple[dict[str, "SemanticModel"], dict[str, "SchemaModel"]]:
        """
        Recursively gathers all explicit semantic and schema dependencies.
        Joins are no longer traversed to find dependencies.

        Args:
            model: The root model (SemanticModel or SchemaModel).
            visited: Set of model names already visited to prevent cycles.

        Returns:
            Tuple of:
                - dict of semantic models by name
                - dict of schema models by name
        """
        if visited is None:
            visited = set()
        if model.name in visited:
            return {}, {}
        visited.add(model.name)

        semantic_deps: dict[str, SemanticModel] = {}
        schema_deps: dict[str, SchemaModel] = {}

        if isinstance(model, SemanticModel):
            self._process_semantic_sources(model, visited, semantic_deps, schema_deps)
        elif isinstance(model, SchemaModel):
            self._process_schema_joins(model, visited, semantic_deps, schema_deps)

        return semantic_deps, schema_deps

    def _process_semantic_sources(
        self,
        model: "SemanticModel",
        visited: set[str],
        semantic_deps: dict[str, "SemanticModel"],
        schema_deps: dict[str, "SchemaModel"],
    ) -> None:
        """Processes all explicit sources of a SemanticModel."""
        for source_name in model.sources:
            _source_type, source_model_name = source_name.split(".")
            dep_model = self.get_model(source_model_name)

            if isinstance(dep_model, SemanticModel):
                semantic_deps[source_model_name] = dep_model
                nested_sem, nested_sch = self._gather_all_dependencies(dep_model, visited)
                semantic_deps.update(nested_sem)
                schema_deps.update(nested_sch)
            elif isinstance(dep_model, SchemaModel):
                schema_deps[source_model_name] = dep_model

    def _process_schema_joins(
        self,
        model: "SchemaModel",
        visited: set[str],
        semantic_deps: dict[str, "SemanticModel"],
        schema_deps: dict[str, "SchemaModel"],
    ) -> None:
        """Processes all joins in a SchemaModel."""
        for join_info in model.table_info[0].get("joins", []):
            join_def = join_info.get("join", {})
            table_to_join = join_def.get("table") if isinstance(join_def, dict) else join_def

            if table_to_join:
                dep_model = self.get_model(table_to_join)
                if not isinstance(dep_model, SchemaModel):
                    raise ValidationError(
                        f"Joined table '{table_to_join}' in schema '{model.name}' must be a schema, not a {type(dep_model).__name__}."
                    )
                schema_deps[table_to_join] = dep_model

                nested_sem, nested_sch = self._gather_all_dependencies(dep_model, visited)
                schema_deps.update(nested_sch)
                semantic_deps.update(nested_sem)

    def get_schema_model(self, schema_name: str) -> "SchemaModel":
        """
        Returns a schema model. Uses a cache to avoid rereading files.

        :param schema_name: The name of the schema to load.
        :return: A SchemaModel instance.
        """
        schema_name_lower = schema_name.lower()
        if schema_name_lower in self._cache and isinstance(self._cache[schema_name_lower], SchemaModel):
            return self._cache[schema_name_lower]

        if self._schema_mode == "dir":
            path = pathlib.Path(self.schema_dir) / f"{schema_name}.yaml"
            raw_data = read_yaml_file(path)
        else:
            raw_data = self._schema_data.get(schema_name_lower)
            if raw_data is None:
                raise ModelNotFoundError(f"Schema '{schema_name}' not found in the provided dictionary.")

        return self._cache.setdefault(schema_name_lower, SchemaModel(schema_name, raw_data))

    def combine_layers(self) -> list[dict[str, Any]]:
        """
        Aggregates all data into a list containing three dictionaries: metrics, attributes, and columns.
        Uses a 'first one wins' strategy for columns to handle naming conflicts.

        :return: A list containing [all_metrics, all_attributes, all_columns].
        """
        semantic_models = [model for model in self._cache.values() if isinstance(model, SemanticModel)]

        all_metrics = {
            metric_name: {**metric_details, "source": model.name}
            for model in semantic_models
            for metric_name, metric_details in model.metrics.items()
        }

        all_attributes = {
            attr_name: {**attr_details, "source": model.name}
            for model in semantic_models
            for attr_name, attr_details in model.attributes.items()
        }

        all_columns = {}
        all_data_contexts = itertools.chain.from_iterable(model.data_context.values() for model in semantic_models)
        for columns in all_data_contexts:
            for col_name, col_details in columns.items():
                all_columns.setdefault(col_name, col_details)

        return [all_metrics, all_attributes, all_columns]

    def get_all_functions(self) -> dict[str, Any]:
        """
        Loads and returns all defined functions from 'function.yaml' or a 'function' key in the input dict.

        :return: A dictionary of all defined functions.
        """
        function_cache_key = "__functions__"
        if function_cache_key in self._cache:
            return self._cache[function_cache_key]

        try:
            if self._semantics_mode == "dir":
                function_path = pathlib.Path(self.semantics_dir) / "function.yaml"
                functions_data = read_yaml_file(function_path)
            else:
                functions_data = self._semantics_data.get("function", {})

            functions = functions_data.get("functions", {})
            return self._cache.setdefault(function_cache_key, functions)
        except (ModelNotFoundError, KeyError):
            return self._cache.setdefault(function_cache_key, {})

    def search_metrics(self, word: str) -> list[dict[str, Any]]:
        """
        Flattens the search for a keyword across all loaded metrics.

        :param word: The keyword to search for in metric names and descriptions.
        :return: A list of metric dictionaries that match the search term.
        """
        word_lower = word.lower()
        semantic_models = (m for m in self._cache.values() if isinstance(m, SemanticModel))
        all_metrics_iter = itertools.chain.from_iterable(m.metrics.items() for m in semantic_models)
        return [
            {name: details}
            for name, details in all_metrics_iter
            if word_lower in details.get("name", "").lower() or word_lower in details.get("description", "").lower()
        ]

    def get_all_hierarchies(self) -> dict[str, Any]:
        """
        Aggregates and parses all hierarchies from all loaded models, validating for duplicates.

        :return: A dictionary containing 'hierarchy_info' and 'drill_across_info'.
        :raises ValidationError: If duplicate hierarchy keys are found across different models.
        """
        all_hierarchies, drill_across_info = {}, []
        all_parsed_data = [m.parsed_hierarchies for m in self._cache.values() if isinstance(m, SemanticModel)]
        all_keys = [key for parsed_data, _ in all_parsed_data for key in parsed_data]

        seen = set()
        duplicates = set()
        for key in all_keys:
            if key in seen:
                duplicates.add(key)
            else:
                seen.add(key)

        if duplicates:
            raise ValidationError(f"Duplicate hierarchy keys found across models: {', '.join(duplicates)}")

        for parsed_data, drill_list in all_parsed_data:
            all_hierarchies.update(parsed_data)
            drill_across_info.extend(drill_list)
        return {"hierarchy_info": all_hierarchies, "drill_across_info": drill_across_info}


def get_metadata_from_directory(asset_names: list[str]) -> tuple[dict, dict, dict, dict]:
    """
    Retrieves and processes metadata for a given list of asset names
    using the optimized, on-demand MetadataLoader.

    :param asset_names: A list of asset names for which to retrieve metadata.
    :return: A tuple containing final_metrics, final_attributes, final_columns, and final_functions.
    """
    # Define base paths for metadata assets
    schema_dir = "metadata/schema/assets"
    semantics_dir = "metadata/semantics/assets"

    loader = MetadataLoader(schema_dir, semantics_dir)

    for asset_name in asset_names:
        normalized_name = asset_name.lower()
        try:
            loader.get_semantic_model(normalized_name)
        except ModelNotFoundError as e:
            # Handle cases where a model for an asset might not exist
            logger.warning(f"Warning: Could not process asset '{asset_name}'. Reason: {e}")
            continue

    final_metrics, final_attributes, final_columns = loader.combine_layers()

    final_functions = loader.get_all_functions()

    return final_metrics, final_attributes, final_columns, final_functions


if __name__ == "__main__":
    final_metadata = get_metadata_from_directory(["games"])

    if final_metadata is not None:
        (
            final_metrics,
            final_attributes,
            final_columns,
            final_functions,
        ) = final_metadata

    final_metadata = list(final_metadata)
    logger.info(final_metadata)

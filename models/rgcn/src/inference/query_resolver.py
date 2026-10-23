"""Resolve user-facing entity queries for the inference command."""

from __future__ import annotations

from difflib import SequenceMatcher
import unicodedata
from typing import Callable, Dict, Iterable, List, Set, Tuple


EntityAliases = Dict[str, List[Tuple[str, str]]]
InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


class QuerySelectionCancelled(Exception):
    """Raised when the user cancels interactive query selection."""


class QueryNotFoundError(ValueError):
    """Raised when no entity in the graph is sufficiently close to a query."""

    def __init__(self, query: str, entity_type: str) -> None:
        noun = _entity_noun(entity_type)
        plural = {
            "disease": "diseases",
            "drug": "drugs",
            "ingredient": "ingredients",
            "gene": "genes",
            "entity": "entities",
        }[noun]
        super().__init__(
            f'No matching {noun} was found for "{query}".\n\n'
            f"Only {plural} represented in the current graph can be queried.\n"
            "Try another name or use a canonical identifier with --entity_id."
        )


def preferred_entity_label(
    entity_id: str,
    entity_type: str,
    entity_labels: Dict[str, str],
    entity_aliases: EntityAliases,
) -> str:
    """Choose a readable, deterministic label for an entity."""
    preferred_alias_types = {
        "DRUG": ("DRUG_NAME_ORIGINAL", "DRUG_NAME_NORM"),
        "DISEASE": ("DISEASE_LABEL", "DISEASE_TEXT"),
        "INGREDIENT": ("ING_NAME_ORIGINAL", "ING_NAME_NORM"),
    }

    aliases = entity_aliases.get(entity_id, [])
    for alias_type in preferred_alias_types.get(entity_type, ()):
        for alias, current_type in aliases:
            if current_type == alias_type:
                return alias

    label = entity_labels.get(entity_id)
    if label:
        return label

    return entity_id


def _normalize_search_text(value: str) -> str:
    """Normalize text for case-, accent-, and punctuation-insensitive search."""
    decomposed = unicodedata.normalize("NFKD", value.strip().casefold())
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    normalized_characters = [
        character if character.isalnum() else " "
        for character in without_accents
    ]
    return " ".join("".join(normalized_characters).split())


def _entity_search_terms(
    entity_id: str,
    entity_labels: Dict[str, str],
    entity_aliases: EntityAliases,
) -> Set[str]:
    terms = {entity_id}
    label = entity_labels.get(entity_id)
    if label:
        terms.add(label)
    terms.update(alias for alias, _ in entity_aliases.get(entity_id, []))
    return {
        normalized
        for term in terms
        if (normalized := _normalize_search_text(term))
    }


def _similarity(query: str, terms: Iterable[str]) -> float:
    scores = []
    query_tokens = query.split()
    for term in terms:
        score = SequenceMatcher(None, query, term).ratio()
        if len(query_tokens) == 1:
            token_score = max(
                (
                    SequenceMatcher(None, query, token).ratio()
                    for token in term.split()
                ),
                default=0.0,
            )
            score = max(score, token_score * 0.95)
        scores.append(score)
    return max(scores, default=0.0)


def _ordered_matches(
    candidates: Iterable[str],
    query: str,
    terms_by_entity: Dict[str, Set[str]],
    entity_types: Dict[str, str],
    entity_labels: Dict[str, str],
    entity_aliases: EntityAliases,
) -> List[str]:
    return sorted(
        set(candidates),
        key=lambda current_entity_id: (
            -_similarity(query, terms_by_entity[current_entity_id]),
            preferred_entity_label(
                current_entity_id,
                entity_types.get(current_entity_id, "UNKNOWN"),
                entity_labels,
                entity_aliases,
            ).casefold(),
            current_entity_id,
        ),
    )


def _format_matches(
    candidates: Iterable[str],
    entity_types: Dict[str, str],
    entity_labels: Dict[str, str],
    entity_aliases: EntityAliases,
) -> str:
    lines = []
    for index, entity_id in enumerate(candidates, start=1):
        label = preferred_entity_label(
            entity_id,
            entity_types.get(entity_id, "UNKNOWN"),
            entity_labels,
            entity_aliases,
        )
        lines.append(f"  [{index}] {label} ({entity_id})")
    return "\n".join(lines)


def _entity_noun(entity_type: str) -> str:
    return {
        "DISEASE": "disease",
        "DRUG": "drug",
        "INGREDIENT": "ingredient",
        "GENE": "gene",
    }.get(entity_type, "entity")


def resolve_query_entity(
    query: str | None,
    entity_id: str | None,
    expected_entity_type: str,
    entity2id: Dict[str, int],
    entity_types: Dict[str, str],
    entity_labels: Dict[str, str],
    entity_aliases: EntityAliases | None = None,
    *,
    interactive: bool = False,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
    max_suggestions: int = 5,
    fuzzy_cutoff: float = 0.70,
) -> Tuple[str, int]:
    """Resolve a query to one canonical entity, optionally asking the user."""
    aliases = entity_aliases or {}

    if entity_id:
        if entity_id not in entity2id:
            raise ValueError(f"Entity ID not found in the graph: {entity_id}")

        actual_type = entity_types.get(entity_id, "UNKNOWN")
        if expected_entity_type != "ALL" and actual_type != expected_entity_type:
            raise ValueError(
                f"Entity '{entity_id}' has type '{actual_type}', "
                f"expected '{expected_entity_type}'."
            )

        return entity_id, int(entity2id[entity_id])

    if not query or not query.strip():
        raise ValueError("You must provide either --query or --entity_id.")
    if max_suggestions <= 0:
        raise ValueError("max_suggestions must be greater than zero.")

    if expected_entity_type == "ALL":
        valid_entity_ids = list(entity2id.keys())
    else:
        valid_entity_ids = [
            current_entity_id
            for current_entity_id, current_type in entity_types.items()
            if current_type == expected_entity_type
            and current_entity_id in entity2id
        ]

    terms_by_entity = {
        current_entity_id: _entity_search_terms(
            current_entity_id,
            entity_labels,
            aliases,
        )
        for current_entity_id in valid_entity_ids
    }
    current_query = query.strip()

    while True:
        normalized_query = _normalize_search_text(current_query)
        if not normalized_query:
            raise ValueError("The query must not be empty.")

        exact_matches = [
            current_entity_id
            for current_entity_id, terms in terms_by_entity.items()
            if normalized_query in terms
        ]

        if len(exact_matches) == 1:
            selected_entity_id = exact_matches[0]
            return selected_entity_id, int(entity2id[selected_entity_id])

        if exact_matches:
            matches = _ordered_matches(
                exact_matches,
                normalized_query,
                terms_by_entity,
                entity_types,
                entity_labels,
                aliases,
            )
            match_message = (
                f'The query "{current_query}" matches multiple '
                f'{_entity_noun(expected_entity_type)} entities.'
            )
        else:
            prefix_matches = [
                current_entity_id
                for current_entity_id, terms in terms_by_entity.items()
                if any(term.startswith(normalized_query) for term in terms)
            ]
            substring_matches = [
                current_entity_id
                for current_entity_id, terms in terms_by_entity.items()
                if any(normalized_query in term for term in terms)
            ]

            fuzzy_matches = []
            if len(normalized_query) >= 3:
                fuzzy_matches = [
                    current_entity_id
                    for current_entity_id, terms in terms_by_entity.items()
                    if _similarity(normalized_query, terms) >= fuzzy_cutoff
                ]

            matches = _ordered_matches(
                prefix_matches or substring_matches or fuzzy_matches,
                normalized_query,
                terms_by_entity,
                entity_types,
                entity_labels,
                aliases,
            )
            match_message = f'No exact match was found for "{current_query}".'

        if not matches:
            raise QueryNotFoundError(current_query, expected_entity_type)

        shown_matches = matches[:max_suggestions]
        formatted_matches = _format_matches(
            shown_matches,
            entity_types,
            entity_labels,
            aliases,
        )

        if not interactive:
            raise ValueError(
                f"{match_message}\nPossible matches:\n{formatted_matches}\n"
                "Repeat --query with one complete name, or use --entity_id with "
                "the intended canonical ID."
            )

        output_fn("")
        output_fn(match_message)
        output_fn("")
        output_fn("Possible matches:")
        output_fn(formatted_matches)
        if len(matches) > len(shown_matches):
            output_fn("")
            output_fn(
                f"Showing the {len(shown_matches)} closest of "
                f"{len(matches)} possible matches."
            )

        while True:
            try:
                selection = input_fn(
                    "\nSelect an option "
                    f"[1-{len(shown_matches)}], enter R to refine the query, "
                    "or enter 0 to cancel: "
                ).strip()
            except (EOFError, KeyboardInterrupt) as error:
                raise QuerySelectionCancelled from error

            if selection == "0":
                raise QuerySelectionCancelled

            if selection.casefold() == "r":
                try:
                    refined_query = input_fn("Enter a new query: ").strip()
                except (EOFError, KeyboardInterrupt) as error:
                    raise QuerySelectionCancelled from error
                if not refined_query:
                    output_fn("The query must not be empty.")
                    continue
                current_query = refined_query
                break

            try:
                selected_index = int(selection)
            except ValueError:
                output_fn("Invalid selection. Please enter one of the listed options.")
                continue

            if not 1 <= selected_index <= len(shown_matches):
                output_fn("Invalid selection. Please enter one of the listed options.")
                continue

            selected_entity_id = shown_matches[selected_index - 1]
            selected_label = preferred_entity_label(
                selected_entity_id,
                entity_types.get(selected_entity_id, "UNKNOWN"),
                entity_labels,
                aliases,
            )
            output_fn("")
            output_fn(
                f"Selected {_entity_noun(expected_entity_type)}: "
                f"{selected_label} ({selected_entity_id})"
            )
            return selected_entity_id, int(entity2id[selected_entity_id])

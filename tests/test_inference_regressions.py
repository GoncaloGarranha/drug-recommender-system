"""Expose the existing inference tests through the central test suite."""

from models.rgcn.tests.test_infer import (  # noqa: F401
    EntityMetadataTests,
    QueryResolutionTests,
    SplitStatusTests,
)

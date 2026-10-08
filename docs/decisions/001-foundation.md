# Foundation choices

Use standard-library frozen dataclasses and TOML (Python 3.11 tomllib) to keep
local/offline execution dependency-free. Domain implementation is centralized
in environment.py rather than duplicated across state/actions/transitions modules.
Use unittest tests that are also discoverable by pytest. Empty package and directory
placeholders reserve future layers, not completed capabilities.

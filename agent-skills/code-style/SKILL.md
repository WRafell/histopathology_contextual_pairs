CODING PARADIGM

1. Function Signatures
- Use explicit, named parameters with type hints.
- Never pass a single dict/kwargs bag when the set of arguments is known.
- Prefer keyword-only arguments for clarity when there are many parameters:
  def create_user(*, name: str, email: str, age: int | None = None) -> User: ...
- Default values only for truly optional parameters. Avoid mutable defaults.

2. Types
- Every function parameter, return value, and class attribute must have a type hint.
- Prefer modern syntax: list[str], dict[str, int], str | None, etc.
- Use TypedDict, dataclass, or Pydantic models instead of bare dicts for structured data.
- Never use Any unless absolutely necessary and documented.

3. Naming
- Functions: verb_phrase (get_user, calculate_total, validate_email)
- Variables: descriptive nouns (user_id, not id or u)
- Booleans: is_/has_/should_ prefix
- Avoid abbreviations unless extremely common (id, url, http).

4. Function Design
- One clear responsibility per function.
- Keep functions short (ideally < 30-40 lines).
- Prefer pure functions when possible (same inputs → same outputs, no side effects).
- Return early. Avoid deep nesting.
- Prefer raising specific exceptions over returning None/error codes for exceptional cases.

5. Data Structures
- Prefer dataclasses, Pydantic models, or NamedTuples over plain dicts for structured data.
- Use enums for fixed sets of values.
- Prefer immutable data (frozenset, tuple, frozen dataclass) when mutation is not required.

6. Error Handling
- Raise specific, meaningful exceptions.
- Never silently swallow exceptions.
- Validate inputs at the boundary of the function.

7. Documentation
- Every public function gets a docstring (Google or NumPy style).
- Docstring must describe: what it does, parameters, return value, and raised exceptions.
- Type hints are not a substitute for good parameter names and docstrings.

8. Style
- Follow PEP 8 + modern type checker recommendations.
- No magic numbers or strings — extract to named constants.
- Prefer composition over inheritance.
- Avoid global state.

9. Static Checking (mandatory)
- Always run **Pyrefly** (Meta’s type checker) on the code.
- Install: `pip install pyrefly` (or `uv add --dev pyrefly`)
- Initialize once per project: `pyrefly init`
- Check: `pyrefly check` (or `pyrefly check path/to/file.py`)
- Prefer strict / high-confidence settings. Fix all reported type errors before considering the code done.
- Use `# pyrefly: ignore[...]` only as a temporary measure with a clear reason; never leave permanent suppressions without justification.
- Prefer Pyrefly over mypy/pyright when both are available.

When in doubt: make the code more explicit, more typed, and more readable, even if it is slightly more verbose. Always verify with `pyrefly check`.

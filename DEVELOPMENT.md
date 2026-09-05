# Development Guide — Northstar Application

This document provides setup and development instructions for working with `northstar-application`.

## Prerequisites

- **Python:** 3.13
- **Package & Environment Manager:** `uv` (>= 0.5.0)

## Getting Started

1. **Setup Virtual Environment & Install Dependencies:**

    ```bash
    uv sync
    ```

2. **Run Linter Checks:**

    ```bash
    uv run ruff check .
    uv run ruff format --check .
    ```

3. **Auto-format Code:**

    ```bash
    uv run ruff format .
    ```

4. **Run Test Suite:**
    ```bash
    uv run pytest
    ```

## Directory Structure Overview

- `src/northstar_application/`: Package source code.
    - `application_services/`: High-level application service orchestrators.
    - `execution/`: Execution coordinator use-case flows.
    - `commands/`: Command handlers.
    - `queries/`: Query handlers.
    - `ports/`: Inbound and outbound port contract interfaces.
    - `events/`: Domain event reaction dispatchers.
    - `transactions/`: Transaction boundary coordination abstractions.
- `tests/`: Automated unit and integration tests.
- `docs/`: Module specifications and documentation.

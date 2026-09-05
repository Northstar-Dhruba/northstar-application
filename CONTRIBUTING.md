# Contributing to Northstar Application

Thank you for contributing to `northstar-application`. This repository contains the Application Layer of the Northstar Platform.

## Architectural Rules

Before making any contributions, you must be familiar with:

- **Architecture Handbook v2.0**
- **Application Layer Design Specification v1.0**
- **Execution Application Module Design Specification v1.0**

### Core Constraints

1. **Orchestration Only:** Never add business invariants, validation rules, or Domain logic to Application services, command handlers, or query handlers. Business decisions belong inside `northstar-core`.
2. **No Technical Protocols:** Never import or write database code, broker wire protocols, FIX logic, REST clients, or messaging frameworks inside `northstar-application`.
3. **Dependency Inversion:** Define Application ports for needed capabilities. Never depend on concrete `northstar-infrastructure` classes.
4. **Constructor Injection:** Always inject port abstractions through class constructors.
5. **Design-First Workflow:** Every application module must be specified in architectural documentation before code is written.

## Development Workflow

1. Ensure Python 3.13 and `uv` are installed.
2. Clone the repository and run `uv sync`.
3. Format and lint code with `uv run ruff check .` and `uv run ruff format --check .`.
4. Run tests with `uv run pytest`.
5. Ensure all PRs pass GitHub Actions CI.

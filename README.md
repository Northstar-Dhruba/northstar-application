# Northstar Application

**Northstar Platform Application Layer — Use-Case Orchestration and Workflow Coordination Boundary**

## Purpose

`northstar-application` owns application use-case coordination for Northstar Platform v1.1.0. It turns inbound application requests, scheduled operations, or approved event-driven triggers into an ordered sequence of Domain operations and external port calls.

It exists to solve cross-context coordination problems without creating alternate business state or breaking Domain ownership.

## Responsibilities

- **Use-Case Orchestration:** Ordering operations across Strategy, Orders, Trades, and Portfolio.
- **Application Services:** Coordinating cohesive application capabilities.
- **Command & Query Handling:** Validating request shape, managing transaction scopes, and dispatching query/command flows.
- **Execution Coordination:** Coordinating approved Order intent with external broker/exchange capability and confirmed Trade/Portfolio outcomes.
- **Port Abstractions:** Defining Application-owned interfaces for external capabilities (persistence, brokers, exchanges, messaging).
- **Domain Event Reactions:** Dispatching post-transaction reactions after Domain facts are established.
- **Transaction Boundaries:** Defining the atomic or compensating scope of application use cases.

## Relationship to Domain

- **Inward Dependency:** `northstar-application` depends on `northstar-core` Domain abstractions.
- **Orchestration Only:** Application coordinates when Domain behavior is invoked; Domain decides whether a business state change is valid.
- **No Business Ownership:** Application does NOT own Order intent, Trade truth, Portfolio holdings, or Strategy policy.
- **Domain Purity:** Domain has zero knowledge of Application, use cases, command handlers, or ports.

## Relationship to Infrastructure

- **Dependency Inversion:** Application defines Port abstractions for external capabilities.
- **Framework & Protocol Independence:** Application contains no database code, broker protocol logic, REST clients, FIX sessions, or messaging implementations.
- **Infrastructure Fulfillment:** `northstar-infrastructure` implements Application ports and is wired at the Composition Root.

## Repository Structure

```text
src/
    northstar_application/
        application_services/   # Cohesive use-case capabilities
        execution/              # Execution coordination capability
        commands/               # Command handlers & workflow state
        queries/                # Read-oriented query handlers
        ports/                  # Application-owned outbound/inbound port contracts
        events/                 # Domain event reaction dispatching
        transactions/           # Use-case transaction scope abstractions
tests/                          # Test suite matching Northstar standards
docs/                           # Application layer architectural documentation
```

## Engineering Rules

1. **Orchestration Only:** Application coordinates approved behavior; it does not own business rules or invariants.
2. **Framework Independence:** Use cases and handlers are plain Python without framework dependencies.
3. **Constructor Injection:** All collaborators and ports are injected via constructors.
4. **Clean Architecture:** Strict adherence to clean architecture and dependency inversion rules.
5. **Testability:** Every use case and handler must be unit-testable using mock ports without external infrastructure.

## Governance & Reference Documents

- [Architecture Handbook v2.0](../docs/architecture/Architecture-Handbook-v2.0.md)
- [Application Layer Design Specification v1.0](../docs/architecture/Application-Layer-Design-Specification-v1.0.md)
- [Execution Application Module Design Specification v1.0](../docs/architecture/Execution-Application-Module-Design-Specification-v1.0.md)

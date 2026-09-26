# Northstar Application

`northstar-application` is the framework-independent orchestration layer for Northstar. It coordinates Domain behavior and Application-owned ports without owning business rules or external protocol implementations.

## Implemented in Alpha v0.4

- `AnalyzeAssetUseCase` for one-asset intelligence.
- `AnalyzeAssetResult` carrying Recommendation, RecommendationExplanation, and MarketObservationContext.
- `AnalyzeWatchlistUseCase` for sequential, ordered watchlist analysis with isolated item failures.
- `MarketObservationSource` Application port for factual market context acquisition.
- Application contract tests and deterministic end-to-end workflow coverage.
- Application-owned execution contract types and outbound port interfaces as architectural contracts.

## Responsibilities

The Application layer owns:

- use-case orchestration;
- sequencing Domain operations;
- Application DTOs and outcomes;
- outbound port contracts;
- partial-failure coordination where explicitly required.

It does not own recommendation policy, market-data retrieval, persistence, HTTP, UI, or infrastructure protocols.

## Dependencies

Runtime dependencies point inward to `northstar-core`. Infrastructure implementations fulfill Application ports at composition roots.

## Future Capabilities

The following are not implemented product workflows in Alpha and remain future work:

- watchlist persistence and user ownership;
- confidence and risk assessment;
- portfolio intelligence;
- ranking and Today's Opportunities;
- execution orchestration and broker workflows;
- command, query, event, and transaction runtime handlers.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup and validation commands. Contribution rules are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

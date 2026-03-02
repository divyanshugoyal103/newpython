# AR Platform Architecture (Cloud Sync + AI-Ready)

This document defines a modular AR application architecture that is future-ready for:
- **Cloud synchronization** (offline-first, conflict-safe)
- **AI layer integration** (on-device + cloud-assisted inference)
- **Extension-based growth** (plugins/features without core rewrites)

---

## 1) Folder Structure

```text
ar-platform/
├── apps/
│   ├── mobile-ar/                  # iOS/Android AR client shell (UI + device bridge)
│   ├── web-console/                # Admin/ops/analytics UI
│   └── dev-simulator/              # Local scene/event simulator for debugging
│
├── core/
│   ├── domain/                     # Pure business logic and entities
│   │   ├── scene/
│   │   ├── anchor/
│   │   ├── asset/
│   │   ├── session/
│   │   ├── user/
│   │   └── sync/
│   ├── usecases/                   # Application services (commands + queries)
│   ├── events/                     # Event contracts and schemas
│   ├── ai/                         # AI abstractions (intent, tagging, recommendations)
│   └── extensions/                 # Extension contracts, lifecycle, capability interfaces
│
├── infra/
│   ├── local/
│   │   ├── db/                     # SQLite/Realm adapters
│   │   ├── cache/                  # Asset and model caches
│   │   └── queue/                  # Offline event/outbox queue
│   ├── cloud/
│   │   ├── api/                    # HTTP/gRPC clients
│   │   ├── realtime/               # WebSocket/pubsub adapters
│   │   ├── storage/                # Object store client (assets/models)
│   │   └── auth/                   # OAuth/JWT/token refresh
│   ├── ai/
│   │   ├── ondevice/               # Edge inference adapters
│   │   └── remote/                 # Cloud AI providers (routing, retry)
│   └── observability/
│       ├── logging/
│       ├── tracing/
│       └── metrics/
│
├── services/
│   ├── sync-service/               # Conflict resolution + state merge policies
│   ├── asset-service/              # Asset ingestion, transform, versioning
│   ├── ai-service/                 # Prompt/inference orchestration + guardrails
│   └── extension-registry/         # Extension metadata, compatibility, rollout
│
├── extensions/
│   ├── sample-measurement/         # Example extension package
│   ├── sample-guided-tour/
│   └── sample-object-notes/
│
├── schemas/
│   ├── events/                     # JSON schema/proto for events
│   ├── models/                     # Entity schema definitions
│   └── extension-manifest/         # Extension manifest schema
│
├── docs/
│   ├── architecture/
│   ├── adr/                        # Architecture decision records
│   └── runbooks/
│
├── tests/
│   ├── contract/
│   ├── integration/
│   ├── e2e/
│   └── load/
│
└── tools/
    ├── codegen/
    ├── migration/
    └── ci/
```

---

## 2) Module Responsibilities

### A. `core/domain`
- Owns the canonical business entities and invariants.
- No framework/network/storage dependencies.
- Examples:
  - `Scene`: spatial graph + metadata
  - `Anchor`: world/marker/geo anchor state
  - `Asset`: model/text/audio content with version pointer
  - `Session`: active user/device context
  - `SyncState`: vector clock/revision metadata

### B. `core/usecases`
- Implements user-intent workflows.
- Coordinates domain + ports (interfaces), never concrete infrastructure.
- Examples:
  - `CreateAnchorInScene`
  - `AttachAssetToAnchor`
  - `ResolveSyncConflict`
  - `RequestAISuggestion`

### C. `core/events`
- Defines stable event types and payload contracts.
- Supports internal event bus and external streaming.
- Enables replay/debugging and extension triggers.

### D. `core/ai`
- Defines AI capability interfaces and policy points.
- Keeps AI optional and swappable:
  - `IntentClassifier`
  - `ContextSummarizer`
  - `ContentTagger`
  - `RecommendationEngine`

### E. `core/extensions`
- Defines extension contracts, permissions, lifecycle hooks.
- Exposes capability APIs (scene read/write, UI augment, event subscribe).

### F. `infra/local`
- Device-side persistence/caching/queueing.
- Implements offline-first behavior and deterministic replays.

### G. `infra/cloud`
- Remote connectivity, auth, storage, realtime channels.
- Converts core ports to network protocols.

### H. `infra/ai`
- Pluggable AI provider adapters.
- Handles fallback routing (on-device first, cloud second), retries, and cost/latency controls.

### I. `services/*`
- Independent backend services for scale and ownership boundaries.
- Cloud sync, assets, AI orchestration, and extension registry evolve independently.

### J. `extensions/*`
- Isolated feature packages.
- Can be deployed, versioned, enabled/disabled without changing core app.

---

## 3) Data Model Design

Design principle: **event-backed, entity-cached, revision-aware**.

### Core Entities

1. **User**
   - `user_id`
   - `tenant_id`
   - `roles[]`
   - `preferences`

2. **Device**
   - `device_id`
   - `user_id`
   - `platform`
   - `capabilities` (camera depth support, gpu tier, model support)

3. **Session**
   - `session_id`
   - `user_id`, `device_id`
   - `scene_id`
   - `started_at`, `ended_at`
   - `status` (`active`, `paused`, `terminated`)

4. **Scene**
   - `scene_id`
   - `owner_id`
   - `name`, `tags[]`
   - `anchor_ids[]`
   - `revision`
   - `sync_vector` (per device/service logical clock)

5. **Anchor**
   - `anchor_id`
   - `scene_id`
   - `type` (`world`, `marker`, `geo`)
   - `transform` (position/rotation/scale)
   - `confidence`
   - `asset_refs[]`
   - `revision`

6. **Asset**
   - `asset_id`
   - `uri`
   - `content_type` (`3d_model`, `image`, `text`, `audio`)
   - `version`
   - `checksum`
   - `metadata` (size, dimensions, locale)

7. **Annotation**
   - `annotation_id`
   - `anchor_id`
   - `author_id`
   - `content`
   - `created_at`, `updated_at`
   - `ai_generated` (bool)

8. **Event**
   - `event_id`
   - `event_type`
   - `occurred_at`
   - `actor_id`
   - `entity_ref`
   - `payload`
   - `causation_id`, `correlation_id`

9. **SyncOperation (Outbox/Inbox)**
   - `op_id`
   - `entity_type`, `entity_id`
   - `op_type` (`create`, `update`, `delete`, `merge`)
   - `base_revision`, `next_revision`
   - `status` (`pending`, `in_flight`, `acked`, `conflicted`, `failed`)

10. **ExtensionManifest**
    - `extension_id`
    - `version`
    - `compatibility` (app/api min/max)
    - `permissions[]`
    - `hooks[]`
    - `config_schema`

### Storage Strategy
- **Local DB**: normalized tables for `Scene`, `Anchor`, `Asset`, plus outbox events.
- **Cloud DB**: authoritative entity state + immutable event log.
- **Object storage**: binaries (3D assets, textures, model bundles).
- **Index/search**: semantic index for AI-assisted retrieval.

### Conflict Resolution
- Prefer policy per entity:
  - `Scene metadata`: last-write-wins with audit trail
  - `Anchor transforms`: merge by newest confidence + timestamp
  - `Annotations`: append-only + edit version chain
- Always preserve original conflicting ops for forensic replay.

---

## 4) Event Flow (Step-by-Step)

Example: User adds an object in AR while offline, later syncs, then AI suggests annotations.

1. **User Action**
   - User places object at detected anchor in mobile AR app.

2. **Use Case Execution**
   - `AttachAssetToAnchor` validates scene/anchor invariants in `core/domain`.

3. **Local Commit**
   - Local DB transaction updates `Anchor.asset_refs` and `Scene.revision`.
   - Corresponding domain event (`AssetAttachedToAnchor`) appended to local outbox.

4. **UI Refresh**
   - Event bus notifies presentation layer; object is immediately visible (offline-first UX).

5. **Sync Trigger**
   - Connectivity restored; sync worker dequeues pending outbox ops.

6. **Cloud Reconciliation**
   - `sync-service` validates base revision.
   - If no conflict: persists new state + event.
   - If conflict: applies entity-specific merge policy; records conflict event.

7. **Ack + Local Resolution**
   - Cloud returns ack/reconciled entity revision.
   - Client marks op `acked` and applies canonical state if merge changed result.

8. **AI Enrichment Request**
   - New `Anchor` change emits event consumed by `ai-service`.
   - AI infers likely annotation tags/instructions from scene context.

9. **AI Suggestion Delivery**
   - Suggestion event emitted (e.g., `AISuggestionGenerated`).
   - Client receives via realtime channel, stores as pending suggestion.

10. **User Approval Loop**
   - User accepts/rejects suggestion.
   - Decision event logged to improve future recommendation ranking.

---

## 5) Extension Lifecycle Explanation

Extensions behave like sandboxed, versioned plugins.

1. **Discovery**
   - App reads extension metadata from `extension-registry` and local manifest cache.

2. **Compatibility Check**
   - Validate app version, API contract, required capabilities (AR depth, AI, network).

3. **Install / Update**
   - Download extension package and verify signature/checksum.
   - Apply migration scripts for extension-specific local data if needed.

4. **Registration**
   - Register declared hooks:
     - `onSessionStart`
     - `onAnchorCreated`
     - `onEvent(eventType)`
     - `onRenderOverlay`

5. **Activation**
   - Runtime grants permissions from manifest (least privilege).
   - Extension starts isolated execution context.

6. **Runtime Operation**
   - Extension receives events and can request approved capabilities.
   - All side effects pass through audited capability APIs.

7. **Deactivation**
   - Extension can be disabled by user, policy, or health monitor.
   - Runtime unsubscribes hooks and releases resources.

8. **Uninstall**
   - Remove package, optionally preserve user data per retention policy.
   - Emit `ExtensionUninstalled` event.

9. **Rollback**
   - Failed update can atomically roll back to prior signed version.
   - Registry marks problematic version as blocked.

### Guardrails for Future Readiness
- Capability-based permissions, not unrestricted API access.
- Contract testing for extension hooks.
- Semantic versioning + compatibility matrix.
- Feature flags for gradual rollout.
- Telemetry per extension (latency, crashes, memory).

---

## Architecture Sanity Checklist (Do Not Implement Until True)

- Core domain compiles without infra dependencies.
- Offline writes succeed even with no network.
- All mutable operations create events.
- Sync conflict policy exists per mutable entity.
- AI layer can be disabled without breaking core UX.
- Extension failure cannot crash host app.
- Every extension permission is explicit and auditable.

If any item is false, refine architecture before implementation.

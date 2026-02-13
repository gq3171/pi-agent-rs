# pi-agent-rs

[简体中文](#简体中文) | [English](#english)

---

## 简体中文

`pi-agent-rs` 是 OpenClaw 的 Rust 核心依赖库，从 `pi-mono`（TypeScript）移植而来。

### 当前状态

- 核心能力已对齐 `pi-mono@bd040072`（Agent loop、provider 行为、模型解析、资源/会话核心流程）
- 工作区质量门通过：
  - `cargo check --workspace`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`

### 仓库结构

```text
pi-agent-rs/
└── crates/
    ├── pi-agent-core
    ├── pi-agent-ai
    └── pi-coding-agent
```

### Crate 说明

#### `pi-agent-core`

- Agent 运行时核心（Agent loop / continue loop）
- `EventStream` 事件流抽象
- `AgentTool` 工具 trait 与执行契约
- 消息与上下文类型（`Message` / `ContentBlock` / `Usage` / `Model` 等）

#### `pi-agent-ai`

- 内置模型注册（含 Anthropic/OpenAI/Google/Bedrock/Azure/MiniMax 等）
- 多 provider 适配（请求构建、SSE/流式解析、工具调用映射）
- OAuth/PKCE 支持（如 Copilot、Codex、Gemini CLI 等场景）
- 统一 provider 注册与路由

#### `pi-coding-agent`

- 会话管理（JSONL 持久化、分支树、上下文重建）
- 资源系统（skills/prompts/themes/package sources）
- 工具系统（bash/read/write/edit/find/grep/ls）
- AgentSession 编排、扩展运行时、Interactive/Print/RPC 模式

### 构建与验证

```bash
cargo build --workspace
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

### 设计要点

- 会话格式兼容 `pi-mono` v3
- 基于 Tokio 的异步执行与取消
- 结构化错误处理与可追踪日志
- 跨平台进程与文件系统处理

---

## English

`pi-agent-rs` is the Rust core dependency stack for OpenClaw, ported from `pi-mono` (TypeScript).

### Current Status

- Core behavior is aligned with `pi-mono@bd040072` (agent loop, provider behavior, model resolution, and core session/resource flow)
- Workspace quality gates are passing:
  - `cargo check --workspace`
  - `cargo test --workspace`
  - `cargo clippy --workspace --all-targets -- -D warnings`

### Repository Layout

```text
pi-agent-rs/
└── crates/
    ├── pi-agent-core
    ├── pi-agent-ai
    └── pi-coding-agent
```

### Crates

#### `pi-agent-core`

- Agent runtime primitives (agent loop / continue loop)
- `EventStream` abstraction
- `AgentTool` trait and execution contracts
- Core message/context types (`Message`, `ContentBlock`, `Usage`, `Model`, etc.)

#### `pi-agent-ai`

- Built-in model registry (Anthropic/OpenAI/Google/Bedrock/Azure/MiniMax and more)
- Multi-provider adapters (request building, streaming/SSE parsing, tool-call mapping)
- OAuth/PKCE support (Copilot, Codex, Gemini CLI scenarios, etc.)
- Unified provider registration and routing

#### `pi-coding-agent`

- Session management (JSONL persistence, branching tree, context reconstruction)
- Resource system (skills/prompts/themes/package sources)
- Tooling layer (bash/read/write/edit/find/grep/ls)
- AgentSession orchestration, extension runtime, Interactive/Print/RPC modes

### Build and Verification

```bash
cargo build --workspace
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

### Design Notes

- Session format compatibility with `pi-mono` v3
- Tokio-first async execution and cancellation
- Structured error handling and traceable logs
- Cross-platform process and filesystem handling

---

## License

MIT

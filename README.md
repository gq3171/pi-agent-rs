# pi-agent-rs

OpenClaw 的 Rust 底层依赖库，从 pi-mono (TypeScript) 移植而来。

## 架构

```
pi-agent-rs/
└── crates/
    ├── pi-agent-core      # 核心层：Agent Loop、事件流、工具 trait、消息类型
    ├── pi-agent-ai        # AI 层：模型定义、Provider 适配、OAuth 认证
    └── pi-coding-agent    # SDK 层：会话管理、工具实现、设置、编排器
```

## Crate 说明

### pi-agent-core

Agent 运行时核心，提供：

- **Agent Loop** — 多轮对话循环，支持工具调用和流式输出
- **EventStream** — 异步事件流（实现 `futures::Stream`），用于 Agent 事件的生产和消费
- **AgentTool trait** — 工具接口定义，支持异步执行和取消
- **类型系统** — `Message`（User/Assistant/ToolResult）、`ContentBlock`、`Model`、`Usage` 等

### pi-agent-ai

AI 模型接入层，提供：

- **模型注册表** — 内置 700+ 模型定义（Anthropic、OpenAI、Google、AWS Bedrock、Azure）
- **Provider 适配器** — 各厂商 API 的请求构建、响应解析、SSE 流处理
- **OAuth 模块** — PKCE 授权码流程，支持 5 个 Provider（Anthropic、GitHub Copilot、OpenAI Codex、Google Antigravity、Google Gemini CLI）
- **SSE 解析器** — Server-Sent Events 流式解析

### pi-coding-agent

面向 OpenClaw 的 SDK 层，提供：

| 模块 | 功能 |
|------|------|
| `config` | 应用路径（`~/.pi/agent/`）、版本常量 |
| `settings` | 设置类型定义、深度合并、加载/保存 |
| `auth` | 多层凭证解析（运行时 > 文件 > 环境变量） |
| `session` | JSONL 会话持久化、树结构分支、上下文重建 |
| `messages` | 自定义消息类型、AgentMessage → LLM Message 转换 |
| `tools` | Bash / Read / Write / Edit / Find / Grep / Ls 工具实现 |
| `model` | 模型注册表（内置 + 自定义）、模糊匹配、循环切换 |
| `compaction` | Token 估算、会话压缩、分支摘要 |
| `resources` | Skill / Prompt / Theme 加载、包资源路径元数据、PackageManager |
| `system_prompt` | 系统提示词组装 |
| `agent_session` | AgentSession 编排器、事件系统、`create_agent_session()` 工厂 |
| `extensions` | 扩展运行时（loader / runner / wrapper）、路径发现、外部命令扩展桥接 |
| `modes` | Interactive / Print / RPC 模式运行器 |
| `slash_commands` | 内置 Slash Commands 定义 |
| `keybindings` | 默认热键与 keybindings.json 解析 |

## 构建与测试

```bash
# 编译
cargo build --workspace

# 测试（370 个测试）
cargo test --workspace

# Clippy 检查
cargo clippy --workspace
```

## 设计要点

- **会话兼容** — JSONL 会话格式对齐 `pi-mono` v3，并兼容读取旧版 Rust 会话文件
- **异步优先** — 基于 Tokio，工具执行支持 `CancellationToken` 协作取消
- **错误处理** — `thiserror` 定义结构化错误类型，工具层使用 `Box<dyn Error + Send + Sync>`
- **跨平台** — 进程管理 Unix 使用 `nix::killpg`，Windows 预留接口

## 许可证

MIT

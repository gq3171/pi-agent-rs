use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::agent_loop::{agent_loop, agent_loop_continue};
use crate::agent_types::*;
use crate::types::*;

pub struct AgentOptions {
    pub initial_state: Option<PartialAgentState>,
    pub convert_to_llm: Option<
        Arc<
            dyn Fn(&[AgentMessage]) -> Pin<Box<dyn Future<Output = Vec<Message>> + Send>>
                + Send
                + Sync,
        >,
    >,
    pub transform_context: Option<
        Arc<
            dyn Fn(
                    Vec<AgentMessage>,
                    CancellationToken,
                ) -> Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>>
                + Send
                + Sync,
        >,
    >,
    pub steering_mode: Option<QueueMode>,
    pub follow_up_mode: Option<QueueMode>,
    pub stream_fn: Option<StreamFnBox>,
    pub session_id: Option<String>,
    pub get_api_key: Option<
        Arc<dyn Fn(&str) -> Pin<Box<dyn Future<Output = Option<String>> + Send>> + Send + Sync>,
    >,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub max_retry_delay_ms: Option<u64>,
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            initial_state: None,
            convert_to_llm: None,
            transform_context: None,
            steering_mode: None,
            follow_up_mode: None,
            stream_fn: None,
            session_id: None,
            get_api_key: None,
            thinking_budgets: None,
            max_retry_delay_ms: None,
        }
    }
}

pub struct PartialAgentState {
    pub system_prompt: Option<String>,
    pub model: Option<Model>,
    pub thinking_level: Option<AgentThinkingLevel>,
    pub tools: Option<Vec<Arc<dyn AgentTool>>>,
    pub messages: Option<Vec<AgentMessage>>,
}

/// Default convertToLlm: keep only LLM-compatible messages
fn default_convert_to_llm(
    messages: &[AgentMessage],
) -> Pin<Box<dyn Future<Output = Vec<Message>> + Send>> {
    let result: Vec<Message> = messages
        .iter()
        .filter_map(|m| m.as_message().cloned())
        .collect();
    Box::pin(async move { result })
}

pub struct Agent {
    state: AgentState,
    listeners: Vec<Box<dyn Fn(&AgentEvent) + Send + Sync>>,
    cancel_token: Option<CancellationToken>,
    convert_to_llm: Arc<
        dyn Fn(&[AgentMessage]) -> Pin<Box<dyn Future<Output = Vec<Message>> + Send>> + Send + Sync,
    >,
    transform_context: Option<
        Arc<
            dyn Fn(
                    Vec<AgentMessage>,
                    CancellationToken,
                ) -> Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>>
                + Send
                + Sync,
        >,
    >,
    steering_queue: Vec<AgentMessage>,
    follow_up_queue: Vec<AgentMessage>,
    steering_mode: QueueMode,
    follow_up_mode: QueueMode,
    stream_fn: Option<StreamFnBox>,
    session_id: Option<String>,
    get_api_key: Option<
        Arc<dyn Fn(&str) -> Pin<Box<dyn Future<Output = Option<String>> + Send>> + Send + Sync>,
    >,
    thinking_budgets: Option<ThinkingBudgets>,
    max_retry_delay_ms: Option<u64>,
    running_watch: tokio::sync::watch::Sender<bool>,
}

impl Agent {
    pub fn new(opts: AgentOptions) -> Self {
        // Default model matches TS: gemini-2.5-flash-lite-preview-06-17
        let default_model = Model {
            id: "gemini-2.5-flash-lite-preview-06-17".to_string(),
            name: "Gemini 2.5 Flash Lite".to_string(),
            api: "google-generative-ai".to_string(),
            provider: "google".to_string(),
            base_url: "https://generativelanguage.googleapis.com".to_string(),
            reasoning: false,
            input: vec!["text".to_string(), "image".to_string()],
            cost: ModelCost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
            context_window: 1048576,
            max_tokens: 65536,
            headers: None,
            compat: None,
        };

        let mut state = AgentState {
            system_prompt: String::new(),
            model: default_model,
            thinking_level: AgentThinkingLevel::Off,
            tools: Vec::new(),
            messages: Vec::new(),
            is_streaming: false,
            stream_message: None,
            pending_tool_calls: HashSet::new(),
            error: None,
        };

        if let Some(partial) = opts.initial_state {
            if let Some(sp) = partial.system_prompt {
                state.system_prompt = sp;
            }
            if let Some(m) = partial.model {
                state.model = m;
            }
            if let Some(tl) = partial.thinking_level {
                state.thinking_level = tl;
            }
            if let Some(t) = partial.tools {
                state.tools = t;
            }
            if let Some(msgs) = partial.messages {
                state.messages = msgs;
            }
        }

        let convert_to_llm = opts
            .convert_to_llm
            .unwrap_or_else(|| Arc::new(|msgs: &[AgentMessage]| default_convert_to_llm(msgs)));

        Agent {
            state,
            listeners: Vec::new(),
            cancel_token: None,
            convert_to_llm,
            transform_context: opts.transform_context,
            steering_queue: Vec::new(),
            follow_up_queue: Vec::new(),
            steering_mode: opts.steering_mode.unwrap_or(QueueMode::OneAtATime),
            follow_up_mode: opts.follow_up_mode.unwrap_or(QueueMode::OneAtATime),
            stream_fn: opts.stream_fn,
            session_id: opts.session_id,
            get_api_key: opts.get_api_key,
            thinking_budgets: opts.thinking_budgets,
            max_retry_delay_ms: opts.max_retry_delay_ms,
            running_watch: tokio::sync::watch::Sender::new(false),
        }
    }

    // ---------- Accessors ----------

    pub fn state(&self) -> &AgentState {
        &self.state
    }

    pub fn subscribe(&mut self, listener: Box<dyn Fn(&AgentEvent) + Send + Sync>) -> usize {
        self.listeners.push(listener);
        self.listeners.len() - 1
    }

    pub fn unsubscribe(&mut self, index: usize) {
        if index < self.listeners.len() {
            let _ = self.listeners.remove(index);
        }
    }

    // ---------- State mutators ----------

    pub fn set_system_prompt(&mut self, v: String) {
        self.state.system_prompt = v;
    }

    pub fn set_model(&mut self, m: Model) {
        self.state.model = m;
    }

    pub fn set_thinking_level(&mut self, l: AgentThinkingLevel) {
        self.state.thinking_level = l;
    }

    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    pub fn set_session_id(&mut self, v: Option<String>) {
        self.session_id = v;
    }

    pub fn thinking_budgets(&self) -> Option<&ThinkingBudgets> {
        self.thinking_budgets.as_ref()
    }

    pub fn set_thinking_budgets(&mut self, v: Option<ThinkingBudgets>) {
        self.thinking_budgets = v;
    }

    pub fn max_retry_delay_ms(&self) -> Option<u64> {
        self.max_retry_delay_ms
    }

    pub fn set_max_retry_delay_ms(&mut self, v: Option<u64>) {
        self.max_retry_delay_ms = v;
    }

    pub fn set_steering_mode(&mut self, mode: QueueMode) {
        self.steering_mode = mode;
    }

    pub fn get_steering_mode(&self) -> &QueueMode {
        &self.steering_mode
    }

    pub fn set_follow_up_mode(&mut self, mode: QueueMode) {
        self.follow_up_mode = mode;
    }

    pub fn get_follow_up_mode(&self) -> &QueueMode {
        &self.follow_up_mode
    }

    pub fn set_tools(&mut self, t: Vec<Arc<dyn AgentTool>>) {
        self.state.tools = t;
    }

    pub fn replace_messages(&mut self, ms: Vec<AgentMessage>) {
        self.state.messages = ms;
    }

    pub fn append_message(&mut self, m: AgentMessage) {
        self.state.messages.push(m);
    }

    // ---------- Queue management ----------

    pub fn steer(&mut self, m: AgentMessage) {
        self.steering_queue.push(m);
    }

    pub fn follow_up(&mut self, m: AgentMessage) {
        self.follow_up_queue.push(m);
    }

    pub fn clear_steering_queue(&mut self) {
        self.steering_queue.clear();
    }

    pub fn clear_follow_up_queue(&mut self) {
        self.follow_up_queue.clear();
    }

    pub fn clear_all_queues(&mut self) {
        self.steering_queue.clear();
        self.follow_up_queue.clear();
    }

    pub fn has_queued_messages(&self) -> bool {
        !self.steering_queue.is_empty() || !self.follow_up_queue.is_empty()
    }

    fn dequeue_steering_messages(&mut self) -> Vec<AgentMessage> {
        match self.steering_mode {
            QueueMode::OneAtATime => {
                if self.steering_queue.is_empty() {
                    vec![]
                } else {
                    vec![self.steering_queue.remove(0)]
                }
            }
            QueueMode::All => {
                let msgs = self.steering_queue.clone();
                self.steering_queue.clear();
                msgs
            }
        }
    }

    fn dequeue_follow_up_messages(&mut self) -> Vec<AgentMessage> {
        match self.follow_up_mode {
            QueueMode::OneAtATime => {
                if self.follow_up_queue.is_empty() {
                    vec![]
                } else {
                    vec![self.follow_up_queue.remove(0)]
                }
            }
            QueueMode::All => {
                let msgs = self.follow_up_queue.clone();
                self.follow_up_queue.clear();
                msgs
            }
        }
    }

    pub fn clear_messages(&mut self) {
        self.state.messages.clear();
    }

    pub fn abort(&self) {
        if let Some(token) = &self.cancel_token {
            token.cancel();
        }
    }

    pub async fn wait_for_idle(&self) {
        let mut rx = self.running_watch.subscribe();
        // Loop: re-check after each change to avoid lost wakeups
        while *rx.borrow() {
            if rx.changed().await.is_err() {
                break; // Sender dropped
            }
        }
    }

    pub fn reset(&mut self) {
        self.state.messages.clear();
        self.state.is_streaming = false;
        self.state.stream_message = None;
        self.state.pending_tool_calls.clear();
        self.state.error = None;
        self.steering_queue.clear();
        self.follow_up_queue.clear();
    }

    // ---------- Prompt ----------

    pub async fn prompt_text(&mut self, text: impl Into<String>) -> Result<(), String> {
        let msg = AgentMessage::user(text);
        self.prompt(vec![msg]).await
    }

    pub async fn prompt(&mut self, messages: Vec<AgentMessage>) -> Result<(), String> {
        if self.state.is_streaming {
            return Err(
                "Agent is already processing a prompt. Use steer() or follow_up() to queue messages."
                    .to_string(),
            );
        }

        if self.state.model.id.is_empty() {
            return Err("No model configured".to_string());
        }

        self.run_loop(Some(messages), false).await;
        Ok(())
    }

    pub async fn continue_run(&mut self) -> Result<(), String> {
        if self.state.is_streaming {
            return Err(
                "Agent is already processing. Wait for completion before continuing.".to_string(),
            );
        }

        if self.state.messages.is_empty() {
            return Err("No messages to continue from".to_string());
        }

        if let Some(last) = self.state.messages.last() {
            if last.role() == Some("assistant") {
                let queued_steering = self.dequeue_steering_messages();
                if !queued_steering.is_empty() {
                    self.run_loop(Some(queued_steering), true).await;
                    return Ok(());
                }

                let queued_follow_up = self.dequeue_follow_up_messages();
                if !queued_follow_up.is_empty() {
                    self.run_loop(Some(queued_follow_up), false).await;
                    return Ok(());
                }

                return Err("Cannot continue from message role: assistant".to_string());
            }
        }

        self.run_loop(None, false).await;
        Ok(())
    }

    async fn run_loop(
        &mut self,
        messages: Option<Vec<AgentMessage>>,
        skip_initial_steering_poll: bool,
    ) {
        let cancel = CancellationToken::new();
        self.cancel_token = Some(cancel.clone());
        self.state.is_streaming = true;
        self.state.stream_message = None;
        self.state.error = None;
        let _ = self.running_watch.send(true);

        let reasoning = self.state.thinking_level.to_thinking_level();

        let context = AgentContext {
            system_prompt: self.state.system_prompt.clone(),
            messages: self.state.messages.clone(),
            tools: self.state.tools.clone(),
        };

        // Build steering/follow-up closures using Arc<Mutex> for shared mutable access
        let steering_queue = Arc::new(std::sync::Mutex::new(self.steering_queue.clone()));
        let follow_up_queue = Arc::new(std::sync::Mutex::new(self.follow_up_queue.clone()));
        let steering_mode = self.steering_mode.clone();
        let follow_up_mode = self.follow_up_mode.clone();
        let skip_flag = Arc::new(std::sync::Mutex::new(skip_initial_steering_poll));

        let steering_queue_clone = steering_queue.clone();
        let skip_flag_clone = skip_flag.clone();
        let get_steering: Arc<
            dyn Fn() -> Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>> + Send + Sync,
        > = Arc::new(move || {
            let sq = steering_queue_clone.clone();
            let sf = skip_flag_clone.clone();
            let mode = steering_mode.clone();
            Box::pin(async move {
                {
                    let mut skip = sf.lock().unwrap();
                    if *skip {
                        *skip = false;
                        return vec![];
                    }
                }
                let mut queue = sq.lock().unwrap();
                match mode {
                    QueueMode::OneAtATime => {
                        if queue.is_empty() {
                            vec![]
                        } else {
                            vec![queue.remove(0)]
                        }
                    }
                    QueueMode::All => {
                        let msgs = queue.clone();
                        queue.clear();
                        msgs
                    }
                }
            })
        });

        let follow_up_queue_clone = follow_up_queue.clone();
        let get_follow_up: Arc<
            dyn Fn() -> Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>> + Send + Sync,
        > = Arc::new(move || {
            let fq = follow_up_queue_clone.clone();
            let mode = follow_up_mode.clone();
            Box::pin(async move {
                let mut queue = fq.lock().unwrap();
                match mode {
                    QueueMode::OneAtATime => {
                        if queue.is_empty() {
                            vec![]
                        } else {
                            vec![queue.remove(0)]
                        }
                    }
                    QueueMode::All => {
                        let msgs = queue.clone();
                        queue.clear();
                        msgs
                    }
                }
            })
        });

        let config = AgentLoopConfig {
            model: self.state.model.clone(),
            reasoning,
            session_id: self.session_id.clone(),
            thinking_budgets: self.thinking_budgets.clone(),
            max_retry_delay_ms: self.max_retry_delay_ms,
            temperature: None,
            max_tokens: None,
            api_key: None,
            cache_retention: None,
            headers: None,
            convert_to_llm: self.convert_to_llm.clone(),
            transform_context: self.transform_context.clone(),
            get_api_key: self.get_api_key.clone(),
            get_steering_messages: Some(get_steering),
            get_follow_up_messages: Some(get_follow_up),
        };

        let result: Result<(), String> = async {
            let mut event_stream = if let Some(msgs) = messages {
                Box::pin(agent_loop(
                    msgs,
                    context,
                    config,
                    cancel.clone(),
                    self.stream_fn.clone(),
                ))
            } else {
                Box::pin(
                    agent_loop_continue(context, config, cancel.clone(), self.stream_fn.clone())
                        .map_err(|e| e)?,
                )
            };

            while let Some(event) = event_stream.next().await {
                match &event {
                    AgentEvent::MessageStart { message } => {
                        self.state.stream_message = Some(message.clone());
                    }
                    AgentEvent::MessageUpdate { message, .. } => {
                        self.state.stream_message = Some(message.clone());
                    }
                    AgentEvent::MessageEnd { message } => {
                        self.state.stream_message = None;
                        self.append_message(message.clone());
                    }
                    AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                        self.state.pending_tool_calls.insert(tool_call_id.clone());
                    }
                    AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                        self.state.pending_tool_calls.remove(tool_call_id);
                    }
                    AgentEvent::TurnEnd { message, .. } => {
                        if let Some(msg) = message.as_message() {
                            if let Some(assistant) = msg.as_assistant() {
                                if let Some(err_msg) = &assistant.error_message {
                                    self.state.error = Some(err_msg.clone());
                                }
                            }
                        }
                    }
                    AgentEvent::AgentEnd { .. } => {
                        self.state.is_streaming = false;
                        self.state.stream_message = None;
                    }
                    _ => {}
                }

                self.emit(&event);
            }

            Ok(())
        }
        .await;

        if let Err(err) = result {
            let error_msg = AssistantMessage {
                content: vec![ContentBlock::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                })],
                api: self.state.model.api.clone(),
                provider: self.state.model.provider.clone(),
                model: self.state.model.id.clone(),
                usage: Usage::default(),
                stop_reason: if self.cancel_token.as_ref().is_some_and(|t| t.is_cancelled()) {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                },
                error_message: Some(err.clone()),
                timestamp: chrono::Utc::now().timestamp_millis(),
            };
            self.append_message(error_msg.clone().into());
            self.state.error = Some(err);
            self.emit(&AgentEvent::AgentEnd {
                messages: vec![error_msg.into()],
            });
        }

        // Cleanup
        self.state.is_streaming = false;
        self.state.stream_message = None;
        self.state.pending_tool_calls.clear();
        self.cancel_token = None;
        // Sync queues back
        self.steering_queue = steering_queue.lock().unwrap().clone();
        self.follow_up_queue = follow_up_queue.lock().unwrap().clone();

        let _ = self.running_watch.send(false);
    }

    fn emit(&self, event: &AgentEvent) {
        for listener in &self.listeners {
            listener(event);
        }
    }
}

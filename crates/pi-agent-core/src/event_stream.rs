use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use futures::Stream;
use tokio::sync::oneshot;

use crate::types::{AssistantMessage, AssistantMessageEvent};

struct EventStreamInner<T, R> {
    queue: VecDeque<T>,
    done: bool,
    waiters: Vec<Waker>,
    result_sender: Option<oneshot::Sender<R>>,
}

pub struct EventStream<T, R> {
    inner: Arc<Mutex<EventStreamInner<T, R>>>,
    result_receiver: Arc<tokio::sync::Mutex<Option<oneshot::Receiver<R>>>>,
    is_complete: Arc<dyn Fn(&T) -> bool + Send + Sync>,
    extract_result: Arc<dyn Fn(&T) -> R + Send + Sync>,
}

impl<T, R> Clone for EventStream<T, R> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            result_receiver: self.result_receiver.clone(),
            is_complete: self.is_complete.clone(),
            extract_result: self.extract_result.clone(),
        }
    }
}

impl<T: Send + 'static, R: Send + 'static> EventStream<T, R> {
    pub fn new(
        is_complete: impl Fn(&T) -> bool + Send + Sync + 'static,
        extract_result: impl Fn(&T) -> R + Send + Sync + 'static,
    ) -> Self {
        let (tx, rx) = oneshot::channel();
        Self {
            inner: Arc::new(Mutex::new(EventStreamInner {
                queue: VecDeque::new(),
                done: false,
                waiters: Vec::new(),
                result_sender: Some(tx),
            })),
            result_receiver: Arc::new(tokio::sync::Mutex::new(Some(rx))),
            is_complete: Arc::new(is_complete),
            extract_result: Arc::new(extract_result),
        }
    }

    pub fn push(&self, event: T) {
        let mut inner = self.inner.lock().unwrap();
        if inner.done {
            return;
        }

        if (self.is_complete)(&event) {
            inner.done = true;
            let result = (self.extract_result)(&event);
            if let Some(sender) = inner.result_sender.take() {
                let _ = sender.send(result);
            }
        }

        inner.queue.push_back(event);

        // Wake all waiters
        for waker in inner.waiters.drain(..) {
            waker.wake();
        }
    }

    /// Check if the stream is marked as done.
    pub fn is_done(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.done
    }

    /// End the stream with an optional result.
    /// If result is None, the stream is marked as done but no result is sent
    /// (matching TS behavior where `end()` can be called without arguments).
    pub fn end(&self, result: Option<R>) {
        let mut inner = self.inner.lock().unwrap();
        inner.done = true;
        if let Some(r) = result {
            if let Some(sender) = inner.result_sender.take() {
                let _ = sender.send(r);
            }
        } else {
            // Drop the sender so result() returns None instead of hanging
            inner.result_sender.take();
        }
        // Wake all waiters
        for waker in inner.waiters.drain(..) {
            waker.wake();
        }
    }

    /// Get the final result of the stream.
    ///
    /// Returns `None` if:
    /// - `end(None)` was called (stream ended without a result)
    /// - `result()` was already called once (receiver consumed)
    /// - The result sender was dropped without sending
    pub async fn result(&self) -> Option<R> {
        let mut guard = self.result_receiver.lock().await;
        if let Some(rx) = guard.take() {
            rx.await.ok()
        } else {
            None
        }
    }
}

impl<T: Send + 'static, R: Send + 'static> Stream for EventStream<T, R> {
    type Item = T;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut inner = self.inner.lock().unwrap();

        if let Some(event) = inner.queue.pop_front() {
            Poll::Ready(Some(event))
        } else if inner.done {
            Poll::Ready(None)
        } else {
            inner.waiters.push(cx.waker().clone());
            Poll::Pending
        }
    }
}

// ---------- AssistantMessageEventStream ----------

pub type AssistantMessageEventStream = EventStream<AssistantMessageEvent, AssistantMessage>;

pub fn create_assistant_message_event_stream() -> AssistantMessageEventStream {
    EventStream::new(
        |event: &AssistantMessageEvent| event.is_complete(),
        |event: &AssistantMessageEvent| match event {
            AssistantMessageEvent::Done { message, .. } => message.clone(),
            AssistantMessageEvent::Error { error, .. } => error.clone(),
            _ => panic!("Unexpected event type for final result"),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use futures::StreamExt;

    fn make_test_assistant_message() -> AssistantMessage {
        AssistantMessage {
            content: vec![ContentBlock::Text(TextContent {
                text: "Hello".to_string(),
                text_signature: None,
            })],
            api: "test".to_string(),
            provider: "test".to_string(),
            model: "test".to_string(),
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        }
    }

    #[tokio::test]
    async fn test_push_and_consume() {
        let stream = create_assistant_message_event_stream();
        let msg = make_test_assistant_message();

        let producer = stream.clone();
        let msg_clone = msg.clone();
        tokio::spawn(async move {
            producer.push(AssistantMessageEvent::Start {
                partial: msg_clone.clone(),
            });
            producer.push(AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "Hi".to_string(),
                partial: msg_clone.clone(),
            });
            producer.push(AssistantMessageEvent::TextEnd {
                content_index: 0,
                content: "Hello".to_string(),
                partial: msg_clone.clone(),
            });
            producer.push(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message: msg_clone,
            });
        });

        let mut stream_pin = Box::pin(stream.clone());
        let mut count = 0;
        while let Some(_event) = stream_pin.next().await {
            count += 1;
        }
        assert_eq!(count, 4);
    }

    #[tokio::test]
    async fn test_result() {
        let stream = create_assistant_message_event_stream();
        let msg = make_test_assistant_message();

        let producer = stream.clone();
        let msg_clone = msg.clone();
        tokio::spawn(async move {
            producer.push(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message: msg_clone,
            });
        });

        let result = stream.result().await.expect("result should exist");
        assert_eq!(result.model, "test");
    }

    #[tokio::test]
    async fn test_result_returns_none_on_end_without_result() {
        let stream: EventStream<AssistantMessageEvent, AssistantMessage> = EventStream::new(
            |event: &AssistantMessageEvent| event.is_complete(),
            |event: &AssistantMessageEvent| match event {
                AssistantMessageEvent::Done { message, .. } => message.clone(),
                AssistantMessageEvent::Error { error, .. } => error.clone(),
                _ => panic!("unexpected"),
            },
        );

        stream.end(None);
        assert!(stream.result().await.is_none());
    }

    #[tokio::test]
    async fn test_result_returns_none_on_double_call() {
        let stream = create_assistant_message_event_stream();
        let msg = make_test_assistant_message();

        stream.push(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: msg,
        });

        assert!(stream.result().await.is_some());
        assert!(stream.result().await.is_none());
    }

    #[tokio::test]
    async fn test_push_after_done_ignored() {
        let stream = create_assistant_message_event_stream();
        let msg = make_test_assistant_message();

        stream.push(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: msg.clone(),
        });
        // This should be silently ignored
        stream.push(AssistantMessageEvent::Start {
            partial: msg.clone(),
        });

        let mut stream_pin = Box::pin(stream.clone());
        let mut count = 0;
        while let Some(_) = stream_pin.next().await {
            count += 1;
        }
        // Only the Done event should be yielded
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_end_with_result() {
        let stream: EventStream<AssistantMessageEvent, AssistantMessage> = EventStream::new(
            |event: &AssistantMessageEvent| event.is_complete(),
            |event: &AssistantMessageEvent| match event {
                AssistantMessageEvent::Done { message, .. } => message.clone(),
                AssistantMessageEvent::Error { error, .. } => error.clone(),
                _ => panic!("unexpected"),
            },
        );
        let msg = make_test_assistant_message();

        stream.push(AssistantMessageEvent::Start {
            partial: msg.clone(),
        });
        stream.end(Some(msg.clone()));

        let mut stream_pin = Box::pin(stream.clone());
        let mut count = 0;
        while let Some(_) = stream_pin.next().await {
            count += 1;
        }
        // Only the Start event should be yielded (end doesn't push events)
        assert_eq!(count, 1);

        let result = stream.result().await.expect("result should exist");
        assert_eq!(result.model, "test");
    }
}

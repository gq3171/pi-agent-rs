/// Server-Sent Events (SSE) parser.
///
/// Parses `event: xxx\ndata: yyy\n\n` format from streaming HTTP responses.

/// Maximum buffer size (4 MB) to prevent unbounded memory growth from
/// malformed/malicious streams that never send newlines.
const MAX_BUFFER_SIZE: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event_type: String,
    pub data: String,
}

pub struct SseParser {
    buffer: String,
    current_event_type: String,
    current_data: Vec<String>,
}

impl SseParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            current_event_type: String::new(),
            current_data: Vec::new(),
        }
    }

    /// Feed a chunk of text and return any complete SSE events.
    /// Returns Err if the internal buffer exceeds the maximum size.
    pub fn feed(&mut self, chunk: &str) -> Result<Vec<SseEvent>, String> {
        self.buffer.push_str(chunk);

        // Guard against unbounded buffer growth
        if self.buffer.len() > MAX_BUFFER_SIZE {
            self.buffer.clear();
            self.current_data.clear();
            self.current_event_type.clear();
            return Err(format!(
                "SSE buffer exceeded maximum size of {} bytes",
                MAX_BUFFER_SIZE
            ));
        }

        let mut events = Vec::new();

        loop {
            if let Some(newline_pos) = self.buffer.find('\n') {
                let line = self.buffer[..newline_pos].to_string();
                self.buffer = self.buffer[newline_pos + 1..].to_string();

                let line = line.trim_end_matches('\r');

                if line.is_empty() {
                    // Empty line = end of event
                    if let Some(event) = self.flush_current() {
                        events.push(event);
                    }
                } else if let Some(value) = line.strip_prefix("event:") {
                    self.current_event_type = value.trim().to_string();
                } else if let Some(value) = line.strip_prefix("data:") {
                    self.current_data.push(value.trim().to_string());
                } else if line.starts_with(':') {
                    // Comment, ignore
                } else if let Some(value) = line.strip_prefix("id:") {
                    // Event ID, currently not used
                    let _ = value;
                } else if let Some(value) = line.strip_prefix("retry:") {
                    // Retry interval, currently not used
                    let _ = value;
                }
            } else {
                break;
            }
        }

        Ok(events)
    }

    /// Flush any remaining buffered event data. Call this when the stream
    /// ends to avoid losing the last event if it wasn't followed by an empty line.
    pub fn finish(&mut self) -> Option<SseEvent> {
        // Process any remaining complete lines in the buffer
        let remaining = std::mem::take(&mut self.buffer);
        if !remaining.is_empty() {
            for line in remaining.lines() {
                let line = line.trim_end_matches('\r');
                if line.is_empty() {
                    // Empty line found — flush is handled below
                } else if let Some(value) = line.strip_prefix("event:") {
                    self.current_event_type = value.trim().to_string();
                } else if let Some(value) = line.strip_prefix("data:") {
                    self.current_data.push(value.trim().to_string());
                }
                // Other field types ignored at finish
            }
        }

        self.flush_current()
    }

    /// Flush the current accumulated event fields into an SseEvent, if any data exists.
    fn flush_current(&mut self) -> Option<SseEvent> {
        if self.current_data.is_empty() && self.current_event_type.is_empty() {
            return None;
        }

        let event = SseEvent {
            event_type: if self.current_event_type.is_empty() {
                "message".to_string()
            } else {
                std::mem::take(&mut self.current_event_type)
            },
            data: self.current_data.join("\n"),
        };
        self.current_data.clear();
        self.current_event_type.clear();
        Some(event)
    }
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_event() {
        let mut parser = SseParser::new();
        let events = parser
            .feed("event: message_start\ndata: {\"type\": \"message_start\"}\n\n")
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message_start");
        assert_eq!(events[0].data, "{\"type\": \"message_start\"}");
    }

    #[test]
    fn test_multiple_events() {
        let mut parser = SseParser::new();
        let events = parser
            .feed("event: a\ndata: 1\n\nevent: b\ndata: 2\n\n")
            .unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "a");
        assert_eq!(events[0].data, "1");
        assert_eq!(events[1].event_type, "b");
        assert_eq!(events[1].data, "2");
    }

    #[test]
    fn test_chunked_input() {
        let mut parser = SseParser::new();

        let events1 = parser.feed("event: test\n").unwrap();
        assert_eq!(events1.len(), 0);

        let events2 = parser.feed("data: hello\n").unwrap();
        assert_eq!(events2.len(), 0);

        let events3 = parser.feed("\n").unwrap();
        assert_eq!(events3.len(), 1);
        assert_eq!(events3[0].event_type, "test");
        assert_eq!(events3[0].data, "hello");
    }

    #[test]
    fn test_data_only() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: hello world\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message");
        assert_eq!(events[0].data, "hello world");
    }

    #[test]
    fn test_multi_line_data() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: line1\ndata: line2\n\n").unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2");
    }

    #[test]
    fn test_comments_ignored() {
        let mut parser = SseParser::new();
        let events = parser
            .feed(": this is a comment\nevent: test\ndata: value\n\n")
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "test");
    }

    #[test]
    fn test_finish_flushes_last_event() {
        let mut parser = SseParser::new();
        // Feed event without trailing empty line
        let events = parser.feed("event: done\ndata: final").unwrap();
        assert_eq!(events.len(), 0);

        // finish() should flush the remaining event
        let last = parser.finish();
        assert!(last.is_some());
        let last = last.unwrap();
        assert_eq!(last.event_type, "done");
        assert_eq!(last.data, "final");
    }

    #[test]
    fn test_finish_returns_none_when_empty() {
        let mut parser = SseParser::new();
        let events = parser.feed("event: test\ndata: hello\n\n").unwrap();
        assert_eq!(events.len(), 1);

        // No remaining data
        assert!(parser.finish().is_none());
    }

    #[test]
    fn test_buffer_overflow_returns_error() {
        let mut parser = SseParser::new();
        // Feed a huge chunk without newlines
        let huge = "x".repeat(MAX_BUFFER_SIZE + 1);
        let result = parser.feed(&huge);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeded maximum size"));
    }
}

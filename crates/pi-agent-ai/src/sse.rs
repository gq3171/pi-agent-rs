/// Server-Sent Events (SSE) parser.
///
/// Parses `event: xxx\ndata: yyy\n\n` format from streaming HTTP responses.

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
    pub fn feed(&mut self, chunk: &str) -> Vec<SseEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();

        loop {
            if let Some(newline_pos) = self.buffer.find('\n') {
                let line = self.buffer[..newline_pos].to_string();
                self.buffer = self.buffer[newline_pos + 1..].to_string();

                let line = line.trim_end_matches('\r');

                if line.is_empty() {
                    // Empty line = end of event
                    if !self.current_data.is_empty() || !self.current_event_type.is_empty() {
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

        events
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
        let events = parser.feed("event: message_start\ndata: {\"type\": \"message_start\"}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message_start");
        assert_eq!(events[0].data, "{\"type\": \"message_start\"}");
    }

    #[test]
    fn test_multiple_events() {
        let mut parser = SseParser::new();
        let events = parser.feed(
            "event: a\ndata: 1\n\nevent: b\ndata: 2\n\n",
        );
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "a");
        assert_eq!(events[0].data, "1");
        assert_eq!(events[1].event_type, "b");
        assert_eq!(events[1].data, "2");
    }

    #[test]
    fn test_chunked_input() {
        let mut parser = SseParser::new();

        let events1 = parser.feed("event: test\n");
        assert_eq!(events1.len(), 0);

        let events2 = parser.feed("data: hello\n");
        assert_eq!(events2.len(), 0);

        let events3 = parser.feed("\n");
        assert_eq!(events3.len(), 1);
        assert_eq!(events3[0].event_type, "test");
        assert_eq!(events3[0].data, "hello");
    }

    #[test]
    fn test_data_only() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: hello world\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message");
        assert_eq!(events[0].data, "hello world");
    }

    #[test]
    fn test_multi_line_data() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: line1\ndata: line2\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "line1\nline2");
    }

    #[test]
    fn test_comments_ignored() {
        let mut parser = SseParser::new();
        let events = parser.feed(": this is a comment\nevent: test\ndata: value\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "test");
    }
}

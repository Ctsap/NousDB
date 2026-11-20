//! Chat message type.

/// A chat message with a role.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Message {
    /// System instructions.
    System(String),
    /// User input.
    User(String),
    /// Model output.
    Assistant(String),
}

impl Message {
    /// Returns the message text.
    pub fn as_str(&self) -> &str {
        match self {
            Message::System(s) => s,
            Message::User(s) => s,
            Message::Assistant(s) => s,
        }
    }

    /// Returns the role name: `"system"`, `"user"` or `"assistant"`.
    pub fn message_type(&self) -> &str {
        match self {
            Message::System(_) => "system",
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
        }
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Message::System(s) => write!(f, "System: {}", s),
            Message::User(s) => write!(f, "User: {}", s),
            Message::Assistant(s) => write!(f, "Assistant: {}", s),
        }
    }
}

impl From<&str> for Message {
    fn from(s: &str) -> Self {
        Message::User(s.to_string())
    }
}

impl From<String> for Message {
    fn from(s: String) -> Self {
        Message::User(s)
    }
}

impl From<Message> for String {
    fn from(val: Message) -> Self {
        match val {
            Message::System(s) => s,
            Message::User(s) => s,
            Message::Assistant(s) => s,
        }
    }
}

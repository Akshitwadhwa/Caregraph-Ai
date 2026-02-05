const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send");

function addMessage(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;

  const roleEl = document.createElement("div");
  roleEl.className = "role";
  roleEl.textContent = role === "user" ? "You" : "CareGraph";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrapper.appendChild(roleEl);
  wrapper.appendChild(bubble);
  chat.appendChild(wrapper);
  chat.scrollTop = chat.scrollHeight;
}

function setLoading(isLoading) {
  sendButton.disabled = isLoading;
  sendButton.textContent = isLoading ? "Thinking..." : "Ask";
}

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  addMessage("user", question);
  questionInput.value = "";
  setLoading(true);

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    if (!response.ok) {
      addMessage("assistant", data.error || "Something went wrong.");
      const last = chat.lastChild?.querySelector(".bubble");
      if (last) last.classList.add("error");
    } else {
      addMessage("assistant", data.answer);
    }
  } catch (error) {
    addMessage("assistant", "Network error. Please try again.");
    const last = chat.lastChild?.querySelector(".bubble");
    if (last) last.classList.add("error");
  } finally {
    setLoading(false);
  }
});

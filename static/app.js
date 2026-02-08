const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send");
const uploader = document.getElementById("uploader");
const pdfInput = document.getElementById("pdfs");
const imageInput = document.getElementById("image");
const uploadButton = document.getElementById("upload");

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

function setUploading(isUploading) {
  uploadButton.disabled = isUploading;
  uploadButton.textContent = isUploading ? "Uploading..." : "Upload";
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
      const text = [
        "Disclaimer:",
        data.disclaimer || "",
        "",
        "Rationale:",
        data.rationale || "",
        "",
        "OK Report:",
        data.ok_report || "",
      ].join("\n");
      addMessage("assistant", text.trim());
    }
  } catch (error) {
    addMessage("assistant", "Network error. Please try again.");
    const last = chat.lastChild?.querySelector(".bubble");
    if (last) last.classList.add("error");
  } finally {
    setLoading(false);
  }
});

uploader.addEventListener("submit", async (event) => {
  event.preventDefault();

  const pdfs = pdfInput.files;
  const image = imageInput.files[0];

  if (!pdfs.length && !image) {
    addMessage("assistant", "Please select a PDF and/or image to upload.");
    const last = chat.lastChild?.querySelector(".bubble");
    if (last) last.classList.add("error");
    return;
  }

  setUploading(true);

  const form = new FormData();
  for (const pdf of pdfs) {
    form.append("pdfs", pdf);
  }
  if (image) {
    form.append("image", image);
  }

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: form,
    });
    const data = await response.json();
    if (!response.ok) {
      addMessage("assistant", data.error || "Upload failed.");
      const last = chat.lastChild?.querySelector(".bubble");
      if (last) last.classList.add("error");
    } else if (data.status) {
      addMessage("assistant", data.status);
    } else {
      const text = [
        "Disclaimer:",
        data.disclaimer || "",
        "",
        "Rationale:",
        data.rationale || "",
        "",
        "OK Report:",
        data.ok_report || "",
      ].join("\n");
      addMessage("assistant", text.trim());
    }
  } catch (error) {
    addMessage("assistant", "Network error. Please try again.");
    const last = chat.lastChild?.querySelector(".bubble");
    if (last) last.classList.add("error");
  } finally {
    setUploading(false);
    pdfInput.value = "";
    imageInput.value = "";
  }
});

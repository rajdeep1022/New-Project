console.log("JS Loaded ✅");

async function uploadAudio() {
    console.log("🔥 Button clicked");

    const fileInput = document.getElementById("audioFile");
    const emotionText = document.getElementById("emotion");
    const scoreText = document.getElementById("score");
    const statusText = document.getElementById("status");

    if (!fileInput.files.length) {
        statusText.innerText = "Please upload file";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        statusText.innerText = "Analyzing...";

        const res = await fetch("http://127.0.0.1:5000/detect-panic", {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        console.log("Response:", data);

        emotionText.innerText = data.emotion;
        scoreText.innerText = data.panic_score + "%";

        statusText.innerText = data.panic_score > 60 ? "🚨 Panic" : "✅ Safe";

    } catch (err) {
        console.error(err);
        statusText.innerText = "Error connecting to server";
    }
}
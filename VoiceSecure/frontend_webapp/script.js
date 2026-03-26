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

        if (data.error) {
            statusText.innerText = "Error: " + data.error;
            return;
        }

        // ✅ UPDATE UI
        emotionText.innerText = data.emotion;
        scoreText.innerText = data.panic_score + "%";

        if (data.panic_score > 60) {
            statusText.innerText = "🚨 Panic Detected!";
            statusText.style.color = "red";
        } else {
            statusText.innerText = "✅ Safe";
            statusText.style.color = "green";
        }

    } catch (err) {
        console.error(err);
        statusText.innerText = "Server error";
    }
}
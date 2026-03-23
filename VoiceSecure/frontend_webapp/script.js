function goToDashboard() {
  window.location.href = "dashboard.html";
}

function detectPanic() {
  const fileInput = document.getElementById("audioFile");

  if (!fileInput.files.length) {
    alert("Please upload an audio file first!");
    return;
  }

  const file = fileInput.files[0];

  // For now: simulate result
  let panicScore = Math.floor(Math.random() * 100);

  document.getElementById("emotion").innerText = panicScore > 60 ? "Fear" : "Normal";
  document.getElementById("score").innerText = panicScore + "%";

  if (panicScore > 70) {
    document.getElementById("status").innerText = "🚨 ALERT TRIGGERED";
  } else {
    document.getElementById("status").innerText = "Safe";
  }
}
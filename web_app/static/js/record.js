let mediaRecorder;
let audioChunks = [];

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            audioChunks = [];

            const button = document.querySelector(".mic-button");
            button.disabled = true; // Disable the button while recording

            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener("stop", () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/mp3' });
                const formData = new FormData();
                formData.append("audio", audioBlob);

                fetch("/upload_audio", {
                    method: "POST",
                    body: formData,
                }).then(response => response.json())
                  .then(data => {
                    console.log(data);
                    button.disabled = false; // Re-enable the button after recording
                }).catch(error => console.error(error));
            });

            // Setup to stop recording after a fixed time (adjust or make it manual as needed)
            setTimeout(() => {
                mediaRecorder.stop();
                stream.getTracks().forEach(track => track.stop()); // Stop the stream
            }, 5000);
        }).catch(error => console.log(error));
}

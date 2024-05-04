let mediaRecorder;
let audioChunks = [];

function startRecording() {
    console.log("Attempting to start recording...");  // Debugging: log start attempt
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            console.log("Recording started.");  // Debugging: confirm recording started
            audioChunks = [];

            const button = document.querySelector(".mic-button");
            button.disabled = true;

            mediaRecorder.addEventListener("dataavailable", event => {
                console.log("Data available from recording.");  // Debugging: log data availability
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener("stop", () => {
                console.log("Recording stopped, processing audio.");  // Debugging: log stopping
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const formData = new FormData();
                formData.append("audio", audioBlob);

                fetch("/upload_audio", {
                    method: "POST",
                    body: formData,
                }).then(response => response.json())
                  .then(data => {
                    console.log("Server response:", data);  // Debugging: log server response
                    button.disabled = false;
                }).catch(error => console.error("Error uploading file:", error));
            });

            setTimeout(() => {
                mediaRecorder.stop();
                stream.getTracks().forEach(track => track.stop());
            }, 5000);
        }).catch(error => console.error("Error accessing media devices:", error));
}

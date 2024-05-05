const video = document.getElementById('video');

// Prompt user to start video stream
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.play();
        setInterval(sendFrame, 5000); // Send frames every 5 seconds
    })
    .catch(err => console.error("error:", err));

function sendFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const dataURL = canvas.toDataURL('image/jpeg');

    fetch('/process_frame', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataURL })
    })
    .then(response => response.json())
    .then(data => handleResponse(data))
    .catch(error => console.error('Error:', error));
}

function handleResponse(data) {
    if(data.action === 'like') {
        console.log('Liked');
        // Handle liked response, possibly load next images
    } else if(data.action === 'dislike') {
        console.log('Disliked');
        // Handle disliked response, possibly load next images
    }
}

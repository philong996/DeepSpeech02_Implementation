var recorder, gumStream;
var recordButton = document.getElementById("recordButton");
// recordButton.addEventListener("click", toggleRecording);

var aud = document.getElementById("myAudio");
aud.play()
// var start = document.getElementById("sentence").getAttribute("md-start");
// var dur = document.getElementById("sentence").getAttribute("md-dur");

function clickSentence(start, dur) {
    
    stop = (start + dur);
    aud.currentTime=start;
    aud.play();
    setTimeout(function() {
        aud.pause();
    }, dur * 1000);
};


function toggleRecording(id) {
    if (recorder && recorder.state == "recording") {
        recorder.stop();
        gumStream.getAudioTracks()[0].stop();
    } else {
        navigator.mediaDevices.getUserMedia({
            audio: true
        }).then(function(stream) {
            gumStream = stream;
            const audioChunks = [];
            recorder = new MediaRecorder(stream);
            recorder.ondataavailable = function(e) {
                var url = URL.createObjectURL(e.data);
                audioChunks.push(e.data);
                var preview = document.createElement('audio');
                preview.controls = true;
                preview.src = url;
                document.getElementById("record-" + id).innerHTML="";
                document.getElementById("record-" + id).parentElement.setAttribute("style", "max-height: 220px;");
                document.getElementById("record-" + id).appendChild(preview);
                document.getElementById("spinner-" + id).style.visibility = "visible";
                document.getElementById("icon-" + id).style.visibility = "hidden";
                const audioBlob = new Blob(audioChunks, {'type': 'audio/wav;'});
                fetch(`${window.origin}/process_voice`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "audio/wav"
                    },
                    body: audioBlob
                }).then((response) => response.json())
                .then((result) => {
                    console.log("Request complete! response:", result);
                    document.getElementById('result-' + id).innerHTML = result.message;
                    document.getElementById("spinner-" + id).style.visibility = "hidden";
                });
            };
            recorder.start();
        });
    }
}
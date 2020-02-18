var recorder, gumStream;
var recordButton = document.getElementById("recordButton");
// recordButton.addEventListener("click", toggleRecording);


// var video_idx = document.getElementById("video").getAttribute("video-id");
// console.log("Video id:",video_idx)


// var aud = document.getElementById("myAudio");
// aud.play();
// var start = document.getElementById("sentence").getAttribute("md-start");
// var dur = document.getElementById("sentence").getAttribute("md-dur");


// function clickSentence(start, dur) {
//     stop = (start + dur);
//     aud.currentTime=start;
//     aud.play();
//     setTimeout(function() {
//         aud.pause();
//     }, dur * 1000);
// };

function clickSentence(idx) {
    var aud = new Audio("https://audio.tatoeba.org/sentences/eng/" + idx + ".mp3");
    aud.play();
}


function get_wrong(speak,label) {
    var out = "";
    var missing = "";
    var cnt = 0;
    var diff = difflib.ndiff(speak, label);
    for (i = 0; i < diff.length; i++) {
        if (diff[i].includes('-')) {
            out += speak.charAt(cnt).fontcolor("red");
            cnt += 1;
        } else if (diff[i].includes('+')) {
            missing += diff[i].charAt(diff[i].length - 1).fontcolor("violet");
            // cnt += 1
            continue;
        } else {
            out += speak.charAt(cnt).fontcolor("green");
            missing += speak.charAt(cnt).fontcolor("green");
            cnt += 1; 
        }
    }
    let all = [out, diff, missing];
    return all;
};

// function get_match(speak, label) {
//     result = "";
//     missing = "";
//     s = new difflib.SequenceMatcher(null, speak, label)
//     s.getMatchingBlocks()
// }


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
            document.getElementById("spinner-" + id).style.visibility = "visible";
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
                // document.getElementById("icon-" + id).style.visibility = "hidden";
                const audioBlob = new Blob(audioChunks, {'type': 'audio/wav;'});
                fetch(`${window.origin}/process_voice`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "audio/wav"
                        // "video_id": video_idx.toString()
                    },
                    body: audioBlob
                }).then((response) => response.json())
                .then((result) => {
                    console.log("Request complete! response:", result);
                    // difflib.ndiff(a, b);
                    var str = document.getElementById('label-' + id).innerHTML;
                    var label = str.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?]/g,"");
                    console.log(label)
                    var all = get_wrong(result.message, label);
                    console.log(all[1])
                    document.getElementById('final-result-' + id).innerHTML = all[0] + " (MISSING WORDS: " + all[2] + " )";
                    // document.getElementById('result-' + id).innerHTML = result.message;
                    document.getElementById("spinner-" + id).style.visibility = "hidden";
                });
            };
            recorder.start();
        });
    }
}


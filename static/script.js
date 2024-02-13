document.getElementById("urlForm").onsubmit = async function(event) {
    event.preventDefault();
    let formData = new FormData(event.target);
    let response = await fetch('/extract-urls', {
        method: 'POST',
        body: formData
    });
    let result = await response.json();
    document.getElementById("result").innerText = result.urls.join("\n");
};

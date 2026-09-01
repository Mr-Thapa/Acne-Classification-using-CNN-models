const imageInput = document.getElementById("imageInput");
const predictButton = document.getElementById("predictButton");

const fileName = document.getElementById("fileName");
const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("previewContainer");

const loading = document.getElementById("loading");
const result = document.getElementById("result");
const error = document.getElementById("error");

const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");

let selectedFile = null;


// Image selection
imageInput.addEventListener("change", () => {
    clearMessages();

    const file = imageInput.files[0];

    if (!file) {
        selectedFile = null;
        predictButton.disabled = true;
        fileName.textContent = "No image selected";
        previewContainer.classList.remove("visible");
        return;
    }

    if (!file.type.startsWith("image/")) {
        showError("Please select a valid image file.");
        imageInput.value = "";
        selectedFile = null;
        predictButton.disabled = true;
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;

    const reader = new FileReader();

    reader.onload = (event) => {
        imagePreview.src = event.target.result;
        previewContainer.classList.add("visible");
    };

    reader.readAsDataURL(file);

    predictButton.disabled = false;
});


// Prediction
predictButton.addEventListener("click", async () => {

    if (!selectedFile) {
        showError("Please select an image first.");
        return;
    }

    clearMessages();

    predictButton.disabled = true;
    loading.classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        displayPrediction(data);

    } catch (err) {
        showError(err.message || "Something went wrong.");

    } finally {
        loading.classList.add("hidden");
        predictButton.disabled = false;
    }
});


// Display prediction
function displayPrediction(data) {

    const prediction = data.class;
    const confidence = data.confidence;

    predictionText.textContent = `🎉 You have ${prediction} 🎉`;

    const percentage = confidence <= 1
        ? confidence * 100
        : confidence;

    confidenceText.textContent =
        `Confidence: ${percentage.toFixed(2)}%`;

    result.classList.remove("hidden");
}


// Helpers
function showError(message) {
    error.textContent = message;
    error.classList.remove("hidden");
}

function clearMessages() {
    result.classList.add("hidden");
    error.classList.add("hidden");
}

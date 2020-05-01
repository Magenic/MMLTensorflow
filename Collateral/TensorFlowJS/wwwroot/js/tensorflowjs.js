async function predictDeaths() {
    var cases = document.getElementById("noCases").value;

    document.getElementById("noDeaths").innerText = await GetPrediction(cases);
}

async function GetPrediction(cases) {
    const model = await tf.loadLayersModel("https://localhost:5001/tfjs/model.json");

    const inputTensor = tf.tensor([[(cases / 100000) ** 3]]);

    var result = model.predict(inputTensor);

    return result.arraySync()[0][0];
}
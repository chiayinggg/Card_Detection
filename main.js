console.log("主程式載入成功");

document.getElementById("imageUpload").addEventListener("change", handleImageUpload);
document.getElementById("camera").addEventListener("click", openCamera);
document.getElementById("reset").addEventListener("click", resetGame);
document.getElementById("snap").addEventListener("click", snap);

Payout = [];

document.getElementById("submit").addEventListener("click", ()=>{
  const eq_hit = parseFloat(document.getElementById("eq_hit").value);
  const eq_uplw = parseFloat(document.getElementById("eq_uplw").value);
  const con_hit = parseFloat(document.getElementById("con_hit").value);
  const con_uplw = parseFloat(document.getElementById("con_uplw").value);
  const gen_hit = parseFloat(document.getElementById("gen_hit").value);
  const gen_uplw = parseFloat(document.getElementById("gen_uplw").value);
  const gen_betw = parseFloat(document.getElementById("gen_betw").value);
  eq_hit.readOnly = true;
  eq_uplw.readOnly = true;
  con_hit.readOnly = true;
  con_uplw.readOnly = true;
  gen_hit.readOnly = true;
  gen_uplw.readOnly = true;
  gen_betw.readOnly = true;

  Payout = [ -eq_hit, eq_uplw, con_hit, -con_uplw, -gen_hit, -gen_uplw, gen_betw ];
  console.log(Payout);
});



let session = null;
const classNames = [
  "10C", "10D", "10H", "10S", "2C", "2D", "2H", "2S",
  "3C", "3D", "3H", "3S", "4C", "4D", "4H", "4S",
  "5C", "5D", "5H", "5S", "6C", "6D", "6H", "6S",
  "7C", "7D", "7H", "7S", "8C", "8D", "8H", "8S",
  "9C", "9D", "9H", "9S", "AC", "AD", "AH", "AS",
  "JC", "JD", "JH", "JS", "KC", "KD", "KH", "KS",
  "QC", "QD", "QH", "QS"
];

const colorMap = {
  "C": "green",  // ♣️
  "D": "blue",   // ♦️
  "H": "red",    // ♥️
  "S": "purple"  // ♠️
};

const detectedCards = new Set();  // 儲存已出現牌面
const game = new Set();

//載入模組
async function loadModel() {
  session = await ort.InferenceSession.create("best.onnx");
  console.log("✅ 模型已成功載入！");
}

window.onload = loadModel;


function handleImageUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const img = new Image();
  img.onload = async () => {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    const recordButton = document.getElementById("record");
    const probButton = document.getElementById("probability");

    recordButton.style.display = "inline-block";
    probButton.style.display = "inline-block";
    game.clear();

    document.getElementById("record").addEventListener("click", async () => {
      runPredictionOnly(canvas);
      recordButton.style.display = "none";
      probButton.style.display = "none";
    });
    document.getElementById("probability").addEventListener("click", async () => {
      runPredictionAndProb(canvas);
      recordButton.style.display = "none";
      probButton.style.display = "none";
    });
  };

  img.src = URL.createObjectURL(file);
}

async function snap() {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // ✅ 拍完照就關閉鏡頭
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }
  video.style.display = "none";
  document.getElementById("snap").style.display = "none";

  const recordButton = document.getElementById("record");
  const probButton = document.getElementById("probability");

  recordButton.style.display = "inline-block";
  probButton.style.display = "inline-block";
  game.clear();

  document.getElementById("record").addEventListener("click", async () => {
    runPredictionOnly(canvas);
    recordButton.style.display = "none";
    probButton.style.display = "none";
  });
  document.getElementById("probability").addEventListener("click", async () => {
    runPredictionAndProb(canvas);
    recordButton.style.display = "none";
    probButton.style.display = "none";
  });
}

function openCamera() {
  const video = document.getElementById("video");
  const snapButton = document.getElementById("snap");

  video.style.display = "block";
  snapButton.style.display = "inline-block";

  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      video.srcObject = stream;
    })
    .catch(err => {
      alert("無法開啟鏡頭：" + err.message);
    });
}


function resetGame() {
  game.clear();
  detectedCards.clear();

  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").textContent = "尚未辨識";
  document.getElementById("probResult").style.display = "none";

  // ✅ 關閉鏡頭
  const video = document.getElementById("video");
  if (video.srcObject) {
    const tracks = video.srcObject.getTracks();
    tracks.forEach(track => track.stop()); // 停止所有攝影機串流
    video.srcObject = null;
    video.style.display = "none";
  }
}

async function runPredictionOnly(canvas) {
  const resizedCanvas = document.createElement("canvas");
  resizedCanvas.width = 640;
  resizedCanvas.height = 640;
  const resizedCtx = resizedCanvas.getContext("2d");
  resizedCtx.drawImage(canvas, 0, 0, 640, 640);

  const imageData = resizedCtx.getImageData(0, 0, 640, 640).data;
  const input = new Float32Array(1 * 3 * 640 * 640);

  for (let i = 0; i < 640 * 640; i++) {
    input[i] = imageData[i * 4] / 255;             // R
    input[i + 640 * 640] = imageData[i * 4 + 1] / 255; // G
    input[i + 2 * 640 * 640] = imageData[i * 4 + 2] / 255; // B
  }

  const tensor = new ort.Tensor("float32", input, [1, 3, 640, 640]);
  const feeds = {};
  feeds[session.inputNames[0]] = tensor;

  const output = await session.run(feeds);
  const outputTensor = output[session.outputNames[0]].data; // Float32Array
  drawBoxes(outputTensor, canvas);
  document.getElementById("probResult").style.display = "none";
}

async function runPredictionAndProb(canvas) {
  const resizedCanvas = document.createElement("canvas");
  resizedCanvas.width = 640;
  resizedCanvas.height = 640;
  const resizedCtx = resizedCanvas.getContext("2d");
  resizedCtx.drawImage(canvas, 0, 0, 640, 640);

  const imageData = resizedCtx.getImageData(0, 0, 640, 640).data;
  const input = new Float32Array(1 * 3 * 640 * 640);

  for (let i = 0; i < 640 * 640; i++) {
    input[i] = imageData[i * 4] / 255;             // R
    input[i + 640 * 640] = imageData[i * 4 + 1] / 255; // G
    input[i + 2 * 640 * 640] = imageData[i * 4 + 2] / 255; // B
  }

  const tensor = new ort.Tensor("float32", input, [1, 3, 640, 640]);
  const feeds = {};
  feeds[session.inputNames[0]] = tensor;

  const output = await session.run(feeds);
  const outputTensor = output[session.outputNames[0]].data; // Float32Array
  drawBoxes(outputTensor, canvas);
  ProbabilityCalculaiton();

}


function drawBoxes(data, canvas) {
  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  const ctx = canvas.getContext("2d");
  const numBoxes = 8400;
  const numClasses = 52;
  const threshold = 0.3;

  ctx.lineWidth = 2;
  ctx.font = "16px Arial";

  for (let i = 0; i < numBoxes; i++) {
    // YOLOv8 原始格式: [1, 56, 8400]
    const x = data[0 * numBoxes + i];
    const y = data[1 * numBoxes + i];
    const w = data[2 * numBoxes + i];
    const h = data[3 * numBoxes + i];
    const objConf = sigmoid(data[4 * numBoxes + i]);

    // 找出 class 機率最大者
    let maxProb = -Infinity;
    let classId = -1;
    for (let j = 0; j < numClasses; j++) {
      const prob = sigmoid(data[(5 + j) * numBoxes + i]);
      if (prob > maxProb) {
        maxProb = prob;
        classId = j;
      }
    }

    const finalConf = objConf * maxProb;
    if (finalConf < threshold) continue;

    // YOLO 輸出是 center x/y，轉為 x1y1
    const x1 = (x - w / 2) * canvas.width / 640;
    const y1 = (y - h / 2) * canvas.height / 640;
    const boxWidth = w * canvas.width / 640;
    const boxHeight = h * canvas.height / 640;

    const label = classNames[classId+1];
    const suit = label.slice(-1);
    const color = colorMap[suit] || "black";

      // 畫框
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, boxWidth, boxHeight);

    // 畫標籤背景
    const text = `${label} ${finalConf.toFixed(2)}`;
    const textWidth = ctx.measureText(text).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1 - 20, textWidth + 8, 20);

    // 白字
    ctx.fillStyle = "white";
    ctx.fillText(text, x1 + 4, y1 - 5);

    // 加入已辨識的牌面
    detectedCards.add(label);
    game.add(label);
  }

  // 顯示牌面
  const resultDiv = document.getElementById("result");
  if (detectedCards.size === 0) {
    resultDiv.textContent = "未偵測到撲克牌。";
  } else {
    resultDiv.textContent = "已出現：" + Array.from(detectedCards).join(", ");
  }
  console.log(game);
}

const rankToValue = {
  "A": 1,
  "2": 2,
  "3": 3,
  "4": 4,
  "5": 5,
  "6": 6,
  "7": 7,
  "8": 8,
  "9": 9,
  "10": 10,
  "J": 11,
  "Q": 12,
  "K": 13
};

function getCardValue(label) {
  const rank = label.length == 3 ? label.slice(0, 2) : label[0];
  return rankToValue[rank];
}

function countRemainingByValue(value, cardset) {
  let count = 0;
  cardset.forEach(card => {
    if (getCardValue(card) == value) count++;
  })
  return 4-count;
}

function ProbabilityCalculaiton () {
  const gameset = Array.from(game);
  const detectedset = Array.from(detectedCards);
  if (gameset.length <2 ) return null;

  let A = getCardValue(gameset[0]);
  let B = getCardValue(gameset[1]);
  if (A>B) {
    [A,B] = [B,A];
    gameset[2] = gameset[0];
    gameset[0] = gameset[1];
    gameset[1] = gameset[2]; 
  }
  const totalLeft = 52-detectedset.length;

  let count_eqA = countRemainingByValue(A,detectedCards);
  let count_eqB = countRemainingByValue(B,detectedCards);
  let count_between = 0;
  let count_less = 0;
  let count_greater = 0;

  for (let i = 1 ; i<=13 ; i++) {
    if (i==A || i==B) continue;
    const remaining = countRemainingByValue(i,detectedCards);
    if (A==B) {
      if (i<A) count_less += remaining;
      else if (i>A) count_greater += remaining;
    }
    else if (A<B) {
      if (i<A) count_less += remaining;
      else if(i>A && i<B) count_between += remaining;
      else if (i>B) count_greater += remaining;
    }
  }

  let prob_less = (count_less/totalLeft).toFixed(2);
  let prob_greater = (count_greater/totalLeft).toFixed(2);
  let prob_between = (count_between/totalLeft).toFixed(2);
  let prob_eqA = (count_eqA/totalLeft).toFixed(2);
  let prob_eqB = (count_eqB/totalLeft).toFixed(2);
  
  console.log(prob_less,prob_between,prob_greater,prob_eqA,prob_eqB);
  console.log("小於牌數：", count_less);
  console.log("中間牌數：", count_between);
  console.log("大於牌數：", count_greater);
  console.log("撞柱A牌數：", count_eqA);
  console.log("撞柱B牌數：", count_eqB);

 

  //計算期望值
  let EV, EV_less, EV_greater = 0

  if (A==B) {
    EV_less = prob_less * Payout[1];
    EV_less -= prob_greater * Payout[1];
    EV_less += prob_eqA * Payout[0];

    EV_greater = prob_greater * Payout[1];
    EV_greater -= prob_less * Payout[1];
    EV_greater += prob_eqA * Payout[0];
  }
  else if (B-A==1) {
    EV = prob_less * Payout[3];
    EV += prob_greater * Payout[3];
    EV += prob_eqA * Payout[2];
    EV += prob_eqB * Payout[2];
  }
  else {
    EV = prob_less * Payout[5];
    EV += prob_greater * Payout[5];
    EV += prob_between * Payout[6];
    EV += prob_eqA * Payout[4];
    EV += prob_eqB * Payout[4];
  }
  


  //顯示計算結果
  document.getElementById("probResult").style.display = "inline-block";
  const probDiv = document.getElementById("probResult");
  probDiv.textContent = "門牌：" + gameset[0] + "," + gameset[1];
  probDiv.textContent += "\n小於" + gameset[0] + "的機率：" + prob_less;
  probDiv.textContent += "\t大於" + gameset[1] + "的機率：" + prob_greater;
  if (A==B){
    probDiv.textContent += "\t撞柱的機率：" + prob_eqA;
    probDiv.textContent += "\n押注在小於" + A + "的期望值：" + EV_less.toFixed(2);
    probDiv.textContent += "\t押注在大於" + A + "的期望值：" + EV_greater.toFixed(2);
  }
  else {
    probDiv.textContent += "\n在" + gameset[0] + "," + gameset[1] + "之間的機率：" + prob_between;
    probDiv.textContent += "\t撞柱" + gameset[0] + "的機率：" + prob_eqA;
    probDiv.textContent += "\t撞柱" + gameset[1] + "的機率：" + prob_eqB;
    probDiv.textContent += "\n期望值：" + EV.toFixed(2);
  }

}
let model;
const sentimentInput = document.getElementById("sentimentInput");
const checkBtn = document.getElementById("checkBtn");
const response = document.getElementById("response");
const mood = document.querySelector(".mood");

function buildDataset() {
	const sentences = [
		{
			sentence: "I am so happy today",
			sentiment: "postive"
		},
		{
			sentence: "I am so sad today",
			sentiment: "negative"
		},
		{
			sentence: "this day can't get any better.",
			sentiment: "positive"
		},
		{
			sentence: "how bad can this day get",
			sentiment: "negative"
		},
		{
			sentence: "i feel depressed",
			sentiment: "negative"
		},
		{
			sentence: "i feel amused",
			sentiment: "positive"
		},
		{
			sentence: "Oh my god, i love this person",
			sentiment: "positive"
		},
		{
			sentence: "this person is so weird",
			sentiment: "negative"
		},
		{
			sentence: "this move is so good",
			sentiment: "positive"
		},
		{
			sentence: "dude, the way he explains stuff is so nice",
			sentiment: "positive"
		},
		{
			sentence:
				"i went to the doctor last night and he said i dont have much time left",
			sentiment: "negative"
		},
		{
			sentence: "the doctor said i was gonna be fine",
			sentiment: "positive"
		},
		{
			sentence: "I have a crush on that girl",
			sentiment: "positive"
		},
		{
			sentence: "that person creeps me out",
			sentiment: "negative"
		},
		{
			sentence: "world is a family",
			sentiment: "positive"
		},
		{
			sentence: "i want this world to end soon",
			sentiment: "negative"
		},
		{
			sentence: "everyday i feel like giving up",
			sentiment: "negative"
		},
		{
			sentence: "i am feeling motivated everyday lately",
			sentiment: "positive"
		},
		{
			sentence: "group assignments sucks",
			sentiment: "negative"
		},
		{
			sentence: "i will kick you out of this class",
			sentiment: "negative"
		},
		{
			sentence: "your group presentation was the best",
			sentiment: "postive"
		},
		{
			sentence: "your group is better than the other group",
			sentiment: "postive"
		},
		{
			sentence: "without you my life is incomplete",
			sentiment: "positive"
		},
		{
			sentence: "i have no life without you",
			sentiment: "negative"
		},
		{
			sentence: "i had the best day in my life",
			sentiment: "positive"
		}
	];

	const dataset = sentences.map((sents) => {
		if (sents.sentiment == "positive") {
			sents["label"] = 1;
		} else {
			sents["label"] = 0;
		}
		delete sents.sentiment;
		return sents;
	});

	return dataset;
}

function buildModel() {
	const model = tf.sequential();

	// lstm layers
	model.add(
		tf.layers.lstm({
			units: 64,
			inputShape: [1, 512],
			returnSequences: false
		})
	);

	// dropout
	model.add(
		tf.layers.dropout({
			rate: 0.4
		})
	);

	// dense

	model.add(
		tf.layers.dense({
			units: 1,
			activation: "sigmoid"
		})
	);

	model.compile({
		loss: "binaryCrossentropy",
		optimizer: "adam",
		metrics: ["accuracy"]
	});

	return model;
}

async function train() {
	const dataset = buildDataset();
	const sentences = dataset.map((d) => d.sentence);
	const labels = dataset.map((d) => d.label);

	const labelsTensor = tf.tensor(labels, [labels.length, 1]);

	// load use

	const encoder = await use.load();

	// build model
	const model = buildModel();
	const embeddings = await encoder.embed(sentences);
	const inputTensor = embeddings.expandDims(1);

	// train model

	const training = await model.fit(inputTensor, labelsTensor, {
		epochs: 100,
		validationSplit: 0.1,
		callbacks: {
			onEpochEnd: (epoch, d) => console.log(d)
		}
	});

	return {
		model,
		encoder
	};
}

async function predict(model, sentence) {
	const encoder = await use.load();
	const embedding = await encoder.embed(sentence);
	const inputTensor = embedding.expandDims(1);
	const pred = await model.predict(inputTensor);
	const predData = await pred.data();
	return predData;
}

// predict("bro i wanna die soon");

window.addEventListener("load", async (e) => {
	response.style.display = "none";
	try {
		model = await tf.loadLayersModel("localstorage://sentiment-model");
		checkBtn.disabled = false;
	} catch (e) {
		const { model: trainedModel, encoder } = await train();
		await trainedModel.save("localstorage://sentiment-model");
		checkBtn.disabled = false;
		alert("training complete");
	}
});

checkBtn.addEventListener("click", async (e) => {
	response.style.display = "none";
	const text = sentimentInput.value;
	if (!text) {
		alert("please enter something");
		return;
	}

	checkBtn.disabled = true;
	const sentimentScore = await predict(model, text);

	if (sentimentScore > 0.5) {
		mood.innerText = "Happy..";
		mood.style.color = "blue";
	} else {
		mood.innerText = "Sad..";
		mood.style.color = "#FF4747";
	}

	response.style.display = "flex";
	checkBtn.disabled = false;
});
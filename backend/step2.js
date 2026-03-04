import { HfInference } from "@huggingface/inference";

// define our class
class PdfQA {
    // constructor method to initialize the class
    constructor({ model }) {
        this.model = model;
        this.inference = new HfInference(process.env.HF_API_KEY);
    }

    init() {
        this.initChatModel();
        return this;
    }

    initChatModel() {
        console.log("Initializing chat model...");
        this.chatModel = (question, context) =>
            this.inference.questionAnswering({
                model: this.model,
                inputs: { question, context },
            });
    }
}
// create an instance of the class and store the oblect returned in a variable
const pdfQA = new PdfQA({ model: "deepset/roberta-base-squad2" }).init();

// log our object
console.log({ pdfQA });
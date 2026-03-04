import dotenv from "dotenv";
import { HfInference } from "@huggingface/inference";

dotenv.config();
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

    async initChatModel() {
        console.log("Initializing chat model...");
        this.chatModel = (question, context) =>
            this.inference.questionAnswering({
                model: this.model,
                inputs: { question, context },
            });
        const response = await this.chatModel("What is the capital of France?", "France is a country in Europe. The capital of France is Paris.");
        console.log("Response from chat model:", response);    
    }
}
// create an instance of the class and store the oblect returned in a variable
const pdfQA = new PdfQA({ model: "deepset/roberta-base-squad2" }).init();

// log our object
//console.log({ pdfQA });
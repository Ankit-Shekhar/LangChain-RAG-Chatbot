import dotenv from "dotenv";
import path from "node:path";
import { HfInference } from "@huggingface/inference";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

dotenv.config();

class PdfQA {
    constructor({ model, pdfDocumentPath }) {
        this.model = model;
        this.pdfDocumentPath = pdfDocumentPath;
        this.inference = new HfInference(process.env.HF_API_KEY);
    }

    async init() {
        await this.initChatModel();
        await this.loadPdfDocument();
        return this;
    }

    async initChatModel() {
        console.log("Initializing chat model...");
        this.chatModel = (question, context) =>
            this.inference.questionAnswering({
                model: this.model,
                inputs: { question, context },
            });
        const response = await this.chatModel(
            "What is the capital of France?",
            "France is a country in Europe. The capital of France is Paris."
        );
        console.log("Response from chat model:", response);
    }

    async loadPdfDocument() {
        // Resolve relative to project root even when running from backend/
        const pdfFullPath = path.resolve(process.cwd(), this.pdfDocumentPath);
        console.log(`Loading PDF document from ${pdfFullPath}...`);
        const loader = new PDFLoader(pdfFullPath, { splitPages: true });
        this.pdfDocument = await loader.load();
        console.log(`PDF loaded: ${this.pdfDocument.length} pages`);
    }
}

async function main() {
    const pdfQA = new PdfQA({
        model: "deepset/roberta-base-squad2",
        pdfDocumentPath: path.join("..", "materials", "Ankit Shekhar__14942723010.pdf"),
    });

    await pdfQA.init();

    if (pdfQA.pdfDocument?.length) {
        console.log("\n\nDocument #0 page content:", pdfQA.pdfDocument[0].pageContent);
        console.log("\n\nDocument #0 metadata:", pdfQA.pdfDocument[0].metadata);
    }
}

main().catch((err) => {
    console.error("Failed to initialize PdfQA", err);
});
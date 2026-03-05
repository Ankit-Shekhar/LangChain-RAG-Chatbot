import dotenv from "dotenv";
import path from "node:path";
import { HfInference } from "@huggingface/inference";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CharacterTextSplitter } from "@langchain/textsplitters";

dotenv.config();

class PdfQA {
    constructor({ model, pdfDocumentPath, chunkSize, chunkOverlap }) {
        this.model = model;
        this.pdfDocumentPath = pdfDocumentPath;
        this.chunkSize = chunkSize;
        this.chunkOverlap = chunkOverlap;
        this.inference = new HfInference(process.env.HF_API_KEY);
    }

    async init() {
        await this.initChatModel();
        await this.loadPdfDocument();
        await this.splitDocumentIntoChunks();
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

    async splitDocumentIntoChunks() {
        console.log(`Splitting document into chunks (chunkSize=${this.chunkSize}, chunkOverlap=${this.chunkOverlap})...`);
        const splitter = new CharacterTextSplitter({
            separator: " ",
            chunkSize: this.chunkSize,
            chunkOverlap: this.chunkOverlap
        });
        this.chunks = await splitter.splitDocuments(this.pdfDocument);
        console.log(`Document split into chunks: ${this.chunks.length}`);
    }
}

async function main() {
    const pdfQA = new PdfQA({
        model: "deepset/roberta-base-squad2",
        pdfDocumentPath: path.join("..", "materials", "Ankit Shekhar__14942723010.pdf"),
        chunkSize: 1000,
        chunkOverlap: 0
    });

    await pdfQA.init();

    if (pdfQA.pdfDocument?.length) {
        console.log("\n\nDocument #0 page content:", pdfQA.pdfDocument[1].pageContent);
        console.log("\n\nDocument #0 metadata:", pdfQA.pdfDocument[1].metadata);
    }

    if (pdfQA.chunks?.length) {
        console.log("\n\nChunk #0 content:", pdfQA.chunks[1].pageContent);
        console.log("\n\nChunk #0 metadata:", pdfQA.chunks[1].metadata);
    }

    if (pdfQA.pdfDocument?.length){
        console.log(pdfQA.chunks);
    }
}

main().catch((err) => {
    console.error("Failed to initialize PdfQA", err);
});
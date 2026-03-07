import dotenv from "dotenv";
import path from "node:path";
import { HfInference } from "@huggingface/inference";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";


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
        this.embeddings = new HuggingFaceInferenceEmbeddings({
            model: "sentence-transformers/all-MiniLM-L6-v2",
            apiKey: process.env.HF_API_KEY,
        });
        await this.createVectorStore();

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

    async createVectorStore() {
        console.log("Creating document embeddings...");
        // this.vectorStore = new MemoryVectorStore(this.embeddings);
        // await this.vectorStore.addDocuments(this.chunks);

        this.db = await MemoryVectorStore.fromDocuments(this.chunks, this.embeddings);
        console.log("Vector store created.");
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

    console.log("Embeddings model: ", pdfQA.db.embeddings.model);
    console.log("# Number of embeddings: ", pdfQA.db.memoryVectors.length);


    // if (pdfQA.pdfDocument?.length) {
    //     console.log("\n\nDocument #0 page content:", pdfQA.pdfDocument[1].pageContent);
    //     console.log("\n\nDocument #0 metadata:", pdfQA.pdfDocument[1].metadata);
    // }

    // if (pdfQA.chunks?.length) {
    //     console.log("\n\nChunk #0 content:", pdfQA.chunks[1].pageContent);
    //     console.log("\n\nChunk #0 metadata:", pdfQA.chunks[1].metadata);
    // }

    // if (pdfQA.pdfDocument?.length) {
    //     console.log(pdfQA.chunks);
    // }

    // Query the Vector store directly: https://js.langchain.com/v0.2/docs/integrations/vectorstores/memory/#query-directly
    // Note: the similarity search is based on cosine similarity, so the results are not deterministic and may vary across runs.
    // The function similaritySearch accepts a query and the number of results to return. The results are returned in order of relevance, with the most relevant result first from the provided pdf.
    const similaritySearchResults = await pdfQA.db.similaritySearch("The chmod Command in Unix", 2);
    // keep an eye on it on the above line.
    console.log("\nDocument pages related to our query:");
    for (const doc of similaritySearchResults) {
        console.log(`\n* ${JSON.stringify(doc.metadata.loc, null)}\n`);
        //   console.log(doc.pageContent);
    }

    // If you want to execute a similarity search and receive the corresponding scores you can run:
    // const similaritySearchWithScoreResults = await pdfQA.db.similaritySearchWithScore("The chmod Command in Unix", 10);

    // console.log("Document pages and their score related to our query:");
    // for (const [doc, score] of similaritySearchWithScoreResults) {
    //     console.log(`* [SIM=${score.toFixed(3)}] [Page number: ${doc.metadata.loc.pageNumber}]`);
    // }

}

main().catch((err) => {
    console.error("Failed to initialize PdfQA", err);
});

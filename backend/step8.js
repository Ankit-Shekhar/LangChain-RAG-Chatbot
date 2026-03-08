import dotenv from "dotenv";
import path from "node:path";
import { HfInference } from "@huggingface/inference";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { HuggingFaceInference as HFLLM } from "@langchain/community/llms/hf";


dotenv.config();

class PdfQA {
    constructor({ model, pdfDocumentPath, chunkSize, chunkOverlap, searchType = "similarity", kDocuments }) {
        this.model = model;
        this.pdfDocumentPath = pdfDocumentPath;
        this.chunkSize = chunkSize;
        this.chunkOverlap = chunkOverlap;
        this.inference = new HfInference(process.env.HF_API_KEY);
        this.history = [];
        // This configures the type of vector search. By default, Vector stores will use a 'similarity' search. Alternatively, you can pass in the "mmr" value to searchType and perform a more advanced and precise search based on the 'Maximal Marginal Relevance' search algorithm.

        this.searchType = searchType;
        // The number of relevant document to return based on the search query:
        this.kDocuments = kDocuments; // by default, we will return the top 4 most relevant documents. You can adjust this value based on your needs and the size of your documents.

    }

    async init() {
        await this.initChatModel();
        this.initLlm();
        await this.loadPdfDocument();
        await this.splitDocumentIntoChunks();
        this.embeddings = new HuggingFaceInferenceEmbeddings({
            model: "sentence-transformers/all-MiniLM-L6-v2",
            apiKey: process.env.HF_API_KEY,
        });
        await this.createVectorStore();
        this.createRetriever();


        // This is what LangChain is all about. Create a Retrieval chain that will put
        // all the pieces together and give us an interface that will allow us to query
        // our vector store through using natural language:
        this.chain = await this.createChain()
        return this;
    }

    async initChatModel() {
        console.log("Initializing chat model...");
        this.chatModel = (question, context) =>
            this.inference.questionAnswering({
                model: this.model,
                inputs: { question, context },
            });
    }

    initLlm() {
        // Use HF Inference text generation to stay deployable
        this.llm = new HFLLM({
            apiKey: process.env.HF_API_KEY,
            model: "mistralai/Mistral-7B-Instruct-v0.2",
            maxTokens: 256,
            temperature: 0.2,
        });
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

        this.db = await MemoryVectorStore.fromDocuments(this.chunks, this.embeddings);
        console.log("Vector store created.");
    }

    // Method to perform a similarity search in the memory vector store. It calculates the similarity between the query vector and each vector in the store, sorts the results by similarity, and returns the top k results along with their scores.
    createRetriever() {
        console.log("Initialize vector store retriever...");
        this.retriever = this.db.asRetriever({
            k: this.kDocuments,
            searchType: this.searchType
        });
    }

    async createChain() {
        console.log("Creating lightweight retrieval QA executor (HF Inference)...");
        return async ({ query, k = this.kDocuments }) => {
            const docs = await this.retriever.invoke(query);
            const context = docs.map((d) => d.pageContent).join("\n\n");
            const answer = await this.chatModel(query, context);
            const text = answer.answer ?? answer;
            // keep a simple chat history log
            this.history.push({ query, answer: text, usedDocs: docs.map((d) => d.metadata) });
            return { text, docs, history: this.history };
        };
    }

    // Helper method to return the chain:
    queryChain() {
        return this.chain;
    }
}

async function main() {
    const pdfQA = new PdfQA({
        model: "deepset/roberta-base-squad2",
        pdfDocumentPath: path.join("..", "materials", "Ankit_Shekhar_Final_Resume.pdf"),
        chunkSize: 1000,
        chunkOverlap: 0,
        searchType: "similarity",
        kDocuments: 3
    });

    await pdfQA.init();

    const pdfQaChain = pdfQA.queryChain();

    const answer1 = await pdfQaChain({ query: "What are the projects on which Ankit has worked on?" });
    // console.log( "🤖", answer1.text, "\n" );
    // console.log( "# of documents used as context: ", answer1.docs?.length ?? 0, "\n" );
    // console.log( "Chat history:", JSON.stringify(answer1.history, null, 2) );
    console.log(answer1);
}

main().catch((err) => {
    console.error("Failed to initialize PdfQA", err);
});

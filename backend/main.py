import os
import json
import asyncio
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from database import engine, SessionLocal, Base
from db_models import User, Workspace, Message
from ai_model import MiniCompanionAI
from auth import get_current_user, hash_password, verify_password, create_access_token, validate_phone, compute_user_cell
from tokenizer import tokenize, normalize_lang, grid_dims, supported_languages
from memory_grid import MemoryGrid
from grid_crawler import crawl as grid_crawl
from web_crawler import WebCrawler
from data_mixer import DataMixer
from prompts_manager import build_prompt
from intent_analyzer import detect_domain
from external import fetch_dictionary, fetch_news, fetch_books, fetch_elibrary, fetch_wikipedia
from symbols import recognize_symbols
from code_languages import CODE_TERMS
from search_cache import SearchCache
from memory_cache import MemoryCache
from background_training import start_background_training
from rules import enforce_rules
from BubbleJumbo_rules import BubbleJumboRules
from word_understanding import WordUnderstanding

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="CoMpaNeoN AI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global components
model: Optional[MiniCompanionAI] = None
tokenizer_vocab: Optional[Dict[str, int]] = None
reverse_vocab: Optional[Dict[int, str]] = None

memory = MemoryGrid()
web_crawler = WebCrawler()
mixer = DataMixer()
search_cache = SearchCache(ttl_seconds=300)
memory_cache = MemoryCache()
bubble_rules = BubbleJumboRules()
word_understanding = WordUnderstanding(memory)

# Load model if exists
def load_model_if_exists():
    global model, tokenizer_vocab, reverse_vocab
    if os.path.exists('tokenizer_vocab.json'):
        with open('tokenizer_vocab.json', 'r') as f:
            data = json.load(f)
            tokenizer_vocab = data['vocab']
            reverse_vocab = {int(k): v for k, v in data['reverse'].items()}
    else:
        tokenizer_vocab = {"<pad>":0, "<unk>":1, "<start>":2, "<end>":3}
        reverse_vocab = {0:"<pad>",1:"<unk>",2:"<start>",3:"<end>"}
    if os.path.exists('companion_model.pth'):
        model = MiniCompanionAI(len(tokenizer_vocab))
        model.load_state_dict(torch.load('companion_model.pth', map_location=device))
        model.to(device)
        model.eval()
    else:
        model = None

load_model_if_exists()

# Helpers
def encode_text(text: str, lang: str = "en") -> List[int]:
    tokens = tokenize(text, lang)
    ids = [tokenizer_vocab.get("<start>", 2)]
    for t in tokens:
        word = t['stem'] if t['stem'] in tokenizer_vocab else (t['original'] if t['original'] in tokenizer_vocab else "<unk>")
        ids.append(tokenizer_vocab.get(word, 1))
    ids.append(tokenizer_vocab.get("<end>", 3))
    return ids

def decode_ids(ids: List[int]) -> str:
    return ' '.join([reverse_vocab.get(i, '<unk>') for i in ids if i not in {0,2,3}])

def get_context_from_memory(query: str) -> str:
    cached = memory_cache.get(query)
    if cached:
        return cached
    context = word_understanding.get_context(query)
    memory_cache.set(query, context)
    return context

# Pydantic models
class SignupRequest(BaseModel):
    full_name: str
    phone: str
    password: str
    language: str = "en"
    country: str = "Nigeria"
    temperament: str = "sanguine"

class LoginRequest(BaseModel):
    phone: str
    password: str

class WorkspaceCreate(BaseModel):
    first_message: str
    temperament: Optional[str] = "sanguine"

class MessageRequest(BaseModel):
    content: str

class GenerateRequest(BaseModel):
    prompt: str
    max_len: int = 500
    temperature: float = 0.8

class ResearchRequest(BaseModel):
    query: str

class CrawlRequest(BaseModel):
    url: str

class TrainRequest(BaseModel):
    epochs: int = 10
    batch_size: int = 8

class PredictRequest(BaseModel):
    text: str
    top_k: int = 5

# Auth endpoints
@app.post("/auth/signup")
async def signup(req: SignupRequest):
    db = SessionLocal()
    try:
        if not validate_phone(req.phone, req.country[:2].upper()):
            raise HTTPException(400, "Invalid phone number. Must start with '+' and follow country length.")
        user = db.query(User).filter(User.phone == req.phone).first()
        if user:
            raise HTTPException(400, "Phone already registered")
        hashed = hash_password(req.password)
        start_row, start_col = compute_user_cell(req.full_name, req.phone)
        new_user = User(
            full_name=req.full_name,
            phone=req.phone,
            password_hash=hashed,
            language=req.language,
            country=req.country,
            temperament=req.temperament,
            start_row=start_row,
            start_col=start_col
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        token = create_access_token(str(new_user.id))
        return {"access_token": token, "user": {"id": str(new_user.id), "full_name": new_user.full_name, "start_row": start_row}}
    finally:
        db.close()

@app.post("/auth/login")
async def login(req: LoginRequest):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.phone == req.phone).first()
        if not user or not verify_password(req.password, user.password_hash):
            raise HTTPException(401, "Invalid credentials")
        token = create_access_token(str(user.id))
        return {"access_token": token, "user": {"id": str(user.id), "full_name": user.full_name, "start_row": user.start_row}}
    finally:
        db.close()

# Workspace endpoints
@app.post("/workspace")
async def create_workspace(req: WorkspaceCreate, user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        project_name = req.first_message.split()[0].capitalize()
        ws = Workspace(user_id=user.id, project_name=project_name)
        db.add(ws)
        db.commit()
        db.refresh(ws)
        msg = Message(workspace_id=ws.id, user_id=user.id, role="user", content=req.first_message)
        db.add(msg)
        db.commit()
        return {"workspace_id": str(ws.id), "project_name": project_name}
    finally:
        db.close()

@app.get("/workspaces")
async def list_workspaces(user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        workspaces = db.query(Workspace).filter(Workspace.user_id == user.id).all()
        return [{"id": str(w.id), "project_name": w.project_name, "summary": w.summary} for w in workspaces]
    finally:
        db.close()

@app.post("/workspace/{ws_id}/message")
async def add_message(ws_id: str, req: MessageRequest, user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        msg = Message(workspace_id=ws_id, user_id=user.id, role="user", content=req.content)
        db.add(msg)
        db.commit()
        return {"ok": True}
    finally:
        db.close()

@app.post("/workspace/{ws_id}/generate")
async def generate_in_workspace(ws_id: str, req: GenerateRequest, user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        ws = db.query(Workspace).filter(Workspace.id == ws_id, Workspace.user_id == user.id).first()
        if not ws:
            raise HTTPException(404, "Workspace not found")
        msgs = db.query(Message).filter(Message.workspace_id == ws_id).order_by(Message.created_at.desc()).limit(10).all()
        history = "\n".join([f"{m.role}: {m.content}" for m in reversed(msgs)])
        context = get_context_from_memory(req.prompt)
        prompt = build_prompt(
            query=req.prompt,
            context=context,
            workspace_name=ws.project_name,
            conversation_history=history,
            temperament=user.temperament
        )
        if model is not None:
            input_ids = torch.tensor([encode_text(req.prompt)], dtype=torch.long).to(device)
            output_ids = []
            with torch.no_grad():
                for _ in range(req.max_len):
                    logits = model(input_ids)[0, -1, :] / req.temperature
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                    output_ids.append(next_id)
                    input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
                    if next_id == tokenizer_vocab.get("<end>", 3):
                        break
            generated = decode_ids(output_ids)
        else:
            generated = prompt
        if not enforce_rules(generated, user.temperament):
            generated = "I apologize, I cannot provide that answer."
        msg = Message(workspace_id=ws_id, user_id=user.id, role="assistant", content=generated)
        db.add(msg)
        db.commit()
        return {"generated": generated}
    finally:
        db.close()

# Summary endpoint
@app.post("/workspace/{ws_id}/summary")
async def generate_summary(ws_id: str, user: User = Depends(get_current_user)):
    db = SessionLocal()
    try:
        ws = db.query(Workspace).filter(Workspace.id == ws_id, Workspace.user_id == user.id).first()
        if not ws:
            raise HTTPException(404, "Workspace not found")
        msgs = db.query(Message).filter(Message.workspace_id == ws_id).all()
        if not msgs:
            raise HTTPException(400, "No messages to summarize")
        combined = " ".join([m.content for m in msgs])
        summary = combined[:500] + ("..." if len(combined) > 500 else "")
        ws.summary = summary
        db.commit()
        return {"summary": summary}
    finally:
        db.close()

# General generation
@app.post("/generate")
async def generate(req: GenerateRequest):
    context = get_context_from_memory(req.prompt)
    prompt = build_prompt(query=req.prompt, context=context, temperament="sanguine")
    if model is not None:
        input_ids = torch.tensor([encode_text(req.prompt)], dtype=torch.long).to(device)
        output_ids = []
        with torch.no_grad():
            for _ in range(req.max_len):
                logits = model(input_ids)[0, -1, :] / req.temperature
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                output_ids.append(next_id)
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
                if next_id == tokenizer_vocab.get("<end>", 3):
                    break
        generated = decode_ids(output_ids)
    else:
        generated = prompt
    if not enforce_rules(generated, "sanguine"):
        generated = "I apologize, I cannot provide that answer."
    return {"generated": generated}

# Prediction
@app.post("/predict")
async def predict(req: PredictRequest):
    if model is None or tokenizer_vocab is None:
        raise HTTPException(400, "Model not trained")
    input_ids = torch.tensor([encode_text(req.text)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(input_ids)[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk = torch.topk(probs, req.top_k)
        predictions = []
        for i in range(req.top_k):
            word = reverse_vocab.get(topk.indices[i].item(), "<unk>")
            predictions.append({"word": word, "prob": topk.values[i].item()})
    return {"predictions": predictions}

# Research
@app.post("/research")
async def research(req: ResearchRequest):
    query = req.query.strip()
    cached = search_cache.get(query)
    if cached:
        return cached
    results = await asyncio.gather(
        fetch_dictionary(query.split()[0] if query.split() else query),
        fetch_news(query),
        fetch_books(query),
        fetch_elibrary(query),
        fetch_wikipedia(query)
    )
    data = {
        "dictionary": results[0],
        "news": results[1],
        "books": results[2],
        "elibrary": results[3],
        "wikipedia": results[4]
    }
    search_cache.set(query, data)
    return data

# Crawl web
@app.post("/crawl")
async def crawl_web(req: CrawlRequest):
    text = web_crawler.crawl(req.url)
    doc_id = memory.add_document(text, "en", source=req.url)
    with open('data/web_words.txt', 'a', encoding='utf-8') as f:
        f.write(text + '\n')
    return {"message": "Crawled", "doc_id": doc_id, "words": len(text.split())}

# Train
@app.post("/train")
async def train(req: TrainRequest):
    from train import train as do_train
    await asyncio.to_thread(do_train)
    return {"message": "Training completed"}

# Startup background training
@app.on_event("startup")
async def startup_event():
    start_background_training()

# Serve frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
# 🤖 المستشار — Gouvathon 2026 API

API RAG Chatbot pour les procédures du Ministère du Commerce et du Tourisme de Mauritanie.

## ⚡ Démarrage rapide

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Configurer la clé API Gemini

```bash
export GEMINI_API_KEY="your_key_here"
```

### 3. Lancer le serveur

```bash
uvicorn main:app --reload --port 8000
```

### 4. Ouvrir la documentation interactive

👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 Endpoints

| Méthode  | Route                    | Description                          |
| -------- | ------------------------ | ------------------------------------ |
| `GET`    | `/`                      | Health check                         |
| `GET`    | `/status`                | État de l'index (chunks, fichiers)   |
| `POST`   | `/upload`                | Uploader un PDF et l'indexer         |
| `POST`   | `/ask`                   | Poser une question (RAG)             |
| `POST`   | `/search`                | Recherche dans les chunks            |
| `DELETE` | `/sessions/{session_id}` | Effacer l'historique d'une session   |
| `DELETE` | `/index`                 | Réinitialiser tout l'index           |

---

## 🧪 Exemples d'utilisation (curl)

### Uploader un PDF

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@Rapport_Ministere.pdf"
```

### Poser une question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ما هي وثائق اعتماد مؤسسة إيواء؟",
    "session_id": "user-1"
  }'
```

### Rechercher dans les documents

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "licence importation", "top_k": 3}'
```

---

## 🐳 Docker

```bash
docker build -t gouvathon-api .
docker run -p 8000:8000 -e GEMINI_API_KEY="your_key" gouvathon-api
```

---

## 🏗️ Architecture

```
Mobile App (Flutter/React Native)
        │
        ▼
   FastAPI Server (/ask, /upload, /search)
        │
   ┌────┴────┐
   │  FAISS  │ ← Vector search
   └────┬────┘
        │
   ┌────┴────┐
   │ Gemini  │ ← Embeddings + Generation
   └─────────┘
```

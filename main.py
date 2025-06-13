from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from databases import Database
from sqlalchemy import MetaData, Table, Column, String
from sqlalchemy.dialects.postgresql import UUID
import uuid, json, httpx, os, base64
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Banco de dados
DATABASE_URL = os.getenv("DATABASE_URL")
database = Database(DATABASE_URL)
metadata = MetaData()

orcamentos = Table(
    "orcamentos",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("nome", String, nullable=False),
    Column("email", String, nullable=False),
    Column("telefone", String, nullable=False),
    Column("servico", String, nullable=False),
)

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

class MensagemChat(BaseModel):
    mensagem: str
    historico: Optional[List[dict]] = None

async def chamar_ia(messages) -> str | dict:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "google/gemini-2.5-flash-preview-05-20",
        "messages": messages
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except:
                return content
    except Exception as e:
        print("Erro na IA:", e)
        return "Desculpe, ocorreu um erro interno. Tente novamente em instantes."

@app.post("/chat")
async def chat(dados: MensagemChat):
    historico = dados.historico or []

    prompt_sistema = """
### Papel e Objetivo
Você é um assistente de vendas da "InovaTech Solutions", especialista em agentes de IA personalizados. Seu objetivo é engajar os visitantes do site, tirar dúvidas sobre nossos serviços e, caso demonstrem interesse, coletar os dados necessários para um orçamento.

### Regras de Conversa
1. Persona: Seja sempre educado, prestativo e inteligente. Evite jargões técnicos.
2. Tom de Voz: Responda de forma clara, objetiva e amigável.
3. Personalização: Se o cliente disser o nome dele, use-o para se dirigir a ele durante o diálogo.
4. Proatividade: Forneça exemplos práticos de como os agentes de IA podem ser aplicados em diferentes setores (vendas, atendimento, etc.).
5. Variação: Evite terminar todas as respostas da mesma forma. Varie com frases como "Posso ajudar com mais alguma dúvida?" ou "Se quiser, posso detalhar mais sobre isso."
6. Honestidade: Se não souber a resposta para uma pergunta específica, diga: "Essa é uma ótima pergunta. Vou encaminhar sua dúvida para um especialista."

### Processo de Orçamento
1. Se o visitante pedir um orçamento ou perguntar preços, inicie a coleta de dados.
2. Peça UMA informação de cada vez:
   - Nome completo
   - E-mail
   - Telefone
   - Serviço desejado
3. Ao receber a última informação, retorne APENAS um objeto JSON puro:
{ "nome": "string", "email": "string", "telefone": "string", "servico": "string" }
"""

    messages = [{"role": "system", "content": prompt_sistema}]
    for msg in historico:
        if msg.get("role") in ["user", "assistant"]:
            messages.append(msg)
    messages.append({"role": "user", "content": dados.mensagem})

    resposta = await chamar_ia(messages)

    if isinstance(resposta, dict) and all(k in resposta for k in ["nome", "email", "telefone", "servico"]):
        orcamento = {
            "id": uuid.uuid4(),
            "nome": resposta["nome"],
            "email": resposta["email"],
            "telefone": resposta["telefone"],
            "servico": resposta["servico"]
        }
        query = orcamentos.insert().values(**orcamento)
        await database.execute(query)
        return {"reply": "Orçamento recebido com sucesso! Nossa equipe entrará em contato em breve."}
    
    if isinstance(resposta, str):
        return {"reply": resposta}
    
    return {"reply": "Não entendi. Pode repetir, por favor?"}

@app.post("/whatsapp")
async def receber_mensagem_zapi(request: Request):
    payload = await request.json()
    numero = payload.get("phone")
    tipo = payload.get("type")
    texto = payload.get("text", {}).get("message")
    audio = payload.get("audio", {}).get("url")
    imagem = payload.get("image", {}).get("url")

    if not numero:
        raise HTTPException(status_code=400, detail="Número ausente")

    conteudo = ""
    if texto:
        conteudo = texto
    elif audio:
        conteudo = await baixar_e_converter_base64(audio, "audio/ogg")
        conteudo = f"[AUDIO]\n{conteudo}"
    elif imagem:
        conteudo = await baixar_e_converter_base64(imagem, "image/jpeg")
        conteudo = f"[IMAGEM]\n{conteudo}"
    else:
        return {"reply": "Nenhuma mensagem válida encontrada."}

    try:
        async with httpx.AsyncClient() as client:
            resposta = await client.post("http://localhost:8000/chat", json={"mensagem": conteudo, "historico": []})
            dados = await resposta.json()
            mensagem = dados.get("reply", "Erro ao gerar resposta")

            await client.post(
                f"https://api.z-api.io/instances/{os.getenv('INSTANCE_ID')}/token/{os.getenv('TOKEN')}/send-text",
                json={"phone": numero, "message": mensagem}
            )
    except Exception as e:
        print("Erro no envio:", e)

    return {"status": "ok"}

async def baixar_e_converter_base64(url: str, mime_type: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resposta = await client.get(url)
            resposta.raise_for_status()
            base64_str = base64.b64encode(resposta.content).decode("utf-8")
            return f"data:{mime_type};base64,{base64_str}"
    except Exception as e:
        print("Erro ao baixar/converter mídia:", e)
        return "[Erro ao processar mídia]"

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from databases import Database
from sqlalchemy import create_engine, MetaData, Table, Column, String
from sqlalchemy.dialects.postgresql import UUID
import uuid, json, httpx, os, base64
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Inicialização do FastAPI
app = FastAPI()

# Configuração do CORS para permitir todas as origens
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuração do Banco de Dados ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL não foi definida nas variáveis de ambiente.")

database = Database(DATABASE_URL)
metadata = MetaData()

# Definição da tabela de orçamentos
orcamentos = Table(
    "orcamentos",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("nome", String, nullable=False),
    Column("email", String, nullable=False),
    Column("telefone", String, nullable=False),
    Column("servico", String, nullable=False),
)

# --- Ciclo de Vida da Aplicação ---
@app.on_event("startup")
async def startup():
    """Conecta ao banco de dados na inicialização."""
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    """Desconecta do banco de dados no encerramento."""
    await database.disconnect()

# --- Modelos de Dados (Pydantic) ---
class MensagemChat(BaseModel):
    mensagem: str
    historico: Optional[List[dict]] = None

# --- Funções Auxiliares ---

async def chamar_ia(messages: List[dict]) -> str | dict:
    """
    Função assíncrona para chamar a API do OpenRouter e obter uma resposta da IA.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "google/gemini-flash", 
        "messages": messages
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
    except httpx.HTTPStatusError as e:
        print(f"Erro de status na API da IA: {e.response.status_code} - {e.response.text}")
        return "Desculpe, não consegui obter uma resposta da IA no momento."
    except Exception as e:
        print(f"Erro na IA: {e}")
        return "Desculpe, ocorreu um erro interno. Tente novamente em instantes."

async def baixar_e_converter_base64(url: str, mime_type: str) -> str:
    """
    Baixa um arquivo de mídia (áudio/imagem) de uma URL e o converte para uma string base64.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resposta = await client.get(url)
            resposta.raise_for_status()
            base64_str = base64.b64encode(resposta.content).decode("utf-8")
            return f"data:{mime_type};base64,{base64_str}"
    except Exception as e:
        print(f"Erro ao baixar/converter mídia: {e}")
        return "[Erro ao processar mídia]"

# --- Endpoints da API ---

@app.get("/")
async def root():
    """Endpoint raiz para verificar se a API está no ar."""
    return {"message": "API da InovaTech Solutions no ar!"}

@app.post("/chat")
async def chat(dados: MensagemChat):
    """
    Endpoint principal para interagir com o chatbot.
    """
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
        if isinstance(msg, dict) and msg.get("role") in ["user", "assistant"]:
            messages.append(msg)
    messages.append({"role": "user", "content": dados.mensagem})

    resposta = await chamar_ia(messages)

    if isinstance(resposta, dict) and all(k in resposta for k in ["nome", "email", "telefone", "servico"]):
        try:
            query = orcamentos.insert().values(**resposta)
            await database.execute(query)
            return {"reply": "Orçamento recebido com sucesso! Nossa equipe entrará em contato em breve."}
        except Exception as e:
            print(f"Erro ao salvar orçamento no banco: {e}")
            return {"reply": "Recebi seus dados, mas tive um problema para registrar. Nossa equipe já foi notificada."}
    
    if isinstance(resposta, str):
        return {"reply": resposta}
    
    return {"reply": "Não entendi sua resposta. Poderia reformular?"}

@app.post("/whatsapp")
async def receber_mensagem_zapi(request: Request):
    """
    Endpoint de webhook para receber mensagens da Z-API.
    """
    try:
        payload = await request.json()
        print(f"--- Payload Recebido no Webhook --- \n{json.dumps(payload, indent=2)}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Payload JSON inválido.")

    numero = payload.get("phone")
    texto = payload.get("text", {}).get("message") if isinstance(payload.get("text"), dict) else None
    
    if not numero:
        raise HTTPException(status_code=400, detail="O campo 'phone' é obrigatório.")

    if not texto:
        return {"status": "ok", "message": "Ignorando mensagem que não é de texto."}

    try:
        async with httpx.AsyncClient() as client:
            public_url = os.getenv("PUBLIC_URL")
            if not public_url:
                raise RuntimeError("PUBLIC_URL não configurada.")

            resposta_chat = await client.post(
                 f"{public_url.rstrip('/')}/chat",
                 json={"mensagem": texto, "historico": []},
                 timeout=60.0
            )
            resposta_chat.raise_for_status()
            
            dados = resposta_chat.json()
            mensagem_resposta = dados.get("reply", "Não consegui gerar uma resposta.")

            print(f"--- Preparando para Enviar para Z-API ---")
            print(f"Número: {numero}")
            print(f"Mensagem: {mensagem_resposta}")

            instance_id = os.getenv("INSTANCE_ID")
            token = os.getenv("TOKEN")
            
            # **MELHORIA**: Captura a resposta da Z-API para depuração
            resposta_zapi = await client.post(
                f"https://api.z-api.io/instances/{instance_id}/token/{token}/send-text",
                json={"phone": numero, "message": mensagem_resposta},
                timeout=30.0
            )
            
            print(f"--- Resposta da Z-API ---")
            print(f"Status Code: {resposta_zapi.status_code}")
            print(f"Body: {resposta_zapi.text}")

            resposta_zapi.raise_for_status()

    except Exception as e:
        print(f"!!! Erro no Webhook /whatsapp: {e} !!!")

    return {"status": "ok"}

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from databases import Database
from sqlalchemy import create_engine, MetaData, Table, Column, String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
import uuid, json, httpx, os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from openai import AsyncOpenAI # Importa a biblioteca da OpenAI

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

# Tabela de histórico com timestamp para expiração e modo snooze
historico_conversas = Table(
    "historico_conversas",
    metadata,
    Column("telefone", String, primary_key=True),
    Column("historico", Text, nullable=False),
    Column("last_updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("snoozed_until", DateTime(timezone=True), nullable=True)
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
    Função assíncrona para chamar a API de chat do OpenRouter.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "Referer": f"{os.getenv('PUBLIC_URL')}" 
    }
    body = {
        "model": "openai/gpt-4o-2024-11-20", 
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

async def transcrever_audio(audio_bytes: bytes) -> str | None:
    """
    Envia os bytes de um áudio para a API do Whisper da OpenAI.
    """
    try:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        audio_file = ("audio.ogg", audio_bytes, "audio/ogg")
        transcription = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcription.text
    except Exception as e:
        print(f"Erro na transcrição de áudio com a API da OpenAI: {e}")
        return None

async def baixar_audio_bytes(url: str) -> bytes | None:
    """
    Baixa o conteúdo de um áudio de uma URL e retorna os bytes brutos.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            zapi_headers = {"Client-Token": os.getenv("CLIENT_TOKEN")}
            resposta = await client.get(url, headers=zapi_headers)
            resposta.raise_for_status()
            return resposta.content
    except Exception as e:
        print(f"Erro ao baixar áudio: {e}")
        return None

# --- Endpoints da API ---

@app.get("/")
async def root():
    """Endpoint GET para verificar se a API está no ar."""
    return {"message": "API da InovaTech Solutions no ar!"}

@app.head("/")
async def head_root():
    """Endpoint HEAD para checagem de status por serviços de monitoramento."""
    return Response(status_code=200)

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
        if isinstance(msg, dict) and msg.get("role") in ["user", "assistant"]:
            messages.append(msg)
    messages.append({"role": "user", "content": dados.mensagem})

    resposta = await chamar_ia(messages)

    if isinstance(resposta, dict) and all(k in resposta for k in ["nome", "email", "telefone", "servico"]):
        try:
            novo_orcamento = {"id": uuid.uuid4(), **resposta}
            query = orcamentos.insert().values(**novo_orcamento)
            await database.execute(query)
            return {"reply": "Orçamento recebido com sucesso! Nossa equipe entrará em contato em breve."}
        except Exception as e:
            print(f"Erro ao salvar orçamento no banco: {e}")
            return {"reply": "Recebi seus dados, mas tive um problema para registrar."}
    
    if isinstance(resposta, str):
        return {"reply": resposta}
    
    return {"reply": "Não entendi sua resposta. Poderia reformular?"}

@app.post("/whatsapp")
async def receber_mensagem_zapi(request: Request):
    try:
        payload = await request.json()
        print(f"--- Payload Recebido --- \n{json.dumps(payload, indent=2)}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Payload JSON inválido.")

    numero_contato = payload.get("phone")
    if not numero_contato:
        raise HTTPException(status_code=400, detail="O campo 'phone' é obrigatório.")

    if payload.get("fromMe"):
        snooze_until = datetime.now(timezone.utc) + timedelta(minutes=30)
        select_query = historico_conversas.select().where(historico_conversas.c.telefone == numero_contato)
        if await database.fetch_one(select_query):
            update_query = historico_conversas.update().where(historico_conversas.c.telefone == numero_contato).values(snoozed_until=snooze_until)
            await database.execute(update_query)
        else:
            insert_query = historico_conversas.insert().values(telefone=numero_contato, historico="[]", last_updated_at=func.now(), snoozed_until=snooze_until)
            await database.execute(insert_query)
        print(f"--- Modo manual ativado para {numero_contato} por 30 minutos. ---")
        return {"status": "ok", "message": "Modo manual ativado."}

    texto_da_mensagem = payload.get("text", {}).get("message") if isinstance(payload.get("text"), dict) else None
    audio_url = payload.get("audio", {}).get("audioUrl") if isinstance(payload.get("audio"), dict) else None
    
    conteudo_processar = None

    if texto_da_mensagem:
        conteudo_processar = texto_da_mensagem
    elif audio_url:
        print("--- Detectado áudio. Iniciando processo de transcrição. ---")
        audio_bytes = await baixar_audio_bytes(audio_url)
        if audio_bytes:
            texto_transcrito = await transcrever_audio(audio_bytes)
            if texto_transcrito:
                print(f"--- Texto Transcrito --- \n{texto_transcrito}")
                # **CORREÇÃO**: Enviamos apenas o texto transcrito para a IA, sem a nota.
                conteudo_processar = texto_transcrito
            else:
                conteudo_processar = "[Não foi possível transcrever o áudio. Peça para o usuário repetir.]"
        else:
            conteudo_processar = "[Houve um erro ao baixar o áudio. Peça para o usuário reenviar.]"

    if not conteudo_processar:
        return {"status": "ok", "message": "Ignorando mensagem sem conteúdo de texto ou áudio."}

    try:
        query_select = historico_conversas.select().where(historico_conversas.c.telefone == numero_contato)
        resultado = await database.fetch_one(query_select)
        
        historico_recuperado = []
        if resultado:
            if resultado["snoozed_until"] and resultado["snoozed_until"] > datetime.now(timezone.utc):
                return {"status": "ok", "message": "Conversa em modo manual."}
            if datetime.now(timezone.utc) - resultado["last_updated_at"] < timedelta(hours=24):
                historico_recuperado = json.loads(resultado["historico"])
            else:
                print(f"--- Histórico expirado para {numero_contato}. ---")

        async with httpx.AsyncClient() as client:
            public_url = os.getenv("PUBLIC_URL")
            resposta_chat = await client.post(
                 f"{public_url.rstrip('/')}/chat",
                 json={"mensagem": conteudo_processar, "historico": historico_recuperado},
                 timeout=90.0
            )
            resposta_chat.raise_for_status()
            
            dados = resposta_chat.json()
            mensagem_resposta = dados.get("reply", "Não consegui gerar uma resposta.")

            historico_atualizado = historico_recuperado + [
                {"role": "user", "content": conteudo_processar},
                {"role": "assistant", "content": mensagem_resposta}
            ]
            historico_str = json.dumps(historico_atualizado[-20:])

            if resultado:
                query_db = historico_conversas.update().where(historico_conversas.c.telefone == numero_contato).values(historico=historico_str, last_updated_at=func.now(), snoozed_until=None)
            else:
                query_db = historico_conversas.insert().values(telefone=numero_contato, historico=historico_str, last_updated_at=func.now(), snoozed_until=None)
            
            await database.execute(query_db)
            print(f"--- Histórico salvo para {numero_contato} ---")

            instance_id = os.getenv("INSTANCE_ID")
            token = os.getenv("TOKEN")
            client_token = os.getenv("CLIENT_TOKEN") 
            
            zapi_headers = {"Client-Token": client_token}
            
            await client.post(
                f"https://api.z-api.io/instances/{instance_id}/token/{token}/send-text",
                json={"phone": numero_contato, "message": mensagem_resposta},
                headers=zapi_headers, 
                timeout=30.0
            )

    except Exception as e:
        print(f"!!! Erro no Webhook /whatsapp: {e} !!!")

    return {"status": "ok"}

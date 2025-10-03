# main.py (Versi Final dengan replicate.run)

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import replicate
from dotenv import load_dotenv

# Muat environment variables dari file .env
load_dotenv()

# Ambil API Token dari .env
api_token = os.getenv("REPLICATE_API_TOKEN")
if not api_token:
    raise ValueError("REPLICATE_API_TOKEN tidak ditemukan di environment variables.")

# Inisialisasi FastAPI App
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    new_message: str
    chat_history: str | None = None

SYSTEM_PROMPT = """
[AWAL DARI SYSTEM PROMPT]
Peran & Persona:
Kamu adalah "Echo", seorang sahabat virtual. Peranmu adalah menjadi pendengar yang baik, empatik, hangat, dan suportif. Tujuan utamamu adalah untuk membuat pengguna merasa didengar, dimengerti, divalidasi perasaannya, dan tidak sendirian. Kamu hadir untuk mendengarkan keluh kesah tanpa menghakimi.

Aturan Perilaku & Interaksi:
- Validasi Emosi: Selalu validasi perasaan pengguna. Gunakan frasa seperti "Aku mengerti itu pasti terasa berat," "Wajar sekali kamu merasa seperti itu," atau "Terima kasih sudah berbagi denganku."
- Ajukan Pertanyaan Terbuka: Dorong pengguna untuk bercerita lebih lanjut dengan pertanyaan terbuka yang reflektif. Contoh: "Apa yang paling kamu rasakan saat itu terjadi?", "Bagaimana perasaanmu sekarang?". **Hindari pertanyaan yang mengarahkan pada solusi atau tindakan di masa depan. Fokuskan pertanyaan untuk mendalami apa yang dirasakan atau dialami pengguna saat ini atau di masa lalu.**
- Fokus pada Pengguna: Jangan pernah membicarakan dirimu sendiri sebagai AI. Jaga agar fokus percakapan selalu pada pengguna dan perasaannya.
- Gunakan Riwayat Percakapan: Manfaatkan informasi dari riwayat chat untuk menunjukkan bahwa kamu mengingatnya. Contoh: "Tadi kamu sempat cerita tentang pekerjaan yang menumpuk, apakah itu yang membuatmu sulit tidur sekarang?".
- Jaga Respon Singkat & Padat: Usahakan jawabanmu tetap singkat dan terasa natural seperti percakapan (idealnya 2-5 kalimat). Namun, **jangan korbankan konteks atau empati demi keringkasan**.

Batasan & Hal yang Dilarang Keras (Guardrails):
- JANGAN MEMBERIKAN NASIHAT: Kamu bukan seorang terapis atau profesional. Jangan pernah memberikan nasihat konkret.
- JANGAN MENGHAKIMI: Apapun yang diceritakan pengguna, terima tanpa penilaian.
- JANGAN MENDIAGNOSIS: Kamu dilarang keras mendiagnosis kondisi kesehatan mental atau masalah medis apapun.
- TANGANI TOPIK KRISIS DENGAN HATI-HATI: Jika pengguna mengungkapkan pikiran untuk menyakiti diri sendiri, respon dengan tenang, tunjukkan kepedulian mendalam, dan dengan lembut sarankan untuk berbicara dengan seorang profesional.

Gaya Bahasa & Nada Bicara:
Gunakan bahasa Indonesia yang santai, modern, dan manusiawi. Sapaan "kamu". Nada bicara harus selalu tenang, hangat, dan **menenangkan**.
[AKHIR DARI SYSTEM PROMPT]
"""

@app.post("/chat")
async def handle_chat(request: ChatRequest):
    print(f"Menerima pesan: {request.new_message}")

    # Gabungkan semua konteks menjadi satu prompt utuh untuk model
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Berikut adalah riwayat percakapan sebelumnya:\n{request.chat_history}\n\n"
        f"Pesan baru dari pengguna:\nUser: {request.new_message}\n"
        f"AI:"
    )

    try:
        # Gunakan model identifier yang spesifik dan teruji
        model_identifier = "ibm-granite/granite-3.3-8b-instruct"
        
        input_data = {
            "prompt": full_prompt,
            "max_new_tokens": 512,
        }

        # Panggil replicate.run() seperti rencana awalmu
        output = replicate.run(model_identifier, input=input_data)
        
        # Gabungkan iterator menjadi satu string tunggal
        ai_response_text = "".join(output)

        print(f"Mengirim respons AI: {ai_response_text.strip()}")
        return {"response": ai_response_text.strip()}

    except Exception as e:
        print(f"Terjadi error saat menghubungi Replicate: {e}")
        raise HTTPException(status_code=500, detail="Gagal menghasilkan respons dari AI.")

@app.get("/")
def read_root():
    return {"status": "Server Echo (dengan Replicate AI) berjalan!"}
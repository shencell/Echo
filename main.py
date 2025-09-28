# main.py (Versi dengan Integrasi AI)

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Impor library baru
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Muat environment variables dari file .env
load_dotenv()

# Ambil API Key dan konfigurasikan Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY tidak ditemukan di environment variables.")
genai.configure(api_key=api_key)

# Inisialisasi model Generative AI
model = genai.GenerativeModel('gemini-2.5-pro')

# --- Inisialisasi FastAPI App (sama seperti sebelumnya) ---
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    new_message: str
    chat_history: str | None = None

# --- [SYSTEM PROMPT UNTUK "ECHO"] ---
# 2. Gabungkan System Prompt, riwayat, dan pesan baru
# Ini adalah "jiwa" dari AI Echo yang Anda definisikan sebelumnya.
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
    """
    Endpoint ini sekarang memanggil API Gemini untuk mendapatkan respons nyata.
    """
    print(f"Menerima pesan: {request.new_message}")

    # Gabungkan semua konteks menjadi satu prompt utuh
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"[BAGIAN 2: RIWAYAT CHAT]\n{request.chat_history}\n\n"
        f"[BAGIAN 3: PESAN BARU DARI PENGGUNA]\nUser: {request.new_message}\n"
        f"AI:"
    )

    try:
        # 3. Lakukan panggilan ke model AI
        # Kita menggunakan generate_content_async karena ini adalah fungsi async
        response = await model.generate_content_async(full_prompt)

        # 4. Ambil teks respons dan kembalikan dalam format JSON
        ai_response_text = response.text
        print(f"Mengirim respons AI: {ai_response_text}")
        return {"response": ai_response_text}

    except Exception as e:
        # Tangani jika ada error dari API
        print("==========================================================")
        print(f"!!! CRITICAL ERROR WHEN CALLING AI API !!!")
        print(f"Error Type: {type(e)}")
        print(f"Error Details: {e}")
        print("==========================================================")
        raise HTTPException(status_code=500, detail="Gagal menghasilkan respons dari AI.")

@app.get("/")
def read_root():
    return {"status": "Server Echo (dengan AI) berjalan!"}
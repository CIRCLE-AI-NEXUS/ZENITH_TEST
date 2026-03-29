"""
CIRCLE AI — Customer Chatbot + ZENITH ACTON Avatar
Video avatar terintegrasi dengan auto-greeting dan gesture trigger
"""

import streamlit as st
import re
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════
#  CIRCLE AI BRAIN
# ══════════════════════════════════════════════

class CircleAIBrain:
    def __init__(self):
        self.business_info = {}
        self.questions = []
        self.answers = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.business_name = "AI Assistant"

    def load(self, data: dict):
        self.business_info = data.get("business_info", {})
        self.business_name = self.business_info.get("name", "AI Assistant")
        self.questions = []
        self.answers = []

        for pair in data.get("qa_pairs", []):
            for q in pair.get("questions", []):
                self.questions.append(q.lower().strip())
                self.answers.append(pair.get("answer", ""))

        self._auto_qa()

        if self.questions:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                token_pattern=r'[a-zA-Z0-9\u00C0-\u024F]+'
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def _auto_qa(self):
        info = self.business_info
        if info.get("name"):
            self._add(["nama bisnis", "ini bisnis apa", "siapa kalian", "ini siapa", "nama toko", "nama cafe"],
                f"Kami adalah **{info['name']}**. {info.get('tagline','')}")
        if info.get("description"):
            self._add(["tentang kalian", "profil bisnis", "apa yang ditawarkan"], info["description"])
        if info.get("location"):
            self._add(["lokasi", "alamat", "dimana", "maps"],
                f"📍 **{info['location']}**" + (f"\n🗺️ {info['maps_link']}" if info.get("maps_link") else ""))
        if info.get("hours"):
            self._add(["jam buka", "operasional", "buka jam berapa", "tutup jam berapa", "kapan buka"],
                "🕐 **Jam Operasional:**\n" + "\n".join([f"• {k}: {v}" for k, v in info["hours"].items()]))
        if info.get("contact"):
            c = info["contact"]
            parts = []
            if c.get("whatsapp"): parts.append(f"📱 WhatsApp: {c['whatsapp']}")
            if c.get("instagram"): parts.append(f"📸 Instagram: {c['instagram']}")
            if c.get("email"): parts.append(f"📧 Email: {c['email']}")
            if parts:
                self._add(["kontak", "hubungi", "whatsapp", "wa", "instagram", "email", "telepon"],
                    "📬 **Hubungi Kami:**\n" + "\n".join(parts))
        if info.get("products"):
            lines = [f"• **{p['name']}**" + (f" — {p['price']}" if p.get("price") else "")
                     + (f"\n  {p['description']}" if p.get("description") else "") for p in info["products"]]
            self._add(["produk", "menu", "paket", "harga", "daftar harga", "ada apa saja"],
                "🛍️ **Produk & Layanan:**\n\n" + "\n".join(lines))
            for p in info["products"]:
                n = p["name"].lower()
                self._add([f"harga {n}", f"berapa {n}", f"info {n}", f"tentang {n}"],
                    f"**{p['name']}**\n" + (f"💰 {p['price']}\n" if p.get("price") else "")
                    + (p.get("description","")) + (f"\n✨ {p['features']}" if p.get("features") else ""))
        if info.get("faq"):
            for faq in info["faq"]:
                self._add(faq.get("questions", []), faq.get("answer", ""))
        if info.get("promo"):
            self._add(["promo", "diskon", "promosi", "penawaran", "ada promo"],
                f"🎉 **Promo Spesial:**\n{info['promo']}")
        if info.get("recommendation"):
            self._add(["rekomendasi", "saran", "terbaik", "populer", "favorit"],
                f"⭐ **Rekomendasi:**\n{info['recommendation']}")

    def _add(self, questions, answer):
        for q in questions:
            self.questions.append(q.lower().strip())
            self.answers.append(answer)

    def respond(self, user_input: str):
        text = re.sub(r'[^\w\s]', ' ', user_input.lower().strip())
        if any(g in text for g in ["halo","hai","hi","hello","hey","selamat","assalam"]) and len(text.split()) <= 5:
            return f"Yo! 👋 Gue ZENITH — asisten AI **{self.business_name}**!\n\nMau tanya-tanya dulu atau langsung cari yang spesifik? 😊", 1.0
        if any(t in text for t in ["terima kasih","makasih","thanks","thx"]):
            return "Sama-sama! 😊 Ada lagi yang bisa dibantu?", 1.0
        if any(b in text for b in ["bye","dadah","sampai jumpa"]):
            return f"Sampai jumpa! Makasih udah mampir ke **{self.business_name}** 👋", 1.0
        try:
            scores = cosine_similarity(self.vectorizer.transform([text]), self.tfidf_matrix)[0]
            best_idx = int(np.argmax(scores))
            if float(scores[best_idx]) > 0.12:
                return self.answers[best_idx], float(scores[best_idx])
            wa = self.business_info.get("contact", {}).get("whatsapp", "")
            return "Wah itu gue kurang tau bro 😅\nMending langsung tanya tim kita aja biar akurat!" + (f"\n📱 WhatsApp: **{wa}**" if wa else ""), 0.0
        except:
            return "Maaf, coba tanya dengan cara lain ya!", 0.0


# ══════════════════════════════════════════════
#  ZENITH ACTON — VIDEO POOL
#  Tambahkan URL video sesuai animasi yang tersedia
# ══════════════════════════════════════════════

ZENITH_VIDEOS = {

    # Greeting — lambaikan tangan (video kamu yang ada sekarang)
    "greeting": [
        "https://drive.google.com/uc?export=download&id=1JrGGTkso7JnCtPZfZ74kardgzraWOxVs",
        # Tambah URL lain kalau ada animasi greeting lain:
        # "https://drive.google.com/uc?export=download&id=XXXXX",
    ],

    # Thinking — saat AI processing (isi nanti setelah animasi jadi)
    "thinking": [
        # "https://drive.google.com/uc?export=download&id=XXXXX",
    ],

    # Talking — saat AI jawab (isi nanti)
    "talking": [
        # "https://drive.google.com/uc?export=download&id=XXXXX",
    ],

    # Idle — diam natural (isi nanti)
    "idle": [
        # "https://drive.google.com/uc?export=download&id=XXXXX",
    ],

    # Farewell — pamit (isi nanti)
    "farewell": [
        # "https://drive.google.com/uc?export=download&id=XXXXX",
    ],

    # Confused — tidak tau jawaban (isi nanti)
    "confused": [
        # "https://drive.google.com/uc?export=download&id=XXXXX",
    ],
}

# Fallback — video yang dipakai kalau pool kosong
FALLBACK_VIDEO = "https://drive.google.com/uc?export=download&id=1JrGGTkso7JnCtPZfZ74kardgzraWOxVs"


def get_random_video(state: str) -> str:
    """Pilih video random dari pool state tertentu."""
    pool = [v for v in ZENITH_VIDEOS.get(state, []) if v]

    # Kalau pool kosong → pakai fallback
    if not pool:
        return FALLBACK_VIDEO

    # Hindari video sama berulang
    last = st.session_state.get("last_video", "")
    available = [v for v in pool if v != last]
    if not available:
        available = pool

    chosen = random.choice(available)
    st.session_state.last_video = chosen
    return chosen


def detect_anim_state(user_input: str, confidence: float) -> str:
    """Deteksi state animasi berdasarkan input user."""
    text = user_input.lower()

    # Greeting
    if any(w in text for w in ["halo","hai","hi","hello","hey","halo ai"]):
        return "greeting"

    # Farewell
    if any(w in text for w in ["bye","dadah","makasih","sampai jumpa","terima kasih"]):
        return "farewell"

    # Confused — confidence rendah
    if confidence < 0.12:
        return "confused"

    # Research — pertanyaan panjang
    if len(text.split()) > 8:
        return "talking"

    # Default
    return "thinking"


# ══════════════════════════════════════════════
#  KNOWLEDGE BASE
# ══════════════════════════════════════════════

KNOWLEDGE = {
  "business_info": {
    "name": "Uluwatu Dream Villa",
    "tagline": "Luxury Feels, Affordable Price — Bali at Its Finest",
    "description": "Uluwatu Dream Villa adalah villa mewah dengan harga terjangkau yang berlokasi di tebing Uluwatu, Bali. Nikmati pemandangan laut lepas Samudra Hindia, kolam renang infinity pool pribadi, dan suasana Bali yang autentik. Cocok untuk pasangan, keluarga, maupun group liburan.",
    "location": "Jl. Pantai Suluban No. 88, Uluwatu, Pecatu, Kuta Selatan, Bali",
    "maps_link": "https://maps.google.com/?q=Uluwatu+Dream+Villa+Bali",
    "hours": {
      "Check-in": "14.00 WITA",
      "Check-out": "12.00 WITA",
      "Front Desk": "24 jam (selalu siap melayani)",
      "Kolam Renang": "06.00 - 22.00 WITA",
      "Restoran Villa": "07.00 - 22.00 WITA"
    },
    "contact": {
      "whatsapp": "+62 813-3800-8888",
      "instagram": "@uluwatudreamvilla",
      "email": "booking@uluwatudreamvilla.com"
    },
    "products": [
      {"name": "Deluxe Ocean View Room", "price": "Rp 850.000 / malam",
       "description": "Kamar deluxe dengan balkon private menghadap Samudra Hindia.",
       "features": "Kapasitas 2 orang • Sarapan included • Free WiFi"},
      {"name": "Private Pool Villa", "price": "Rp 2.100.000 / malam",
       "description": "Villa pribadi dengan infinity pool menghadap laut.",
       "features": "Kapasitas 2-4 orang • Sarapan included • Butler service"},
      {"name": "Family Cliff Villa", "price": "Rp 3.500.000 / malam",
       "description": "Villa keluarga 3 kamar tidur di tepi tebing.",
       "features": "Kapasitas 6 orang • BBQ area • Free airport transfer"},
      {"name": "Honeymoon Cliffside Suite", "price": "Rp 2.800.000 / malam",
       "description": "Suite romantis dengan jacuzzi outdoor & candle light dinner.",
       "features": "Kapasitas 2 orang • Rose bath • Free couples massage 60 menit"},
      {"name": "Group Retreat Package", "price": "Mulai Rp 8.000.000 / malam",
       "description": "Paket eksklusif untuk group 10-20 orang.",
       "features": "Private chef • Activity organizer • Free shuttle"}
    ],
    "promo": "🌙 PROMO RAMADHAN 40% OFF!\n• Gratis welcome dates & kurma\n• Gratis sahur box ke kamar\n• Gratis one way airport transfer\n• Diskon 30% couple spa\nKode: RAMADHAN40 | Min. 2 malam\nWA: +62 813-3800-8888",
    "recommendation": "💑 Honeymoon → Honeymoon Cliffside Suite\n👨‍👩‍👧 Keluarga → Family Cliff Villa\n👥 Group → Group Retreat Package\n🎯 First timer → Private Pool Villa",
    "faq": [
      {"questions": ["cara booking","cara pesan","reservasi","booking gimana"],
       "answer": "📋 Cara Booking:\n1️⃣ WA: +62 813-3800-8888\n2️⃣ Pilih kamar & tanggal\n3️⃣ DP 30% untuk konfirmasi\n4️⃣ Pelunasan H-7 check-in"},
      {"questions": ["fasilitas","ada apa","amenities"],
       "answer": "🏊 Fasilitas:\n• Infinity pool view laut\n• Restoran & bar rooftop\n• Spa & massage center\n• Yoga deck tepi tebing\n• Free WiFi 100 Mbps\n• Room service 24 jam"},
      {"questions": ["berapa jauh bandara","jarak bandara","transport bandara"],
       "answer": "✈️ Dari Bandara Ngurah Rai:\n±25 km • 45-60 menit\nGrab/Gojek: Rp 80-120rb\nBeberapa paket include free transfer!"},
      {"questions": ["bawa anak","family friendly","ramah anak"],
       "answer": "👶 Sangat family friendly!\n• Baby cot gratis (request dulu)\n• Menu anak tersedia\n• Area bermain anak\n• Family Cliff Villa paling cocok"},
      {"questions": ["cancel","refund","batalkan","pembatalan"],
       "answer": "❌ Kebijakan Batal:\n• 14+ hari: Refund 100%\n• 7-13 hari: Refund 50%\n• <7 hari: DP hangus\n• Reschedule gratis hingga H-3"},
      {"questions": ["wisata sekitar","tempat wisata","aktivitas"],
       "answer": "🌊 Wisata Dekat:\n• Pura Uluwatu (5 menit)\n• Pantai Suluban (5 menit)\n• Pantai Padang Padang (10 menit)\n• GWK Cultural Park (20 menit)"},
      {"questions": ["kolam renang","infinity pool","swimming pool"],
       "answer": "🏊 Kolam Renang:\n• Infinity Pool Umum: 06.00-22.00\n• Private Pool: khusus villa tertentu\n• Rooftop Jacuzzi: khusus Honeymoon Suite"},
      {"questions": ["sarapan","breakfast","makanan","restoran"],
       "answer": "🍳 Sarapan included semua paket!\nServed 07.00-10.00 WITA\nMenu: Continental + Indonesian\nLokasi: Rooftop dengan view laut 🌅"}
    ]
  },
  "qa_pairs": [
    {"questions": ["worth it","mahal tidak","harga sesuai","recommended"],
     "answer": "💯 Sangat worth it!\n✅ View laut dramatis\n✅ Infinity pool pribadi\n✅ Sarapan included\n✅ Service bintang 5\n500+ review bintang 5 di Google! ⭐"},
    {"questions": ["promo ramadhan","diskon ramadhan","ramadhan deal"],
     "answer": "🌙 Promo Ramadhan 40% OFF!\nKode: RAMADHAN40\nMin. 2 malam • Berlaku sampai 10 Syawal\nWA: +62 813-3800-8888"},
    {"questions": ["sahur","fasilitas ramadhan","buka puasa"],
     "answer": "🌙 Fasilitas Ramadhan:\n• Sahur box ke kamar (gratis)\n• Buka puasa set di restoran\n• Takjil welcome (dates & kurma)\n• Sajadah & arah kiblat di kamar\n• Musholla tersedia"},
    {"questions": ["honeymoon","bulan madu","romantic"],
     "answer": "💑 Honeymoon Suite:\n• Jacuzzi outdoor\n• Rose bath setup\n• Dekorasi bunga & candle\n• Free couples massage 60 menit\nRp 2.800.000/malam\nWA: +62 813-3800-8888"}
  ]
}

# ══════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title=f"ZENITH — {KNOWLEDGE['business_info']['name']}",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.stApp, .block-container, section.main {
    background: #ffffff !important;
    color: #1a1a1a !important;
}
.block-container { padding-top: 0.5rem !important; }

/* Avatar container */
.zenith-avatar {
    width: 100%;
    max-width: 280px;
    margin: 0 auto 8px auto;
    border-radius: 16px;
    overflow: hidden;
    background: #0a0a1a;
    display: block;
}

/* Header */
.chat-header {
    text-align: center;
    padding: 8px 0;
    margin-bottom: 12px;
}
.chat-name {
    font-size: 1rem;
    font-weight: 700;
    color: #0099cc;
}
.chat-status {
    font-size: 0.72rem;
    color: #00a896;
    margin-top: 2px;
}

/* Bubbles */
.bubble-user {
    background: linear-gradient(135deg, #00c2a8, #0099cc);
    color: #ffffff;
    padding: 10px 14px;
    border-radius: 16px 16px 4px 16px;
    margin: 6px 0;
    margin-left: 20%;
    font-size: 0.88rem;
    line-height: 1.5;
}
.bubble-ai {
    background: #1a73e8;
    color: #ffffff;
    padding: 10px 14px;
    border-radius: 16px 16px 16px 4px;
    margin: 6px 0;
    margin-right: 20%;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* Suggestion buttons */
div[data-testid="stButton"] > button {
    background: #f0f8ff !important;
    color: #0066cc !important;
    border: 1px solid #b3d9ff !important;
    border-radius: 20px !important;
    font-size: 0.78rem !important;
    padding: 4px 10px !important;
}
div[data-testid="stButton"] > button:hover {
    background: #dceefb !important;
    border-color: #0099cc !important;
}

/* Video player — sembunyikan kontrol default */
video {
    border-radius: 12px;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  INIT SESSION STATE
# ══════════════════════════════════════════════
if "brain" not in st.session_state:
    b = CircleAIBrain()
    b.load(KNOWLEDGE)
    st.session_state.brain = b

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_video" not in st.session_state:
    st.session_state.current_video = get_random_video("greeting")

if "has_greeted" not in st.session_state:
    st.session_state.has_greeted = False

if "last_video" not in st.session_state:
    st.session_state.last_video = ""

brain    = st.session_state.brain
biz_name = KNOWLEDGE["business_info"]["name"]

# ══════════════════════════════════════════════
#  AUTO GREETING ON FIRST LOAD
# ══════════════════════════════════════════════
if not st.session_state.has_greeted:
    st.session_state.has_greeted   = True
    st.session_state.current_video = get_random_video("greeting")
    st.session_state.messages.append({
        "role"   : "ai",
        "content": f"Yo! 👋 Gue ZENITH — asisten AI **{biz_name}**!\n\nMau tanya-tanya dulu atau langsung cari yang spesifik? 😊"
    })

# ══════════════════════════════════════════════
#  HELPER — PROCESS CHAT
# ══════════════════════════════════════════════
def process_chat(text: str):
    response, confidence = brain.respond(text)
    anim_state = detect_anim_state(text, confidence)
    st.session_state.current_video = get_random_video(anim_state)
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.messages.append({"role": "ai",  "content": response})

# ══════════════════════════════════════════════
#  RENDER — ZENITH AVATAR VIDEO
# ══════════════════════════════════════════════
st.markdown('<div class="zenith-avatar">', unsafe_allow_html=True)
st.video(
    st.session_state.current_video,
    autoplay=True,
    loop=True,
    muted=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  RENDER — HEADER
# ══════════════════════════════════════════════
st.markdown(f"""
<div class="chat-header">
    <div class="chat-name">⚡ ZENITH ACTON</div>
    <div class="chat-status">● Online — {biz_name} AI</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  RENDER — CHAT HISTORY
# ══════════════════════════════════════════════
for msg in st.session_state.messages:
    css  = "bubble-user" if msg["role"] == "user" else "bubble-ai"
    icon = "👤" if msg["role"] == "user" else "⚡"
    st.markdown(
        f'<div class="{css}">{icon} {msg["content"]}</div>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════
#  RENDER — SUGGESTED QUESTIONS
# ══════════════════════════════════════════════
if len(st.session_state.messages) <= 1:
    st.caption("💡 Mau tanya soal:")
    sugs = [
        "halo AI!",
        "Ada promo apa?",
        "Menu & harga?",
        "Cara reservasi?",
        "Dimana lokasinya?",
        "Fasilitas apa saja?"
    ]
    cols = st.columns(2)
    for i, s in enumerate(sugs):
        with cols[i % 2]:
            if st.button(s, key=f"s{i}", use_container_width=True):
                process_chat(s)
                st.rerun()

# ══════════════════════════════════════════════
#  RENDER — CHAT INPUT
# ══════════════════════════════════════════════
user_input = st.chat_input("Tanya ZENITH...")
if user_input:
    process_chat(user_input)
    st.rerun()

# ══════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;color:#bbbbbb;font-size:0.65rem;margin-top:16px;padding-bottom:10px">
    ⚡ ZENITH ACTON — Powered by CIRCLE AI 🧠
</div>
""", unsafe_allow_html=True)
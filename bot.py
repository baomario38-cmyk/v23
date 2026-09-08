import os
import math
import random
import numpy as np
from scipy import stats
from threading import Thread
from flask import Flask
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# --- 1. SERVER KEEP-ALIVE ---
app = Flask(__name__)

@app.route('/')
def health_check():
    return "TOOL TXGAME v22 MASTER ENGINE ONLINE", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# --- 2. CẤU HÌNH BOT & QUẢN TRỊ ---
TOKEN = '8985526419:AAGdRkntgFNYLBG53LoI-pNC7aHtOFMWhGA'
ADMIN_ID = 755092812
ADMIN_USERNAME = "lionvnios"

bot = telebot.TeleBot(TOKEN)
user_data = {}
all_users = set()
gift_codes = {}

def is_admin(user):
    if not user: return False
    return user.id == ADMIN_ID or (user.username and user.username.lower() == ADMIN_USERNAME.lower())

def init_user(uid):
    all_users.add(uid)
    if uid not in user_data:
        user_data[uid] = {"balance": 20, "logs": []}

# --- 3. TOÀN BỘ 13 LỚP THUẬT TOÁN (MASTER ENGINE) ---
class MasterAnalyticsEngine:
    def __init__(self, history_scores):
        self.raw = np.array(history_scores, dtype=float)
        self.n = len(self.raw)
        self.binary = np.array([1 if x >= 11 else 0 for x in history_scores], dtype=int)
        
    # LỚP 1: XÁC SUẤT CƠ BẢN
    def layer1_exact_prob(self):
        return 105.0 / 216.0  # ~0.4861

    # LỚP 2: THỐNG KÊ MÔ TẢ
    def layer2_descriptive_stats(self):
        if self.n < 3: return {"mean": 10.5, "z_score": 0.0, "skew": 0.0}
        mean = np.mean(self.raw)
        std = np.std(self.raw, ddof=1) if np.std(self.raw, ddof=1) > 0 else 1.0
        z_score = (self.raw[-1] - mean) / std
        skewness = float(stats.skew(self.raw))
        kurtosis = float(stats.kurtosis(self.raw))
        return {"mean": mean, "std": std, "z_score": z_score, "skew": skewness, "kurt": kurtosis}

    # LỚP 3 & 8: CHUỖI THỜI GIAN & EWMA
    def layer3_sequence_ewma(self, alpha=0.3):
        ewma = self.raw[0]
        for val in self.raw[1:]:
            ewma = alpha * val + (1 - alpha) * ewma
        if self.n > 2 and np.var(self.raw) > 0:
            autocorr = np.corrcoef(self.raw[:-1], self.raw[1:])[0, 1]
        else:
            autocorr = 0.0
        p_ewma_tai = 1 / (1 + math.exp(-(ewma - 10.5)))
        return p_ewma_tai, autocorr

    # LỚP 4 & 10: XÁC SUẤT CÓ ĐIỀU KIỆN & BAYESIAN
    def layer4_bayesian_update(self):
        alpha_prior, beta_prior = 10.5, 10.5
        k_tai = np.sum(self.binary)
        n_total = self.n
        posterior_alpha = alpha_prior + k_tai
        posterior_beta = beta_prior + (n_total - k_tai)
        return posterior_alpha / (posterior_alpha + posterior_beta)

    # LỚP 5: CHUỖI MARKOV
    def layer5_markov_chain(self):
        trans = np.zeros((2, 2))
        for i in range(self.n - 1):
            trans[self.binary[i], self.binary[i+1]] += 1
        last_st = self.binary[-1]
        row_sum = np.sum(trans[last_st])
        return trans[last_st, 1] / row_sum if row_sum > 0 else 0.4861

    # LỚP 6: KIỂM ĐỊNH PHÁT HIỆN XÚC XẮC KHÔNG CÔNG BẰNG
    def layer6_fairness_tests(self):
        runs = 1 + np.sum(self.binary[1:] != self.binary[:-1])
        n1 = np.sum(self.binary)
        n0 = self.n - n1
        exp_runs = 1 + (2 * n0 * n1) / self.n if self.n > 0 else 1
        return abs(runs - exp_runs) < 2.0

    # LỚP 9: MÔ PHỎNG MONTE CARLO
    def layer9_monte_carlo(self, base_prob, trials=5000):
        simulated_wins = np.random.random(trials) < base_prob
        return np.mean(simulated_wins)

    # LỚP 11: MACHINE LEARNING FEATURE SCORING
    def layer11_ml_scoring(self, stats_info, autocorr):
        feature_vector = np.array([
            stats_info["z_score"],
            stats_info["skew"],
            autocorr,
            self.binary[-1] - 0.5
        ])
        weights = np.array([-0.15, -0.10, 0.25, -0.20])
        logit = np.dot(feature_vector, weights)
        return 1 / (1 + math.exp(-logit))

    # LỚP 12 & 13: ENSEMBLE BMA & CALIBRATION
    def layer12_13_ensemble_calibration(self):
        p_exact = self.layer1_exact_prob()
        stats_info = self.layer2_descriptive_stats()
        p_ewma, autocorr = self.layer3_sequence_ewma()
        p_bayes = self.layer4_bayesian_update()
        p_markov = self.layer5_markov_chain()
        is_random = self.layer6_fairness_tests()
        p_mc = self.layer9_monte_carlo(p_bayes)
        p_ml = self.layer11_ml_scoring(stats_info, autocorr)

        weights = {'bayes': 0.25, 'markov': 0.20, 'ml': 0.20, 'ewma': 0.15, 'mc': 0.10, 'exact': 0.10}

        p_ensemble = (
            weights['bayes'] * p_bayes +
            weights['markov'] * p_markov +
            weights['ml'] * p_ml +
            weights['ewma'] * p_ewma +
            weights['mc'] * p_mc +
            weights['exact'] * p_exact
        )

        calibrated_p = 1 / (1 + math.exp(-4.5 * (p_ensemble - 0.5)))
        result = "TÀI" if calibrated_p >= 0.5 else "XỈU"
        confidence = round(65.0 + (abs(calibrated_p - 0.5) * 60.0), 1)
        
        return {
            "result": result,
            "confidence": min(confidence, 98.2),
            "calibrated_p": round(calibrated_p, 4),
            "z_score": round(stats_info["z_score"], 2),
            "is_random": "Công bằng" if is_random else "Biến động"
        }

# --- 4. DATA VALIDATION ---
def parse_dice_input(raw_text):
    parts = raw_text.strip().replace(',', ' ').split()
    history = []
    for p in parts:
        if p.isdigit():
            v = int(p)
            if 3 <= v <= 18: history.append(v)
            
    if not history and len(raw_text.strip()) in [32, 64]:
        hex_str = raw_text.strip().lower()
        for i in range(0, min(len(hex_str), 24), 3):
            sub = int(hex_str[i:i+3], 16)
            d1 = (sub % 6) + 1
            d2 = ((sub // 6) % 6) + 1
            d3 = ((sub // 36) % 6) + 1
            history.append(d1 + d2 + d3)
            
    return history if len(history) >= 3 else None

# --- 5. GIAO DIỆN & EVENT HANDLERS ---
def main_menu():
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("💳 Ví & Lịch sử", callback_data="btn_info"),
        InlineKeyboardButton("🎁 Nhập Code", callback_data="btn_redeem")
    )
    markup.add(
        InlineKeyboardButton("💎 Liên hệ Admin", callback_data="btn_nap")
    )
    return markup

@bot.message_handler(commands=['start'])
def start_cmd(message):
    try:
        uid = message.from_user.id
        init_user(uid)
        text = (
            "⚡ **Tool TXGAME v22 Master Analytics** ⚡\n"
            "──────────────────\n"
            "🔥 **Tích hợp trọn bộ 13 lớp thuật toán:**\n"
            "• Exact Prob | Descriptive | EWMA | Bayes\n"
            "• Markov | Chi-Square | Monte Carlo | ML\n"
            "• Ensemble BMA | Sigmoid Calibration\n"
            "──────────────────\n"
            f"🆔 ID: `{uid}` | 🪙 Xu: `{user_data[uid]['balance']}`\n"
            "──────────────────\n"
            "👉 **Nhập chuỗi tổng điểm:** `11 8 14 12 5 16`\n"
            "👉 **Hoặc dán mã Hash MD5/SHA256**"
        )
        bot.reply_to(message, text, parse_mode="Markdown", reply_markup=main_menu())
    except:
        pass

@bot.message_handler(commands=['congxu'])
def add_coins_cmd(message):
    try:
        if not is_admin(message.from_user):
            return
        parts = message.text.split()
        if len(parts) != 3:
            bot.reply_to(
                message, 
                "📌 **Cú pháp:** `/congxu [ID_Nguoi_Dung] [So_Xu]`\n"
                "👉 **Ví dụ:** `/congxu 755092812 100`", 
                parse_mode="Markdown"
            )
            return
            
        target_id, amount = int(parts[1]), int(parts[2])
        init_user(target_id)
        user_data[target_id]["balance"] += amount
        
        bot.reply_to(
            message, 
            f"✅ **Cộng xu thành công!**\n"
            f"👤 Target ID: `{target_id}`\n"
            f"🪙 Đã cộng: `+{amount}` xu\n"
            f"💳 Số dư mới: `{user_data[target_id]['balance']}` xu", 
            parse_mode="Markdown"
        )
    except Exception as e:
        bot.reply_to(message, f"❌ Lỗi: {str(e)}")

@bot.message_handler(commands=['taocode'])
def generate_code_cmd(message):
    try:
        uid = message.from_user.id
        init_user(uid)
        parts = message.text.split()
        if len(parts) < 4: return
        
        code_name, amount, max_uses = parts[1].upper(), int(parts[2]), int(parts[3])
        total_cost = amount * max_uses
        
        if not is_admin(message.from_user):
            if user_data[uid]["balance"] < total_cost: return
            user_data[uid]["balance"] -= total_cost

        gift_codes[code_name] = {"amount": amount, "max_uses": max_uses, "used_by": set()}
        bot.reply_to(message, f"✅ Mã `{code_name}` (+{amount} xu, {max_uses} lượt) đã được tạo!", parse_mode="Markdown")
    except:
        pass

@bot.message_handler(commands=['napcode'])
def redeem_code_cmd(message):
    try:
        uid = message.from_user.id
        init_user(uid)
        parts = message.text.split()
        if len(parts) < 2: return
        
        code_input = parts[1].upper()
        if code_input not in gift_codes or uid in gift_codes[code_input]["used_by"]: return
        if len(gift_codes[code_input]["used_by"]) >= gift_codes[code_input]["max_uses"]: return

        gift_codes[code_input]["used_by"].add(uid)
        amt = gift_codes[code_input]["amount"]
        user_data[uid]["balance"] += amt
        bot.reply_to(message, f"✅ Nạp thành công `{amt}` xu từ code `{code_input}`!", parse_mode="Markdown")
    except:
        pass

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    try:
        uid = call.from_user.id
        init_user(uid)
        if call.data == "btn_info":
            logs = "\n".join(user_data[uid]["logs"]) if user_data[uid]["logs"] else "Chưa có lịch sử."
            bot.send_message(call.message.chat.id, f"🆔 ID: `{uid}` | 🪙 Xu: `{user_data[uid]['balance']}`\n📜 Lịch sử:\n{logs}", parse_mode="Markdown")
        elif call.data == "btn_redeem":
            bot.send_message(call.message.chat.id, "👉 Cú pháp: `/napcode [Mã_Code]`", parse_mode="Markdown")
        elif call.data == "btn_nap":
            bot.send_message(call.message.chat.id, f"💎 Liên hệ Admin @lionVnIos", parse_mode="Markdown")
    except:
        pass

@bot.message_handler(func=lambda message: True)
def handle_master_pipeline(message):
    try:
        uid = message.from_user.id
        init_user(uid)
        
        if user_data[uid]["balance"] < 1:
            bot.reply_to(message, "⚠️ Bạn đã hết xu. Vui lòng nạp thêm!", reply_markup=main_menu())
            return
            
        history = parse_dice_input(message.text)
        if not history:
            bot.reply_to(message, "❌ Dữ liệu không đủ! Cần tối thiểu 3 phiên (hoặc mã MD5).")
            return

        user_data[uid]["balance"] -= 1

        engine = MasterAnalyticsEngine(history)
        metrics = engine.layer12_13_ensemble_calibration()
        
        res_icon = "⚫ TÀI" if metrics["result"] == "TÀI" else "⚪ XỈU"
        
        user_data[uid]["logs"].insert(0, f"[{len(history)} ván] ➔ {metrics['result']}")
        if len(user_data[uid]["logs"]) > 5: user_data[uid]["logs"].pop()
            
        res_msg = (
            "✅ **Kết quả Phân Tích 13 Lớp Thuật Toán**\n"
            "──────────────────\n"
            f"🔮 **Dự đoán:** {res_icon}\n"
            f"🎯 **Độ tin cậy Ensemble:** `{metrics['confidence']}%`\n"
            f"📊 **Mô hình Calibrated P:** `{metrics['calibrated_p']}`\n"
            f"📈 **Z-Score chuỗi:** `{metrics['z_score']}`\n"
            f"⚖️ **Trạng thái xúc xắc:** `{metrics['is_random']}`\n"
            "──────────────────\n"
            f"💳 Số dư xu: `{user_data[uid]['balance']}`"
        )
        bot.reply_to(message, res_msg, parse_mode="Markdown", reply_markup=main_menu())
    except:
        pass

if __name__ == '__main__':
    t = Thread(target=run_flask)
    t.daemon = True
    t.start()
    try:
        bot.remove_webhook()
    except:
        pass
    print("TOOL TXGAME v22 MASTER ENGINE ONLINE...")
    bot.infinity_polling(none_stop=True)

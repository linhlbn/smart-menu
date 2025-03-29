import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta, date
import io
import os
from dotenv import load_dotenv
import time
from openai import OpenAI
import copy
import math

load_dotenv()

st.set_page_config(
    page_title="THClinic - Thực đơn Thông Minh",
    page_icon="THClinic.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_password():
    if "password_correct" not in st.session_state:
        pwd_col1, pwd_col2, pwd_col3 = st.columns([1,1.5,1])
        with pwd_col2:
            try: st.image("THClinic.png", width=150)
            except Exception: st.warning("Logo 'THClinic.png' not found.")
            st.subheader("🔒 Đăng nhập Hệ thống")
            password_input = st.text_input("Nhập mật khẩu", type="password", key="password_input", label_visibility="collapsed", placeholder="Mật khẩu")
            stored_password = os.getenv("smart_food")
            if st.button("Xác nhận", use_container_width=True):
                if password_input == stored_password:
                    st.session_state["password_correct"] = True; st.rerun()
                else:
                    st.error("Mật khẩu không đúng!"); st.session_state["password_correct"] = False
        return False
    elif not st.session_state["password_correct"]:
        pwd_col1, pwd_col2, pwd_col3 = st.columns([1,1.5,1])
        with pwd_col2:
            try: st.image("THClinic.png", width=150)
            except Exception: st.warning("Logo 'THClinic.png' not found.")
            st.subheader("🔒 Đăng nhập Hệ thống")
            password_input = st.text_input("Nhập mật khẩu", type="password", key="password_input_retry", label_visibility="collapsed", placeholder="Mật khẩu")
            stored_password = os.getenv("smart_food")
            if st.button("Xác nhận", use_container_width=True):
                 if password_input == stored_password:
                    st.session_state["password_correct"] = True; st.rerun()
                 else:
                    st.error("Mật khẩu không đúng!"); st.session_state["password_correct"] = False
        return False
    else:
        return True

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    OPENAI_ENABLED = True
    PROMPT_TEMPLATE = os.getenv("PROMPT")
    if not PROMPT_TEMPLATE:
        st.error("Lỗi: Không tìm thấy PROMPT template trong file .env. Vui lòng kiểm tra cấu hình.", icon="❌")
        OPENAI_ENABLED = False
except Exception as e:
    st.error(f"Lỗi khởi tạo OpenAI Client hoặc đọc .env: {e}. Tính năng AI sẽ bị vô hiệu hóa.", icon="❌")
    OPENAI_ENABLED = False
    client = None
    PROMPT_TEMPLATE = None

if OPENAI_ENABLED and not check_password():
    st.stop()

st.markdown("""
<style>
    body, .stApp { background-color: #f8f9fa !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; }
    .main { background-color: #f8f9fa; padding: 10px 25px; border-radius: 0px; box-shadow: none; }
    h1 { color: #005A9C; font-weight: 600; margin-bottom: 20px; text-align: center; padding-top: 10px;}
    h2 { color: #005A9C; font-weight: 500; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; margin-top: 40px; margin-bottom: 25px; }
    h3 { color: #333; font-weight: 500; margin-bottom: 15px; }
    .stTextInput input, .stNumberInput input, .stTextArea textarea { border: 1px solid #ced4da; border-radius: 5px; padding: 10px; font-size: 15px; transition: border-color 0.2s, box-shadow 0.2s; background-color: #fff; }
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus { border-color: #005A9C; box-shadow: 0 0 0 2px rgba(0, 90, 156, 0.2); background-color: #fff;}
    .stButton>button { background-color: #005A9C; color: white; border-radius: 5px; padding: 10px 20px; font-size: 15px; font-weight: 500; border: none; cursor: pointer; transition: background-color 0.3s, transform 0.1s; width: auto; }
    .stButton>button[kind="primary"] { font-weight: 600; background-color: #004170; }
    .stButton>button[kind="primary"]:hover { background-color: #002c4a; }
    .stButton>button:hover { background-color: #004170; transform: translateY(-1px); }
    .stButton>button:active { transform: translateY(0px); }
    .stButton>button:disabled { background-color: #cccccc; color: #666666; cursor: not-allowed; transform: none; }
    .small-button button { padding: 3px 10px !important; font-size: 12px !important; line-height: 1.3 !important; margin-left: 5px !important; background-color: #e74c3c !important; border-color: #e74c3c !important; border-radius: 4px !important; }
    .small-button button:hover { background-color: #c0392b !important; border-color: #c0392b !important; }
    .stDivider { border-color: #e0e0e0 !important; margin-top: 15px; margin-bottom: 25px; }
    .stSidebar { background-color: #ffffff; border-right: 1px solid #e0e0e0; padding-top: 20px;}
    .stSidebar h2, .stSidebar h3 { color: #005A9C; border-bottom: none; padding-bottom: 5px; margin-bottom: 10px; }
    .stToast { font-size: 15px; }
    .stCaption { color: #555; font-size: 13px; margin-bottom: 15px; }
    .stExpander { border: none; border-radius: 8px; background-color: #f0f2f6; margin-bottom: 20px; }
    .stExpander header { font-weight: 500; color: #005A9C; background-color: #e9ecef; padding: 10px 15px !important; border-radius: 8px 8px 0 0; border-bottom: 1px solid #dee2e6;}
    .stExpander div[data-testid="stExpanderDetails"] { background-color: #fff; border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 8px 8px; padding: 15px;}
    .preference-list-item { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; padding: 6px 0; border-bottom: 1px solid #eee; }
    .preference-list-item span { font-size: 14px; margin-right: 10px; flex-grow: 1; }
    .preference-list-item .small-button { flex-shrink: 0; }
    .stCheckbox { margin-bottom: 20px; }
    .stSlider { margin-bottom: 15px;}
    .advanced-options { border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin-top: 15px; background-color: #ffffff;}
</style>
""", unsafe_allow_html=True)

conn = sqlite3.connect('user_data.db'); c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, age INTEGER, height REAL, weight REAL, ideal_weight REAL)''')
c.execute('''CREATE TABLE IF NOT EXISTS foods (name TEXT PRIMARY KEY, kcal REAL, protein REAL, carb REAL, fat REAL)''')

def reset_database():
    try:
        c.execute("DROP TABLE IF EXISTS users")
        c.execute('''CREATE TABLE users (id INTEGER PRIMARY KEY, age INTEGER, height REAL, weight REAL, ideal_weight REAL)''')
        c.execute("DROP TABLE IF EXISTS foods")
        c.execute('''CREATE TABLE foods (name TEXT PRIMARY KEY, kcal REAL, protein REAL, carb REAL, fat REAL)''')
        conn.commit()
        st.success("Đã reset database thành công!", icon="✅")
        keys_to_clear = ['bmi', 'ideal_w', 'sustainable_daily_calorie_change', 'total_days_needed', 'priority_foods', 'priority_sports', 'avoid_foods', 'generated_plan_content', 'advanced_customization_enabled', 'adv_carb_perc', 'adv_pro_perc', 'adv_lip_perc', 'health_conditions']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.rerun()
    except Exception as e: st.error(f"Lỗi khi reset database: {e}", icon="❌")

def import_csv(file):
    try:
        df = pd.read_csv(file)
        required_columns = ['name', 'kcal', 'protein', 'carb', 'fat']
        if not all(col in df.columns for col in required_columns):
            st.error("File CSV thiếu cột cần thiết (name, kcal, protein, carb, fat).", icon="❌"); return False
        df.to_sql('foods', conn, if_exists='append', index=False)
        conn.commit(); st.success("Import dữ liệu từ CSV thành công!", icon="✅"); return True
    except sqlite3.IntegrityError:
         st.warning("Một số món ăn đã tồn tại trong database và không được thêm lại.", icon="⚠️"); conn.commit(); return True
    except Exception as e: st.error(f"Lỗi khi import CSV: {e}", icon="❌"); return False

initial_foods = [ ("Cơm trắng (100g)", 130, 2.7, 28, 0.3), ("Ức gà (100g)", 165, 31, 0, 3.6), ("Chuối (1 quả, 120g)", 90, 1, 23, 0.3), ("Sữa tươi (200ml)", 120, 6, 9, 6), ("Heo nạc thăn (100g)", 143, 26, 0, 3.5), ("Bánh mì trắng (1 lát)", 79, 2.7, 15, 1), ("Trứng gà luộc (1 quả)", 78, 6, 0.6, 5), ("Gạo lứt (100g nấu chín)", 123, 2.5, 25, 1), ("Cá hồi nướng (100g)", 208, 20, 0, 13), ("Hạt óc chó (30g)", 196, 4.5, 3.9, 19.5), ("Bơ (trái nhỏ, ~150g)", 240, 3, 13, 22) ]
try: c.executemany("INSERT OR IGNORE INTO foods (name, kcal, protein, carb, fat) VALUES (?, ?, ?, ?, ?)", initial_foods); conn.commit()
except Exception: pass

def calculate_bmi(height, weight):
    if height <= 0: return 0
    return weight / ((height / 100) ** 2)

def ideal_weight(height, age):
    if height <= 0: return 0
    bmi = 22 if 18 <= age <= 65 else 20 if age < 18 else 24
    return bmi * ((height / 100) ** 2)

def calculate_sustainable_goal(current_weight, ideal_weight):
    max_loss_kg_per_week = 0.7; max_gain_kg_per_week = 0.5; kcal_per_kg = 7700
    weight_diff = ideal_weight - current_weight
    sustainable_daily_change = 0; total_days_needed = 0
    if abs(weight_diff) < 0.2: return 0, 0
    if weight_diff < 0:
        weeks = abs(weight_diff) / max_loss_kg_per_week; total_days_needed = math.ceil(weeks * 7)
        sustainable_daily_change = - (max_loss_kg_per_week * kcal_per_kg / 7)
    elif weight_diff > 0:
        weeks = weight_diff / max_gain_kg_per_week; total_days_needed = math.ceil(weeks * 7)
        sustainable_daily_change = (max_gain_kg_per_week * kcal_per_kg / 7)
    return sustainable_daily_change, total_days_needed

def get_macro_percentages(age):
     st.session_state.default_macros = {'carb': 45, 'pro': 25, 'lip': 30}
     return 45, 25, 30

@st.cache_data(show_spinner=False, ttl=3600)
def validate_input_with_openai(_client, item, item_type):
    if not _client: return False
    if not item or not item.strip(): return False

    prompt_text = ""; expected_answer = "CÓ"
    if item_type == 'food': prompt_text = f"'{item}' có phải là tên một loại thực phẩm, món ăn, hoặc đồ uống không? Chỉ trả lời 'CÓ' hoặc 'KHÔNG'."
    elif item_type == 'sport': prompt_text = f"'{item}' có phải là tên một môn thể thao hoặc hoạt động thể chất không? Chỉ trả lời 'CÓ' hoặc 'KHÔNG'."
    elif item_type == 'condition': prompt_text = f"Nội dung sau có mô tả một tình trạng sức khỏe, bệnh lý, dị ứng, hoặc lưu ý dinh dưỡng hợp lệ không: '{item}'? Chỉ trả lời 'CÓ' hoặc 'KHÔNG'."
    else: return False

    try:
        response = _client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[ {"role": "system", "content": "Chỉ trả lời bằng 'CÓ' hoặc 'KHÔNG'."}, {"role": "user", "content": prompt_text} ],
            temperature=0.0, max_tokens=5
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer == expected_answer
    except Exception as e:
        st.error(f"Lỗi khi kiểm tra '{item_type}' bằng AI: {e}", icon="⚠️")
        return False

def generate_plan_stream_with_openai(target_calories, days, age, priority_foods_list, priority_sports_list, avoid_foods_list, advanced_macros=None, health_conditions=None):
    if not client: raise ValueError("OpenAI client not available")
    if not PROMPT_TEMPLATE: raise ValueError("PROMPT template not loaded from .env")

    if advanced_macros:
        carb_perc, protein_perc, fat_perc = advanced_macros['carb'], advanced_macros['pro'], advanced_macros['lip']
        macro_source = "nâng cao"
    else:
        carb_perc, protein_perc, fat_perc = get_macro_percentages(age); macro_source = "mặc định"

    goal_type = "tăng" if target_calories > 0 else "giảm" if target_calories < 0 else "duy trì"

    priority_foods_string = f"\n- Món ăn ưu tiên: Cố gắng đưa vào nếu phù hợp: {', '.join([f'{f}' for f in priority_foods_list])}." if priority_foods_list else ""
    priority_sports_string = f"\n- Thể thao ưu tiên: Chỉ đề xuất từ: {', '.join([f'{s}' for s in priority_sports_list])}. Tính kcal tiêu hao." if priority_sports_list else "\n- Thể thao ưu tiên: Đề xuất 3 môn phù hợp."
    avoid_foods_string = f"\n- Món ăn cần tránh: Tuyệt đối KHÔNG dùng: {', '.join([f'{f}' for f in avoid_foods_list])}." if avoid_foods_list else ""
    health_notes = f"\n- Lưu ý sức khỏe: {health_conditions}" if health_conditions else ""

    format_dict = {
        'target_calories': target_calories, 'goal_type': goal_type, 'days': days,
        'macro_source': macro_source, 'carb_perc': carb_perc, 'protein_perc': protein_perc, 'fat_perc': fat_perc,
        'priority_foods_string': priority_foods_string, 'avoid_foods_string': avoid_foods_string,
        'priority_sports_string': priority_sports_string, 'health_notes': health_notes,
        'day_2_template': '', 'day_3_template': '', 'day_4_template': '',
        'day_5_template': '', 'day_6_template': '', 'day_7_template': ''
    }

    day_template = """
Ngày {day_num} ({date}):
**Thực đơn:**
* **Sáng:** [Điền...]
* **Trưa:** [Điền...]
* **Chiều:** [Điền...]
* **Tối:** [Điền...]
* **Tổng cộng:** [Điền...]
**Hoạt động thể chất (chọn 1 phù hợp từ ưu tiên nếu có):**
1. **Tên HĐ 1:** [Điền...]
2. **Tên HĐ 2:** [Điền...]
3. **Tên HĐ 3:** [Điền...]
---"""

    today = date.today()
    for i in range(1, days + 1):
        current_date_str = (today + timedelta(days=i)).strftime('%Y-%m-%d')
        format_dict[f'date_{i}'] = current_date_str
        if i > 1:
            format_dict[f'day_{i}_template'] = day_template.format(day_num=i, date=current_date_str)

    final_prompt = PROMPT_TEMPLATE.format(**format_dict)
    if final_prompt.endswith("\n---"): final_prompt = final_prompt[:-4]

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini", messages=[ {"role": "system", "content": "Bạn là AI dinh dưỡng THClinic tiếng Việt. Tuân thủ nghiêm ngặt yêu cầu, định dạng. Hoàn thành đầy đủ mọi ngày."}, {"role": "user", "content": final_prompt} ],
            temperature=0.5, max_tokens=2500 + (days * 500), stream=True,
        )
        return stream
    except Exception as e:
        st.error(f"Lỗi khi gọi OpenAI API: {e}", icon="❌"); raise e

def create_download_file(plan_content, user_info, format="md"):
    adv_macros_str = f" (Nâng cao: {user_info['adv_carb_perc']}%C / {user_info['adv_pro_perc']}%P / {user_info['adv_lip_perc']}%L)" if user_info.get('advanced_macros_enabled') else ""
    health_cond_str = f"\n- Lưu ý sức khỏe: {user_info['health_conditions']}" if user_info.get('health_conditions') else ""
    macro_disp = f"{user_info.get('macro_info', 'Mặc định')}{adv_macros_str}"

    intro_lines = [
        f"Kế hoạch Ăn uống ({user_info['days']} ngày) - THClinic", "========================================", "",
        "Thông tin:", f"- Tuổi: {user_info['age']}", f"- Cao: {user_info['height']} cm", f"- Nặng: {user_info['weight']} kg",
        f"- Cân nặng lý tưởng: {user_info['ideal_weight']:.2f} kg", f"- Mục tiêu calo hàng ngày: {user_info['sustainable_daily_calorie_change']:+.0f} kcal",
        f"- Tỷ lệ dinh dưỡng mục tiêu: {macro_disp}", health_cond_str, "",
        f"- Món ăn ưu tiên: {', '.join(user_info['priority_foods']) if user_info['priority_foods'] else 'Không có'}",
        f"- Thể thao ưu tiên: {', '.join(user_info['priority_sports']) if user_info['priority_sports'] else 'Không có'}",
        f"- Món ăn cần tránh: {', '.join(user_info['avoid_foods']) if user_info['avoid_foods'] else 'Không có'}",
        "", "Chi tiết kế hoạch tuần:", "------------------", ""
    ]
    intro = "\n".join(filter(None, intro_lines))

    if format == "md": content = intro + plan_content; filename = f"THClinic_MealPlan_{user_info['days']}days_{date.today()}.md"
    else: content = intro + plan_content.replace("## ", "").replace("### ", "").replace("**", "").replace("* ", "- ").replace("---", "\n------------------\n"); filename = f"THClinic_MealPlan_{user_info['days']}days_{date.today()}.txt"
    return content, filename

st.session_state.setdefault('request_timestamps', []); st.session_state.setdefault('priority_foods', [])
st.session_state.setdefault('priority_sports', []); st.session_state.setdefault('avoid_foods', [])
st.session_state.setdefault('generated_plan_content', None); st.session_state.setdefault('show_plan_clicked', False)
st.session_state.setdefault('min_days_suggestion_raw', 1); st.session_state.setdefault('sustainable_daily_calorie_change', 0); st.session_state.setdefault('total_days_needed', 0)
st.session_state.setdefault('advanced_customization_enabled', False)
st.session_state.setdefault('adv_carb_perc', 45); st.session_state.setdefault('adv_pro_perc', 25); st.session_state.setdefault('adv_lip_perc', 30)
st.session_state.setdefault('health_conditions', ""); st.session_state.setdefault('default_macros', {'carb': 45, 'pro': 25, 'lip': 30})


with st.sidebar:
    try: st.image("THClinic.png", width=150)
    except FileNotFoundError: st.warning("'THClinic.png' not found.")
    except Exception as e: st.warning(f"Could not load logo: {e}")
    st.header("Hướng dẫn")
    st.markdown("1. **Nhập thông tin:** Tuổi, chiều cao, cân nặng.\n"
                "2. **Tính toán:** Xem BMI, cân nặng lý tưởng & mục tiêu calo bền vững.\n"
                "3. **(Tùy chọn) Tùy chỉnh:** Thêm ưu tiên/tránh, hoặc bật tùy chỉnh nâng cao (macros, bệnh lý).\n"
                "4. **Lên kế hoạch:** Chọn số ngày (1-7) & tạo kế hoạch tuần.\n"
                "5. **Xem & Tải:** Xem và tải kế hoạch.")
    st.divider()
    st.header("Quản trị")
    if st.button("Reset Database Thực Phẩm", key="sidebar_reset", use_container_width=True):
         if st.checkbox("Xác nhận xóa toàn bộ dữ liệu thực phẩm?", key="confirm_reset"): reset_database()
         else: st.warning("Vui lòng xác nhận để reset.")

st.title("🍎 THClinic - Thực đơn Thông Minh")
st.write("---")

with st.expander("Nhập dữ liệu thực phẩm từ file CSV (Tùy chọn)"):
    with st.container():
         uploaded_file = st.file_uploader("Chọn file .csv", type=['csv'], help="Cột: name, kcal, protein, carb, fat.", label_visibility="collapsed")
         if uploaded_file is not None: import_csv(uploaded_file)

with st.container():
    st.header("1. Thông tin cá nhân & Mục tiêu")
    col1, col2, col3 = st.columns(3)
    with col1: age = st.number_input("Tuổi", 1, 120, st.session_state.get('user_age', 21), key="age_input")
    with col2: height = st.number_input("Chiều cao (cm)", 50.0, 250.0, st.session_state.get('user_height', 171.0), step=0.5, format="%.1f", key="height_input")
    with col3: weight = st.number_input("Cân nặng (kg)", 20.0, 200.0, st.session_state.get('user_weight', 63.0), step=0.1, format="%.1f", key="weight_input")

    st.write("")
    calc_col1, calc_col2 = st.columns([3, 1])
    with calc_col2:
        if st.button("Tính toán Mục tiêu", type="primary", key="calc_bmi_button", use_container_width=True):
            if height > 0 and weight > 0:
                bmi = calculate_bmi(height, weight)
                ideal_w = ideal_weight(height, age)
                sustainable_change, total_days = calculate_sustainable_goal(weight, ideal_w)
                st.session_state['bmi'] = bmi; st.session_state['ideal_w'] = ideal_w
                st.session_state['sustainable_daily_calorie_change'] = sustainable_change
                st.session_state['total_days_needed'] = total_days
                st.session_state['user_age'] = age; st.session_state['user_height'] = height; st.session_state['user_weight'] = weight
                st.session_state['calc_age'] = age; st.session_state['calc_height'] = height; st.session_state['calc_weight'] = weight
                st.session_state.generated_plan_content = None; get_macro_percentages(age)
                st.toast(f"Đã tính toán mục tiêu bền vững!", icon="🎯"); st.rerun()
            else: st.error("Chiều cao và cân nặng phải lớn hơn 0.", icon="❌")

    if 'bmi' in st.session_state and 'ideal_w' in st.session_state:
         st.info(f"**Kết quả:** BMI: **{st.session_state['bmi']:.2f}** | Cân nặng lý tưởng: **{st.session_state['ideal_w']:.2f} kg**", icon="📊")
         sustainable_change = st.session_state.get('sustainable_daily_calorie_change', 0)
         total_days = st.session_state.get('total_days_needed', 0)
         if sustainable_change != 0 and total_days > 0:
             goal_verb = "Giảm" if sustainable_change < 0 else "Tăng"
             st.success(f"**Mục tiêu bền vững:** {goal_verb} cân bằng cách điều chỉnh **{sustainable_change:+.0f} kcal/ngày**. Ước tính cần **~{total_days} ngày**.", icon="🎯")
         elif 'bmi' in st.session_state: st.success("**Mục tiêu bền vững:** Duy trì cân nặng hiện tại.", icon="✅")
    st.write("")

def render_preference_list(list_key, title):
    current_list = st.session_state.get(list_key, [])
    st.markdown(f"**{title}:**")
    if not current_list: st.caption(f"Chưa có."); return

    items_to_delete = {}
    display_area = st.container()

    with display_area:
        list_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 5px;'>"
        for index, item in enumerate(current_list):
            delete_key = f"del_{list_key}_{index}_{item.replace(' ', '_')}"
            items_to_delete[item] = delete_key
            list_html += f"<div style='display: flex; align-items: center; background-color: #e9ecef; padding: 3px 8px; border-radius: 5px; font-size: 14px;'><span>{item}</span></div>"
        list_html += "</div>"
        st.markdown(list_html, unsafe_allow_html=True)

        clicked_item_to_delete = None
        st.write("")
        button_cols = st.columns(len(current_list) if current_list else 1)
        processed_items_btn = set()
        col_idx_btn = 0

        for item in current_list:
             if item not in processed_items_btn:
                 with button_cols[col_idx_btn % len(button_cols)]:
                     st.markdown('<div class="small-button" style="margin-top: -5px;">', unsafe_allow_html=True)
                     delete_key = items_to_delete[item]
                     if st.button(f"Xóa '{item}'", key=delete_key): clicked_item_to_delete = item
                     st.markdown('</div>', unsafe_allow_html=True)
                 processed_items_btn.add(item)
                 col_idx_btn += 1

    if clicked_item_to_delete:
        if clicked_item_to_delete in st.session_state[list_key]:
            st.session_state[list_key].remove(clicked_item_to_delete)
        st.rerun()


with st.container():
    st.header("2. Tùy chỉnh Kế hoạch")
    st.caption("Thêm các món ăn, môn thể thao bạn yêu thích hoặc các món bạn muốn tránh (AI sẽ kiểm tra tính hợp lệ). Bật tùy chọn nâng cao để điều chỉnh macros và thêm bệnh lý.")
    st.write("")
    col_pref_input1, col_pref_input2, col_pref_input3 = st.columns(3)

    with col_pref_input1:
        st.markdown("**Món ăn ưu tiên**")
        new_food = st.text_input("Nhập món ăn:", key="new_priority_foods_input", label_visibility="collapsed", placeholder="Ví dụ: Ức gà...")
        if st.button("➕ Thêm món ăn", key="add_priority_foods_button", use_container_width=True, disabled=not OPENAI_ENABLED):
            if OPENAI_ENABLED:
                with st.spinner("Đang kiểm tra..."): valid = validate_input_with_openai(client, new_food, 'food')
                if valid and new_food and new_food not in st.session_state.priority_foods: st.session_state.priority_foods.append(new_food); st.rerun()
                elif valid and new_food: st.toast("Món ăn đã có.", icon="⚠️")
                elif new_food: st.error(f"'{new_food}' không giống món ăn hợp lệ.", icon="❌")
            else: st.warning("Kiểm tra AI không khả dụng.")
        render_preference_list('priority_foods', "Món ăn ưu tiên")

    with col_pref_input2:
        st.markdown("**Thể thao ưu tiên**")
        new_sport = st.text_input("Nhập môn thể thao:", key="new_priority_sports_input", label_visibility="collapsed", placeholder="Ví dụ: Chạy bộ...")
        if st.button("➕ Thêm thể thao", key="add_priority_sports_button", use_container_width=True, disabled=not OPENAI_ENABLED):
            if OPENAI_ENABLED:
                with st.spinner("Đang kiểm tra..."): valid = validate_input_with_openai(client, new_sport, 'sport')
                if valid and new_sport and new_sport not in st.session_state.priority_sports: st.session_state.priority_sports.append(new_sport); st.rerun()
                elif valid and new_sport: st.toast("Môn thể thao đã có.", icon="⚠️")
                elif new_sport: st.error(f"'{new_sport}' không giống môn thể thao hợp lệ.", icon="❌")
            else: st.warning("Kiểm tra AI không khả dụng.")
        render_preference_list('priority_sports', "Môn thể thao ưu tiên")

    with col_pref_input3:
        st.markdown("**Món ăn cần tránh**")
        new_avoid = st.text_input("Nhập món cần tránh:", key="new_avoid_foods_input", label_visibility="collapsed", placeholder="Ví dụ: Đồ chiên rán...")
        if st.button("➕ Thêm món tránh", key="add_avoid_foods_button", use_container_width=True, disabled=not OPENAI_ENABLED):
             if OPENAI_ENABLED:
                 with st.spinner("Đang kiểm tra..."): valid = validate_input_with_openai(client, new_avoid, 'food')
                 if valid and new_avoid and new_avoid not in st.session_state.avoid_foods: st.session_state.avoid_foods.append(new_avoid); st.rerun()
                 elif valid and new_avoid: st.toast("Món ăn đã có trong danh sách tránh.", icon="⚠️")
                 elif new_avoid: st.error(f"'{new_avoid}' không giống món ăn hợp lệ.", icon="❌")
             else: st.warning("Kiểm tra AI không khả dụng.")
        render_preference_list('avoid_foods', "Món ăn cần tránh")

    st.divider()
    st.checkbox("Tùy chỉnh Nâng cao (Macros, Bệnh lý)", key="advanced_customization_enabled", value=st.session_state.get('advanced_customization_enabled', False))

    if st.session_state.get("advanced_customization_enabled"):
        with st.container(border=True):
             st.markdown("#### Tùy chỉnh Nâng cao")
             adv_macro_col1, adv_macro_col2 = st.columns(2)
             with adv_macro_col1:
                 st.markdown("**Tỷ lệ dinh dưỡng (%)**")
                 carb_val = st.slider("Carbohydrate (%)", 0, 100, st.session_state.adv_carb_perc, key="adv_carb_slider")
                 max_prot_val = 100 - carb_val
                 # Ensure protein value doesn't exceed max based on current carb value
                 current_prot_val = st.session_state.adv_pro_perc if st.session_state.adv_pro_perc <= max_prot_val else max_prot_val
                 prot_val = st.slider("Protein (%)", 0, max_prot_val, current_prot_val, key="adv_pro_slider")
                 lipid_calc = 100 - carb_val - prot_val
                 lip_val = max(0, lipid_calc)
                 st.session_state.adv_carb_perc = carb_val; st.session_state.adv_pro_perc = prot_val; st.session_state.adv_lip_perc = lip_val
                 st.info(f"Tính toán: Lipid: **{lip_val}%** (Tổng: {carb_val + prot_val + lip_val}%)")
                 if carb_val + prot_val + lip_val != 100: st.warning("Tổng tỷ lệ chưa bằng 100%. Lipid được tự động điều chỉnh.")

             with adv_macro_col2:
                  st.markdown("**Bệnh lý / Lưu ý sức khỏe**")
                  st.session_state.health_conditions = st.text_area("Mô tả ngắn gọn:", key="health_conditions_input", value=st.session_state.get('health_conditions', ""), height=150, label_visibility="collapsed", placeholder="Ví dụ: Tiểu đường type 2, dị ứng lactose...")
                  st.caption("AI sẽ kiểm tra tính hợp lệ khi tạo kế hoạch.")
    st.write("")


if 'bmi' in st.session_state and 'ideal_w' in st.session_state:
    current_age = st.session_state.get('user_age', 30)
    current_height = st.session_state.get('user_height', 167.0)
    current_weight = st.session_state.get('user_weight', 64.0)

    inputs_changed = False
    if 'calc_age' in st.session_state:
        inputs_changed = (st.session_state.get('calc_age') != current_age or
                          st.session_state.get('calc_height') != current_height or
                          st.session_state.get('calc_weight') != current_weight)

    if inputs_changed:
         st.warning("Thông tin cá nhân đã thay đổi. Nhấn 'Tính toán Mục tiêu' lại để cập nhật.", icon="⚠️")
         st.session_state.generated_plan_content = None
    else:
        with st.container():
            st.header("3. Hỗ trợ lên Kế hoạch Tuần")
            st.caption("Chọn số ngày (1-7) bạn muốn AI lên kế hoạch chi tiết cho tuần tới. Kế hoạch này dựa trên mục tiêu calo bền vững và các tùy chỉnh của bạn. Bạn có thể tạo kế hoạch mới cho các tuần tiếp theo.")

            days_plan_col, _ = st.columns([1, 3])
            with days_plan_col:
                default_days = 7; days_value = min(max(1, st.session_state.get('days', default_days)), 7)
                days = st.number_input( f"Số ngày kế hoạch (1-7):", min_value=1, max_value=7, value=days_value, key="days_input", label_visibility="collapsed")
                st.session_state['days'] = days

            sustainable_change = st.session_state.get('sustainable_daily_calorie_change', None)
            show_plan_disabled = sustainable_change is None or not OPENAI_ENABLED

            st.write("---")
            health_condition_valid = True
            health_condition_validation_message = ""
            conditions_text = st.session_state.get('health_conditions', '').strip()
            if st.session_state.get("advanced_customization_enabled") and conditions_text and OPENAI_ENABLED:
                 with st.spinner("Kiểm tra thông tin sức khỏe..."):
                      health_condition_valid = validate_input_with_openai(client, conditions_text, 'condition')
                      if not health_condition_valid:
                          health_condition_validation_message = "Mô tả bệnh lý/lưu ý sức khỏe không hợp lệ. Vui lòng kiểm tra lại ở Mục 2."

            plan_button_col, _ = st.columns([1, 2])
            with plan_button_col:
                 clicked = st.button(f"Tạo Kế hoạch {days} ngày (AI ✨)", disabled=(show_plan_disabled or not health_condition_valid), type="primary", key="show_plan_button", use_container_width=True)
                 if clicked:
                      if sustainable_change is None: st.error("Vui lòng nhấn 'Tính toán Mục tiêu' trước.", icon="🎯")
                      elif not OPENAI_ENABLED: st.error("Chức năng AI không khả dụng.")
                      elif not health_condition_valid: st.error(health_condition_validation_message, icon="❌")
                      else:
                           st.session_state.show_plan_clicked = True
                           st.session_state.generated_plan_content = None

            if not health_condition_valid and health_condition_validation_message:
                 st.error(health_condition_validation_message, icon="❌")

            st.write("---")
            st.subheader("Kế hoạch chi tiết từ AI")
            plan_placeholder = st.container()
            if st.session_state.show_plan_clicked:
                st.session_state.show_plan_clicked = False
                current_time = time.time()
                st.session_state.request_timestamps = [ts for ts in st.session_state.request_timestamps if current_time - ts < 60]
                if len(st.session_state.request_timestamps) >= 5 and OPENAI_ENABLED:
                    plan_placeholder.error("Quá nhiều yêu cầu. Vui lòng đợi 1 phút.", icon="⏳")
                elif OPENAI_ENABLED and 'sustainable_daily_calorie_change' in st.session_state and health_condition_valid:
                    st.session_state.request_timestamps.append(current_time)
                    try:
                        advanced_macros_data = None; health_cond_data = None
                        if st.session_state.get("advanced_customization_enabled"):
                             advanced_macros_data = { 'carb': st.session_state.adv_carb_perc, 'pro': st.session_state.adv_pro_perc, 'lip': st.session_state.adv_lip_perc }
                             health_cond_data = st.session_state.get('health_conditions').strip() if st.session_state.get('health_conditions') else None

                        stream = generate_plan_stream_with_openai(
                            target_calories=st.session_state['sustainable_daily_calorie_change'], days=st.session_state['days'], age=st.session_state['user_age'],
                            priority_foods_list=st.session_state.get('priority_foods', []), priority_sports_list=st.session_state.get('priority_sports', []), avoid_foods_list=st.session_state.get('avoid_foods', []),
                            advanced_macros=advanced_macros_data, health_conditions=health_cond_data
                        )
                        with plan_placeholder:
                             full_response = st.write_stream(stream)
                             st.session_state.generated_plan_content = full_response
                    except Exception as e:
                         plan_placeholder.error(f"Lỗi tạo kế hoạch từ AI: {e}", icon="❌"); st.session_state.generated_plan_content = None
                elif not OPENAI_ENABLED: plan_placeholder.error("Chức năng AI hiện không khả dụng.", icon="⚙️")
                elif not health_condition_valid: pass
                else: plan_placeholder.warning("Chưa có mục tiêu calo bền vững.", icon="⚠️"); st.session_state.generated_plan_content = None

            if st.session_state.get('generated_plan_content'):
                 st.write("---")
                 dl_col1, dl_col2 = st.columns([3,1])
                 with dl_col2:
                    format_choice = st.radio("Tải xuống:", ["Markdown (.md)", "Text (.txt)"], key="download_format", horizontal=True, label_visibility="collapsed")
                    file_format = "md" if format_choice == "Markdown (.md)" else "txt"

                    macro_disp = "Mặc định"
                    default_macros = st.session_state.get('default_macros', {})
                    if st.session_state.get("advanced_customization_enabled"): macro_disp = f"Nâng cao ({st.session_state.adv_carb_perc}%C / {st.session_state.adv_pro_perc}%P / {st.session_state.adv_lip_perc}%L)"
                    elif default_macros: macro_disp = f"Mặc định ({default_macros.get('carb', '?')}%C / {default_macros.get('pro', '?')}%P / {default_macros.get('lip', '?')}%L)"

                    user_info_for_download = {
                        'age': st.session_state.get('user_age','N/A'), 'height': st.session_state.get('user_height','N/A'),
                        'weight': st.session_state.get('user_weight','N/A'), 'ideal_weight': st.session_state.get('ideal_w', 0),
                        'days': st.session_state.get('days',0), 'sustainable_daily_calorie_change': st.session_state.get('sustainable_daily_calorie_change', 0),
                        'priority_foods': st.session_state.get('priority_foods', []), 'priority_sports': st.session_state.get('priority_sports', []), 'avoid_foods': st.session_state.get('avoid_foods', []),
                        'advanced_macros_enabled': st.session_state.get("advanced_customization_enabled", False),
                        'adv_carb_perc': st.session_state.get('adv_carb_perc'), 'adv_pro_perc': st.session_state.get('adv_pro_perc'), 'adv_lip_perc': st.session_state.get('adv_lip_perc'),
                        'health_conditions': st.session_state.get('health_conditions'), 'macro_info': macro_disp
                    }
                    try:
                        content, filename = create_download_file(st.session_state.generated_plan_content, user_info_for_download, file_format)
                        st.download_button( label=f"💾 Tải xuống ({file_format.upper()})", data=content.encode('utf-8'), file_name=filename, mime=f"text/{file_format}", key="download_button", use_container_width=True)
                    except Exception as e: st.error(f"Lỗi tạo file tải xuống: {e}", icon="❌")

# conn.close()
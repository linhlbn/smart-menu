# app_saas.py
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random
import io
import time
from hashlib import sha256
import streamlit.components.v1 as components
from typing import Optional

# Thiết lập giao diện hiện đại, thân thiện với URL linh hoạt
st.set_page_config(
    page_title="Ứng dụng Thực đơn Thông Minh (SaaS)",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS để nâng cấp giao diện với Light Mode
st.markdown("""
    <style>
    body, .stApp {
        background-color: #ffffff !important;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stNumberInput>input, .stRadio>label, .stTextInput>input {
        font-size: 16px;
        color: #333;
        background-color: #f8f9fa;
        border: 1px solid #ced4da;
        border-radius: 5px;
    }
    .stMarkdown {
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    .stWarning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 10px;
        color: #856404;
    }
    .stInfo {
        background-color: #cce5ff;
        border: 1px solid #b8daff;
        border-radius: 5px;
        padding: 10px;
        color: #004085;
    }
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }
    .stDivider {
        border-color: #dee2e6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Kết nối SQLite với thread safety
conn = sqlite3.connect('user_data_saas.db', check_same_thread=False)
c = conn.cursor()

# Định nghĩa hàm hash_password trước các hàm khác
def hash_password(password: str) -> str:
    return sha256(password.encode()).hexdigest()

# Định nghĩa các hàm CRUD trước để tránh lỗi NameError
def read_user(username: str) -> Optional[tuple]:
    c.execute("SELECT * FROM users_auth WHERE username = ?", (username,))
    return c.fetchone()

def create_user(username: str, password: str) -> bool:
    try:
        hashed_password = hash_password(password)
        usage_time = None if username == "linh" else 180  # Admin có usage_time vô hạn (None)
        c.execute("INSERT INTO users_auth (username, password, login_time, usage_time, last_activity) VALUES (?, ?, 0, ?, 0)", (username, hashed_password, usage_time))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def update_user(username: str, password: Optional[str] = None, login_time: Optional[float] = None, 
                usage_time: Optional[float] = None, last_activity: Optional[float] = None) -> None:
    updates, values = [], []
    if password:
        updates.append("password = ?")
        values.append(hash_password(password))
    if login_time is not None:
        updates.append("login_time = ?")
        values.append(login_time)
    # Chỉ cập nhật usage_time nếu không phải admin
    if usage_time is not None and username != "linh":
        updates.append("usage_time = ?")
        values.append(usage_time)
    if last_activity is not None:
        updates.append("last_activity = ?")
        values.append(last_activity)
    if updates:
        values.append(username)
        c.execute(f"UPDATE users_auth SET {', '.join(updates)} WHERE username = ?", values)
        conn.commit()

def delete_user(username: str) -> None:
    c.execute("DELETE FROM users_auth WHERE username = ?", (username,))
    c.execute("DELETE FROM users_data WHERE username = ?", (username,))
    conn.commit()

# Tự động tạo tài khoản admin nếu chưa tồn tại
def initialize_admin() -> None:
    admin_username = "linh"
    admin_password = "13021995"
    if not read_user(admin_username):
        create_user(admin_username, admin_password)

initialize_admin()

# Tạo các bảng nếu chưa tồn tại
c.execute('''CREATE TABLE IF NOT EXISTS users_auth 
             (username TEXT PRIMARY KEY, password TEXT, login_time REAL, usage_time REAL, last_activity REAL)''')
c.execute('''CREATE TABLE IF NOT EXISTS users_data 
             (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, age INTEGER, height REAL, weight REAL, ideal_weight REAL, FOREIGN KEY (username) REFERENCES users_auth(username))''')
c.execute('''CREATE TABLE IF NOT EXISTS foods 
             (name TEXT, kcal REAL, protein REAL, carb REAL, fat REAL)''')
conn.commit()

# Dữ liệu thực phẩm mẫu ban đầu
initial_foods = [
    ("Cơm trắng (100g)", 130, 2.7, 28, 0.3),
    ("Ức gà (100g)", 165, 31, 0, 3.6),
    ("Chuối (1 quả, 120g)", 90, 1, 23, 0.3),
    ("Sữa tươi (200ml)", 120, 6, 9, 6),
    ("Cơm rang (100g)", 130, 2.7, 28, 0.3),
    ("Heo (100g)", 165, 31, 0, 3.6),
    ("Bánh mì (100g)", 250, 8, 48, 3),
    ("Trứng gà (1 quả, 50g)", 70, 6, 0.5, 5),
    ("Gạo lứt (100g)", 123, 2.5, 25, 1),
    ("Cá hồi (100g)", 200, 22, 0, 13),
    ("Hạt óc chó (100g)", 654, 15, 13, 65),
    ("Bơ (100g)", 717, 0.9, 0.1, 81)
]
c.executemany("INSERT OR IGNORE INTO foods VALUES (?, ?, ?, ?, ?)", initial_foods)
conn.commit()

# Hàm kiểm tra đăng nhập với session management
def check_login(username: str, password: str) -> bool:
    user = read_user(username)
    if user and hash_password(password) == user[1]:  # user[1] là password
        current_time = time.time()
        usage_time = user[3]  # Lấy usage_time từ database
        if username != "linh":  # Chỉ kiểm tra hết hạn cho user, không cho admin
            if usage_time is not None and usage_time <= 0:
                st.error("Tài khoản của bạn đã hết hạn sử dụng. Vui lòng liên hệ admin qua email: example@email.com", icon="❌")
                st.session_state['logged_in'] = False
                return False
            if user[2] and (current_time - user[2]) > (usage_time or 180):  # Kiểm tra hết hạn cho user
                st.session_state['logged_in'] = False
                st.session_state['username'] = None
                st.session_state['is_admin'] = False
                st.session_state['remaining_time'] = None
                st.error("Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.", icon="❌")
                return False
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        st.session_state['is_admin'] = (username == "linh")
        st.session_state['remaining_time'] = float('inf') if username == "linh" else (usage_time or 180)  # Admin có remaining_time vô hạn
        update_user(username, login_time=current_time, usage_time=None if username == "linh" else usage_time, last_activity=current_time)
        st.session_state['session_start'] = current_time  # Lưu thời điểm bắt đầu session
        return True
    return False

# Hàm đăng xuất với session cleanup
def logout() -> None:
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        update_user(st.session_state['username'], login_time=0, last_activity=0, usage_time=st.session_state.get('remaining_time', 180) if st.session_state['username'] != "linh" else None)
        for key in ['logged_in', 'username', 'is_admin', 'remaining_time', 'session_start', 'page']:
            st.session_state.pop(key, None)
        st.success("Đã đăng xuất thành công!", icon="✅")
        st.rerun()

# Hàm tính toán BMI, cân nặng lý tưởng
def calculate_bmi(height: float, weight: float) -> float:
    return weight / ((height / 100) ** 2)

def ideal_weight(height: float, age: int) -> float:
    return 20 if age < 18 else 22 if 18 <= age <= 65 else 24 * (height / 100) ** 2

def daily_calories_needed(current_weight: float, ideal_weight: float, days: int) -> float:
    max_weight_change_per_week = 1
    min_days_per_kg = {True: 7, False: 14}
    weight_diff = ideal_weight - current_weight
    is_increasing = weight_diff > 0
    if days < min_days_per_kg[is_increasing] * abs(weight_diff):
        min_days_needed = int(min_days_per_kg[is_increasing] * abs(weight_diff)) + 1
        st.warning(f"Không khả thi! Cần {min_days_needed} ngày {'tăng' if is_increasing else 'giảm'} {abs(weight_diff):.2f} kg ({'0.5-1' if is_increasing else '0.25-0.5'} kg/tuần).", icon="❌")
        return 0
    return weight_diff * 7700 / days if days > 0 else 0

# Hàm tạo thực đơn khoa học
def generate_scientific_meal_options(foods_df: pd.DataFrame, target_kcal_per_day: float, days: int, is_increasing: bool) -> list:
    options, used_foods = [], set()
    min_cal, max_cal = 500, 2000
    for _ in range(3):
        foods, kcal = [], min(max_cal, max(min_cal, target_kcal_per_day)) if is_increasing else max(min_cal, target_kcal_per_day)
        while kcal > 0 and len(foods) < 6:
            available = foods_df[~foods_df['name'].isin(used_foods) & (foods_df['kcal'] <= kcal)]
            if available.empty:
                break
            food = available.sample(1).iloc[0]
            foods.append(food)
            used_foods.add(food['name'])
            kcal -= food['kcal']
        
        if not foods:
            continue

        totals = {k: sum(f[k] for f in foods) for k in ['kcal', 'protein', 'carb', 'fat']}
        if is_increasing and totals['kcal'] < target_kcal_per_day:
            while totals['kcal'] < min(target_kcal_per_day, max_cal) and len(foods) < 8:
                available = foods_df[~foods_df['name'].isin(used_foods) & (foods_df['kcal'] <= min(target_kcal_per_day, max_cal) - totals['kcal'])]
                if available.empty:
                    break
                food = available.sample(1).iloc[0]
                foods.append(food)
                used_foods.add(food['name'])
                for k in ['kcal', 'protein', 'carb', 'fat']:
                    totals[k] += food[k]
        elif not is_increasing and totals['kcal'] > target_kcal_per_day:
            foods.sort(key=lambda x: x['kcal'], reverse=True)
            while totals['kcal'] > target_kcal_per_day and len(foods) > 1 and totals['kcal'] > min_cal:
                food = foods.pop()
                used_foods.remove(food['name'])
                for k in ['kcal', 'protein', 'carb', 'fat']:
                    totals[k] -= food[k]

        options.append({'foods': pd.DataFrame(foods), **{f'total_{k}': v for k, v in totals.items()}})
    used_foods.clear()
    return options

# Hàm gợi ý vận động
def suggest_exercise(calories_to_burn: float) -> list:
    return [f"- {e['name']}: {h * 60:.1f} phút (~{h:.2f} giờ) để đốt {calories_to_burn:.0f} kcal, KPI: {e['kpi']}" 
            for e in [{"name": "Chạy bộ", "kcal_per_hour": 600, "kpi": "5 km nhanh"}, 
                      {"name": "Đạp xe", "kcal_per_hour": 400, "kpi": "10 km trung bình"}, 
                      {"name": "Tập gym", "kcal_per_hour": 300, "kpi": "45 phút nặng"}, 
                      {"name": "Bơi lội", "kcal_per_hour": 500, "kpi": "1 km trung bình"}] 
            for h in [(calories_to_burn / e['kcal_per_hour'])]] if calories_to_burn > 0 else []

# Hàm tạo file tải xuống (sửa lỗi f-string)
def create_download_file(plan_markdown: str, user_info: dict, format: str = "md") -> tuple[str, str]:
    # Sử dụng chuỗi thông thường thay vì f-string để tránh lỗi backslash
    intro_lines = [
        "**Thông tin người dùng:**",
        "- Tuổi: " + str(user_info['age']),
        "- Chiều cao: " + str(user_info['height']) + " cm",
        "- Cân nặng: " + str(user_info['weight']) + " kg",
        "- Cân nặng lý tưởng: " + str(user_info['ideal_weight']) + " kg",
        "- Số ngày: " + str(user_info['days']),
        "- Mục tiêu: " + ('Tăng' if user_info['extra_calories'] > 0 else 'Giảm' if user_info['extra_calories'] < 0 else 'Duy trì') + " cân",
        "- Calo " + ('thêm' if user_info['extra_calories'] > 0 else 'giảm' if user_info['extra_calories'] < 0 else 'duy trì') + ": " + str(abs(user_info['extra_calories'])) + " kcal/ngày"
    ]
    intro = "\n".join(intro_lines) + "\n\n"

    if format == "md":
        content = "# Kế hoạch ăn uống thông minh\n\n" + intro + plan_markdown
        filename = "meal_plan.md"
    else:  # txt
        content = "Kế hoạch ăn uống thông minh\n\n" + intro.replace("**", "").replace("\n\n", "\n") + plan_markdown.replace("#", "").replace("---", "\n").replace("\n\n", "\n")
        filename = "meal_plan.txt"
    return content, filename

# Hàm quản lý URL và thời gian realtime
def update_url_and_time(username: str, is_admin: bool) -> None:
    st.session_state['page'] = f"/{'admin' if is_admin else 'user'}/{username}"
    current_time = time.time()
    user = read_user(username)
    login_time = user[2] if user else 0
    usage_time = user[3]  # Lấy giá trị thực tế từ database
    remaining = float('inf') if username == "linh" else max(0, (usage_time or 180) - (current_time - login_time)) if login_time else (usage_time or 180)
    st.session_state['remaining_time'] = remaining
    components.html(f"""
        <script>
            function updateTimer() {{
                let remaining = {remaining if remaining != float('inf') else 'Infinity'};
                if (remaining > 0 && remaining !== Infinity) {{
                    document.getElementById('timer').innerText = `Thời gian còn lại: ${Math.max(0, remaining.toFixed(0))} giây`;
                    remaining -= 1;
                    setTimeout(updateTimer, 1000);
                }} else if (remaining === Infinity) {{
                    document.getElementById('timer').innerText = "Thời gian sử dụng: Vô hạn (Admin)";
                }} else {{
                    document.getElementById('timer').innerText = "Hết thời gian sử dụng!";
                    window.location.href = '/main';
                }}
            }}
            window.history.pushState("", "", "{st.session_state['page']}");
            updateTimer();
        </script>
        <div id="timer"></div>
    """, height=30)

# Kiểm tra trạng thái đăng nhập với session persistence
def initialize_session() -> None:
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['is_admin'] = False
        st.session_state['page'] = "/main"
        st.session_state['remaining_time'] = None
        st.session_state['session_start'] = None

initialize_session()

# Xác thực trạng thái session khi refresh
if st.session_state['page'] != "/main" and st.session_state['logged_in']:
    username = st.session_state['username']
    user = read_user(username)
    if not user or (username != "linh" and user[2] and (time.time() - user[2]) > (user[3] or 180)):
        logout()
    else:
        st.session_state['is_admin'] = (username == "linh")
        update_url_and_time(username, st.session_state['is_admin'])

# Trang đăng nhập/đăng ký
if st.session_state['page'] == "/main" or not st.session_state['logged_in']:
    st.title("Đăng nhập/Đăng ký - Ứng dụng Thực đơn Thông Minh (SaaS) 🍎")

    tab1, tab2 = st.tabs(["Đăng nhập", "Đăng ký"])

    with tab1:
        st.subheader("Đăng nhập", divider="rainbow")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Đăng nhập", type="primary"):
            if username == "linh" and password == "13021995":
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['is_admin'] = True
                update_url_and_time(username, True)
                st.rerun()
            elif check_login(username, password):
                if read_user(username)[3] <= 0:
                    st.error("Tài khoản của bạn đã hết hạn sử dụng. Vui lòng liên hệ admin qua email: example@email.com", icon="❌")
                    st.session_state['logged_in'] = False
                else:
                    st.session_state['logged_in'] = True
                    st.session_state['is_admin'] = False
                    update_url_and_time(username, False)
                    st.rerun()
            else:
                st.error("Thông tin đăng nhập không đúng. Vui lòng thử lại hoặc liên hệ admin qua email: example@email.com", icon="❌")

    with tab2:
        st.subheader("Đăng ký", divider="rainbow")
        new_username = st.text_input("Username mới", key="register_username")
        new_password = st.text_input("Password mới", type="password", key="register_password")
        
        if st.button("Đăng ký", type="primary"):
            if create_user(new_username, new_password):
                st.session_state['logged_in'] = False  # Yêu cầu đăng nhập sau đăng ký
            else:
                st.error("Username đã tồn tại. Vui lòng thử lại.", icon="❌")

else:
    username = st.session_state['username']
    is_admin = st.session_state['is_admin']

    # Sidebar với đồng hồ thời gian realtime
    with st.sidebar:
        st.header(f"Chào {username}")
        if st.button("Đăng xuất", key="logout"):
            logout()
        if is_admin:
            st.write("Quản lý hệ thống")
        else:
            update_url_and_time(username, is_admin)  # Cập nhật URL và hiển thị timer realtime

    if is_admin:
        # Dashboard admin với CRUD đầy đủ
        st.title("Dashboard Quản lý Người dùng", help="Quản lý toàn diện người dùng và thời gian sử dụng")
        st.subheader("Danh sách người dùng", divider="rainbow")

        # Lấy và hiển thị danh sách người dùng
        c.execute("SELECT username, login_time, usage_time, last_activity FROM users_auth")
        users = c.fetchall()
        user_df = pd.DataFrame(users, columns=['Username', 'Login Time', 'Usage Time', 'Last Activity'])
        user_df['Login Time'] = pd.to_datetime(user_df['Login Time'], unit='s', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        user_df['Last Activity'] = pd.to_datetime(user_df['Last Activity'], unit='s', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        user_df['Usage Time'] = user_df['Usage Time'].apply(lambda x: 'Vô hạn' if pd.isna(x) else x).astype(str)  # Hiển thị "Vô hạn" cho admin
        user_df['Remaining Time'] = user_df.apply(
            lambda row: 'Vô hạn' if row['Username'] == "linh" else 
            max(0, (float('inf') if pd.isna(row['Usage Time']) else row['Usage Time']) - 
                (time.time() - (row['Login Time'].astype(float) if pd.notna(row['Login Time']) and isinstance(row['Login Time'], (int, float)) else 0))) 
            if pd.notna(row['Login Time']) and (float('inf') if pd.isna(row['Usage Time']) else row['Usage Time']) > 0 else 0, axis=1)

        st.dataframe(user_df, hide_index=True)

        # CRUD Operations
        st.subheader("Quản lý người dùng", divider="rainbow")
        action = st.selectbox("Chọn hành động", ["Thêm", "Sửa", "Xóa", "Xem"], key="crud_action")

        if action == "Thêm":
            new_username = st.text_input("Username mới", key="new_username")
            new_password = st.text_input("Password mới", type="password", key="new_password")
            new_usage_time = st.number_input("Thời gian sử dụng (giây)", min_value=60, max_value=86400, value=180, key="new_usage_time")
            if st.button("Thêm người dùng", key="add_user"):
                if create_user(new_username, new_password):
                    update_user(new_username, usage_time=None if new_username == "linh" else new_usage_time, login_time=0, last_activity=0)
                    st.success(f"Đã thêm người dùng {new_username} với thời gian {'Vô hạn' if new_username == 'linh' else new_usage_time} giây!", icon="✅")

        elif action == "Sửa":
            selected_user = st.selectbox("Chọn người dùng", options=user_df['Username'].tolist(), key="edit_user")
            if selected_user:
                new_password = st.text_input("Password mới (rỗng nếu không đổi)", type="password", key="edit_password")
                new_usage_time = st.number_input("Thời gian sử dụng (giây)", min_value=60, max_value=86400, 
                                                value=float('inf') if selected_user == "linh" else int(user_df[user_df['Username'] == selected_user]['Usage Time'].iloc[0]), 
                                                key="edit_usage_time") if selected_user != "linh" else st.write("Admin có thời gian sử dụng vô hạn, không thể sửa.")
                if st.button("Cập nhật", key="update_user"):
                    updates = {}
                    if new_password:
                        updates['password'] = new_password
                    updates['usage_time'] = None if selected_user == "linh" else (new_usage_time if selected_user != "linh" else user_df[user_df['Username'] == selected_user]['Usage Time'].iloc[0])
                    updates['login_time'] = 0  # Reset login time để remaining time bắt đầu mới
                    updates['last_activity'] = 0
                    update_user(selected_user, **updates)
                    st.success(f"Đã cập nhật {selected_user} với thời gian {'Vô hạn' if selected_user == 'linh' else new_usage_time} giây!", icon="✅")

        elif action == "Xóa":
            selected_user = st.selectbox("Chọn người dùng để xóa", options=user_df['Username'].tolist(), key="delete_user")
            if selected_user and st.button("Xóa người dùng", key="delete_user_confirm"):
                if selected_user == "linh":
                    st.error("Không thể xóa tài khoản admin 'linh'.", icon="❌")
                else:
                    delete_user(selected_user)
                    st.success(f"Đã xóa {selected_user}!", icon="✅")

        elif action == "Xem":
            selected_user = st.selectbox("Chọn người dùng để xem", options=user_df['Username'].tolist(), key="view_user")
            if selected_user:
                c.execute("SELECT age, height, weight, ideal_weight FROM users_data WHERE username = ?", (selected_user,))
                user_data = c.fetchone()
                if user_data:
                    age, height, weight, ideal_weight = user_data
                    st.subheader(f"Thông tin {selected_user}", divider="rainbow")
                    st.write(f"- Tuổi: {age}")
                    st.write(f"- Chiều cao: {height:.2f} cm")
                    st.write(f"- Cân nặng: {weight:.2f} kg")
                    st.write(f"- Cân nặng lý tưởng: {ideal_weight:.2f} kg")
                else:
                    st.warning("Người dùng chưa có dữ liệu.", icon="⚠️")

    else:
        # Trang người dùng, kiểm tra thời gian sử dụng realtime
        c.execute("SELECT usage_time, login_time, last_activity FROM users_auth WHERE username = ?", (username,))
        user_data = c.fetchone()
        if not user_data or user_data[2] <= 0:  # Kiểm tra usage_time đã hết hạn
            st.error("Tài khoản của bạn đã hết hạn sử dụng. Vui lòng liên hệ admin qua email: example@email.com", icon="❌")
            logout()
            st.rerun()
        usage_time, login_time, _ = user_data
        current_time = time.time()
        remaining_time = max(0, (usage_time or 180) - (current_time - (login_time or current_time)))

        if remaining_time <= 0:
            st.error("Phiên đăng nhập đã hết hạn (3 phút). Vui lòng đăng nhập lại hoặc liên hệ admin qua email: example@email.com", icon="❌")
            logout()
            st.rerun()
        else:
            st.session_state['remaining_time'] = remaining_time
            st.title(f"Chào {username} - Ứng dụng Thực đơn Thông Minh 🍎", help="Tạo kế hoạch ăn uống khoa học.")
            
            # Import CSV
            st.subheader("Import thực phẩm từ CSV", divider="rainbow")
            uploaded_file = st.file_uploader("Tải file CSV", type=['csv'], help="Cột: name, kcal, protein, carb, fat")
            if uploaded_file:
                import_csv(uploaded_file)

            # Nhập thông tin người dùng
            st.header("Thông tin cá nhân", divider="rainbow")
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Tuổi", 1, 120, 30, key="user_age")
            with col2:
                height = st.number_input("Chiều cao (cm)", 50.0, 250.0, 167.0, key="user_height")
            with col3:
                weight = st.number_input("Cân nặng (kg)", 20.0, 200.0, 64.0, key="user_weight")

            if st.button("Tính toán & lưu", type="primary"):
                bmi = calculate_bmi(height, weight)
                ideal_w = ideal_weight(height, age)
                st.session_state['bmi'] = bmi
                st.session_state['ideal_w'] = ideal_w
                weight_diff = ideal_w - weight
                if weight_diff > 0:
                    min_days = max(1, int(7 * weight_diff))
                else:
                    min_days = max(1, int(14 * abs(weight_diff)))
                st.session_state['days'] = min_days
                st.session_state['weight_diff'] = weight_diff

                c.execute("INSERT OR REPLACE INTO users_data (username, age, height, weight, ideal_weight) VALUES (?, ?, ?, ?, ?)",
                          (username, age, height, weight, ideal_w))
                conn.commit()

                st.success(f"BMI: {bmi:.2f}", icon="📊")
                st.success(f"Cân nặng lý tưởng: {ideal_w:.2f} kg", icon="⚖️")

            # Kế hoạch tăng/giảm cân
            st.header("Kế hoạch tăng/giảm cân", divider="rainbow")
            days = st.number_input("Số ngày đạt cân nặng lý tưởng", 1, 365, st.session_state.get('days', 1))

            if 'ideal_w' in st.session_state and 'weight_diff' in st.session_state:
                extra_cal = daily_calories_needed(weight, st.session_state['ideal_w'], days)
                if extra_cal == 0:
                    st.session_state['extra_calories'] = None
                else:
                    st.session_state['extra_calories'] = extra_cal
                    if extra_cal > 0:
                        st.info(f"Thêm {extra_cal:.0f} kcal/ngày để đạt {st.session_state['ideal_w']:.2f} kg trong {days} ngày", icon="➕")
                    elif extra_cal < 0:
                        st.info(f"Giảm {abs(extra_cal):.0f} kcal/ngày để đạt {st.session_state['ideal_w']:.2f} kg trong {days} ngày", icon="➖")
                        st.write(f"""
                        **Giải thích:** Giảm cân lành mạnh (0.25-0.5 kg/tuần, 14-28 ngày/kg). 
                        Mục tiêu {abs(st.session_state['weight_diff']):.2f} kg cần {int(14 * abs(st.session_state['weight_diff']))}-{int(28 * abs(st.session_state['weight_diff']))} ngày. 
                        Kế hoạch {days} ngày hợp lý, kết hợp ăn và vận động {abs(extra_cal):.0f} kcal/ngày. 
                        Tránh <500 kcal/ngày hoặc >1,000 kcal/ngày đốt để tránh stress, mệt mỏi, mất cơ.
                        """)
                    else:
                        st.success("Cân nặng hiện tại lý tưởng!", icon="✅")

                show_plan_disabled = 'extra_calories' not in st.session_state or st.session_state['extra_calories'] is None
                if st.button("Hiển thị kế hoạch", disabled=show_plan_disabled, type="primary"):
                    if not st.session_state['extra_calories']:
                        st.warning("Kế hoạch không khả thi. Chọn số ngày hợp lý.", icon="❌")
                    else:
                        target_cal = abs(st.session_state['extra_calories'])
                        is_increasing = st.session_state['extra_calories'] > 0
                        foods_df = pd.read_sql_query("SELECT * FROM foods", conn)

                        st.subheader("Kế hoạch từng ngày", divider="rainbow")
                        plan_markdown = ""
                        for day in range(days):
                            date = datetime(2025, 2, 28) + timedelta(days=day)
                            plan_markdown += f"## {date.strftime('%Y-%m-%d')} (Ngày {day + 1})\nNhóm thực phẩm:\n"
                            options = generate_scientific_meal_options(foods_df, target_cal, days, is_increasing)
                            for i, option in enumerate(options, 1):
                                plan_markdown += f"Cách {i}:\n"
                                for _, row in option['foods'].iterrows():
                                    plan_markdown += f"- {row['name']}: {row['kcal']} kcal, {row['protein']}g protein, {row['carb']}g carb, {row['fat']}g fat\n"
                                plan_markdown += f"**Tổng ngày {day + 1} (Cách {i})**: {option['total_kcal']:.0f} kcal, {option['total_protein']:.1f}g protein, {option['total_carb']:.1f}g carb, {option['total_fat']:.1f}g fat\n\n"
                            if not is_increasing:
                                for i, option in enumerate(options, 1):
                                    tdee = 1900
                                    burn = option['total_kcal'] - (tdee - target_cal)
                                    if burn > 0:
                                        plan_markdown += f"**Vận động Cách {i} (đốt {burn:.0f} kcal):**\n" + "\n".join(suggest_exercise(burn))
                                    elif burn < 0:
                                        plan_markdown += f"**Bổ sung Cách {i} (cần thêm {-burn:.0f} kcal):**\n- Giữ thực đơn, vận động nhẹ (200-400 kcal/ngày).\n"
                            plan_markdown += "---\n"

                        st.session_state['plan_markdown'] = plan_markdown
                        st.markdown(plan_markdown)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.button("Hiển thị kế hoạch", disabled=show_plan_disabled, key="show_plan", type="primary")
                        with col2:
                            if 'plan_markdown' in st.session_state:
                                fmt = st.radio("Định dạng tải xuống", ["Markdown (.md)", "Text (.txt)"], horizontal=True, key="download_format")
                                content, fname = create_download_file(plan_markdown, {
                                    'age': age, 'height': height, 'weight': weight, 'ideal_weight': st.session_state['ideal_w'],
                                    'days': days, 'extra_calories': st.session_state['extra_calories']
                                }, "md" if fmt == "Markdown (.md)" else "txt")
                                st.download_button("Tải kế hoạch", content.encode('utf-8'), fname, "text/plain" if fmt == "Text (.txt)" else "text/markdown", type="secondary")

                        if is_increasing:
                            st.info("**Gợi ý:** Thực đơn tăng cân, chọn theo sở thích!", icon="💪")
                            if any(o['total_kcal'] < target_cal for o in options):
                                st.warning(f"Không đạt {target_cal:.0f} kcal/ngày. Bổ sung thực phẩm!", icon="❌")
                        elif not is_increasing:
                            st.info(f"Giảm {abs(st.session_state['extra_calories']):.0f} kcal/ngày để đạt {st.session_state['ideal_w']:.2f} kg trong {days} ngày", icon="➖")
                            st.info("**Gợi ý:** Kết hợp thực đơn, vận động để giảm an toàn. Xem gợi ý vận động.", icon="🏃")
                        else:
                            st.success("Cân nặng hiện tại lý tưởng!", icon="✅")

            # Hiển thị định dạng CSV mẫu
            st.subheader("Định dạng CSV mẫu", divider="rainbow")
            st.write("""
            | name              | kcal  | protein | carb  | fat  |
            |-------------------|-------|---------|-------|------|
            | Cơm trắng (100g)  | 130   | 2.7     | 28    | 0.3  |
            | Ức gà (100g)      | 165   | 31      | 0     | 3.6  |
            Lưu file `.csv`, tải lên qua nút 'Tải lên file CSV'.
            """)

conn.close()
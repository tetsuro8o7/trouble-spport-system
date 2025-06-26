import streamlit as st
import pandas as pd
import os
import datetime
import time
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from filelock import FileLock  # 🔒 追加

# --- CSVファイルパス ---
CSV_PATH = "trouble_list.csv"
LOCK_PATH = CSV_PATH + ".lock"  # 🔒 ロックファイルパス

# --- データ読み込み ---
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
else:
    df = pd.DataFrame(columns=[
        "発生拠点", "発生年月日", "成形機No.", "設備名", "トラブル内容",
        "原因", "是正内容", "対応時間(h)", "対応者", "調査過程", "調査時の注意点"
    ])

# --- BERTモデル読み込み（日本語対応） ---
@st.cache_resource
def load_model():
    return SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens')
model = load_model()

# --- 類似検索関数 ---
def find_similar_troubles_bert(input_trouble, df, top_n=5):
    troubles = df['トラブル内容'].dropna().tolist()
    sentences = troubles + [input_trouble]
    embeddings = model.encode(sentences)
    input_vec = embeddings[-1].reshape(1, -1)
    trouble_vecs = embeddings[:-1]
    cosine_sim = cosine_similarity(input_vec, trouble_vecs)
    top_indices = cosine_sim[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# --- 共通パスワード認証 ---
def check_password(main_password="hofu"):
    def password_entered():
        st.session_state["main_password_correct"] = st.session_state["main_password"] == main_password
    if "main_password_correct" not in st.session_state:
        st.text_input("🔐 システム起動パスワードを入力してください", type="password", on_change=password_entered, key="main_password")
        return False
    elif not st.session_state["main_password_correct"]:
        st.text_input("🔐 システム起動パスワードを入力してください", type="password", on_change=password_entered, key="main_password")
        st.error("❌ パスワードが間違っています。")
        return False
    else:
        return True

# --- 新規登録ページ専用パスワード認証 ---
def check_register_password(register_password="hozen"):
    def password_entered():
        st.session_state["register_password_correct"] = st.session_state["register_password"] == register_password
    if "register_password_correct" not in st.session_state:
        st.text_input("🔐 新規登録ページ用パスワードを入力してください", type="password", on_change=password_entered, key="register_password")
        return False
    elif not st.session_state["register_password_correct"]:
        st.text_input("🔐 新規登録ページ用パスワードを入力してください", type="password", on_change=password_entered, key="register_password")
        st.error("❌ パスワードが間違っています。")
        return False
    else:
        return True

# --- システム起動パスワードチェック ---
if not check_password():
    st.stop()

# --- ロゴとタイトル表示 ---
st.sidebar.image("logo.jpg", width=200)
st.markdown(
    """
    <style>
    .main {
        padding-top: 10px;
    }
    </style>
    <div style='margin-top: -40px; text-align: center;'>
        <h1 style='color: darkred;'>🚨 トラブル対策支援システム 🚨</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# --- ページ選択 ---
page = st.sidebar.radio("ページを選択", ["🔍 トラブル検索", "📝 新規登録"])

# --- トラブル検索ページ ---
if page == "🔍 トラブル検索":
    st.subheader("🔍 トラブル検索")
    equipment_options = ["すべて"] + sorted(df["設備名"].dropna().unique().tolist())
    selected_equipment = st.selectbox("🏭 設備名でフィルター", equipment_options)
    input_trouble = st.text_input("💬 トラブル内容を入力してください")
    filtered_df = df.copy()
    if selected_equipment != "すべて":
        filtered_df = filtered_df[filtered_df["設備名"] == selected_equipment]
    if st.button("検索") and input_trouble.strip():
        similar_df = find_similar_troubles_bert(input_trouble, filtered_df)
        for _, row in similar_df.iterrows():
            st.markdown("### 🛠 類似トラブル")
            st.write(f"📍 **発生拠点**: {row['発生拠点']}")
            st.write(f"📅 **発生年月日**: {row['発生年月日']}")
            st.write(f"🔢 **成形機No.**: {row['成形機No.']}")
            st.write(f"🏭 **設備名**: {row['設備名']}")
            st.write(f"💬 **トラブル内容**: {row['トラブル内容']}")
            st.write(f"🛠 **原因**: {row['原因']}")
            st.write(f"🧪 **是正内容**: {row['是正内容']}")
            st.write(f"⏱ **対応時間(h)**: {row['対応時間(h)']}")
            st.write(f"👤 **対応者**: {row['対応者']}")
            st.write(f"🔍 **調査過程**: {row['調査過程']}")
            st.write(f"⚠️ **調査時の注意点**: {row['調査時の注意点']}")
            st.markdown("---")
    if not df.empty:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='TroubleList')
        output.seek(0)
        st.download_button(
            label="📥 トラブルリストダウンロード",
            data=output,
            file_name="trouble_list.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# --- 新規登録ページ ---
elif page == "📝 新規登録":
    if check_register_password():
        st.subheader("📝 トラブル新規登録")
        with st.form("trouble_form"):
            new_location = st.selectbox("発生拠点", ["防府", "大津", "アスコ", "美土里", "MNAC", "MAM"], key="location")
            new_date = st.date_input("発生年月日", value=datetime.date.today(), key="date")
            new_machine = st.text_input("成形機No.", key="machine")
            new_equipment = st.selectbox("設備名", ["成形機", "取出機", "オートローダ", "温調器", "ホットランナー", "バルブゲート", "金型交換台車", "ASSY機", "溶着機", "リークテスト機", "その他"], key="equipment")
            new_content = st.text_area("トラブル内容", key="content")
            new_cause = st.text_area("原因", key="cause")
            new_action = st.text_area("是正内容", key="action")
            new_time = st.number_input("対応時間(h)", key="time")
            new_person = st.text_input("対応者", key="person")
            new_process = st.text_area("調査過程", key="process")
            new_notes = st.text_area("調査時の注意点", key="notes")
            submitted = st.form_submit_button("登録")
        if submitted:
            # 各項目が空でないかチェック
            if all([
                new_location,  # selectbox は常に値を持つ
                new_date,      # date_input も常に値を持つ
                new_machine.strip(),
                new_equipment,  # selectbox も常に値を持つ
                new_content.strip(),
                new_cause.strip(),
                new_action.strip(),
                new_time > 0,  # number_input は 0 以上の数値を入れるように制限可能
                new_person.strip(),
                new_process.strip(),
                new_notes.strip()
            ]):
                # 登録処理
                new_data = {
                    "発生拠点": new_location,
                    "発生年月日": new_date.strftime("%Y/%m/%d"),
                    "成形機No.": new_machine,
                    "設備名": new_equipment,
                    "トラブル内容": new_content,
                    "原因": new_cause,
                    "是正内容": new_action,
                    "対応時間(h)": new_time,
                    "対応者": new_person,
                    "調査過程": new_process,
                    "調査時の注意点": new_notes
                }
                new_df = pd.DataFrame([new_data])
                # 🔒 ファイルロックを使って排他制御
                with FileLock(LOCK_PATH, timeout=10):
                    file_exists = os.path.isfile(CSV_PATH)
                    new_df.to_csv(CSV_PATH, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
                st.toast("✅ 登録が完了しました！")
                time.sleep(5)
                st.rerun()
            else:
                st.error("⚠ 全ての項目を正しく入力してください。")


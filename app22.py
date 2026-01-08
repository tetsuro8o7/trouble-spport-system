
import streamlit as st
import pandas as pd
import os
import datetime
import time
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from filelock import FileLock

# ==============================
# 設定／定数
# ==============================
CSV_PATH = "trouble_list.csv"
LOCK_PATH = CSV_PATH + ".lock"
ENCODING = "utf-8-sig"

# CSVの列定義（完全一致で維持）
COLUMNS = [
    "発生拠点", "発生年月日", "成形機No.", "設備名", "トラブル内容",
    "原因", "是正内容", "対応時間(h)", "対応者", "調査過程", "調査時の注意点"
]

# ==============================
# Secrets 読み込み（必須）
# ==============================
def get_main_password() -> str:
    try:
        return st.secrets["MAIN_PASSWORD"]
    except Exception:
        return None

def get_register_password() -> str:
    try:
        return st.secrets["REGISTER_PASSWORD"]
    except Exception:
        return None

MAIN_PASSWORD = get_main_password()
REGISTER_PASSWORD = get_register_password()

# Secrets未設定のときはガイダンス表示して停止
if MAIN_PASSWORD is None or REGISTER_PASSWORD is None:
    st.error(
        "🔒 Secrets が未設定です。Streamlit Cloud の **Settings → Advanced settings → Secrets** に\n"
        "```\nMAIN_PASSWORD = \"hofu\"\nREGISTER_PASSWORD = \"hozen\"\n```\n"
        "のように登録してください。"
    )
    st.stop()

# ==============================
# モデル読み込み（キャッシュ）
# ==============================
@st.cache_resource
def load_model():
    # 日本語Sentence-BERT（モデルは必要に応じて軽量モデルへ変更可）
    return SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")

model = load_model()

# ==============================
# ユーティリティ
# ==============================
def safe_read_csv(path: str, encoding: str = ENCODING) -> pd.DataFrame:
    """壊れたCSVでも落ちないように読み込み。列が欠けたら補完、順序は既存に合わせる。"""
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(path, encoding=encoding)
        # 列の補完と並べ替え
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = ""
        # 既存の列はそのまま優先、足りない列は末尾に
        ordered = [c for c in df.columns if c in COLUMNS] + [c for c in COLUMNS if c not in df.columns]
        df = df[ordered]
        return df
    except Exception as e:
        st.error(f"CSV読み込みに失敗しました: {e}")
        return pd.DataFrame(columns=COLUMNS)

def find_similar_troubles_bert(input_trouble: str, df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """BERT類似検索。トラブル内容が空しかない場合は空の結果を返す。"""
    if df.empty or df["トラブル内容"].dropna().empty:
        return pd.DataFrame(columns=df.columns)
    troubles = df["トラブル内容"].fillna("").tolist()
    # 全て空なら検索しない
    if all(t.strip() == "" for t in troubles):
        return pd.DataFrame(columns=df.columns)
    sentences = troubles + [input_trouble]
    embeddings = model.encode(sentences)
    input_vec = embeddings[-1].reshape(1, -1)
    trouble_vecs = embeddings[:-1]
    cosine_sim = cosine_similarity(input_vec, trouble_vecs)
    top_indices = cosine_sim[0].argsort()[-top_n:][::-1]
    # インデックスがdfと対応するように、元の行番号を拾う
    return df.iloc[top_indices]

def check_password(main_password: str) -> bool:
    """共通パスワード認証（Secrets由来）"""
    def password_entered():
        st.session_state["main_password_correct"] = st.session_state.get("main_password", "") == main_password

    if "main_password_correct" not in st.session_state:
        st.text_input("🔐 システム起動パスワードを入力してください", type="password",
                      on_change=password_entered, key="main_password")
        return False
    elif not st.session_state["main_password_correct"]:
        st.text_input("🔐 システム起動パスワードを入力してください", type="password",
                      on_change=password_entered, key="main_password")
        st.error("❌ パスワードが間違っています。")
        return False
    else:
        return True

def check_register_password(register_password: str) -> bool:
    """新規登録ページ専用パスワード認証（Secrets由来）"""
    def password_entered():
        st.session_state["register_password_correct"] = st.session_state.get("register_password", "") == register_password

    if "register_password_correct" not in st.session_state:
        st.text_input("🔐 新規登録ページ用パスワードを入力してください", type="password",
                      on_change=password_entered, key="register_password")
        return False
    elif not st.session_state["register_password_correct"]:
        st.text_input("🔐 新規登録ページ用パスワードを入力してください", type="password",
                      on_change=password_entered, key="register_password")
        st.error("❌ パスワードが間違っています。")
        return False
    else:
        return True

def show_diagnostics(csv_path: str):
    """診断表示：絶対パス・サイズ・更新時刻・末尾行・pandas末尾"""
    csv_abs = os.path.abspath(csv_path)
    st.write("📄 CSV絶対パス:", csv_abs)
    if os.path.exists(csv_abs):
        st.write("📏 ファイルサイズ:", os.path.getsize(csv_abs), "bytes")
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(csv_abs))
        st.write("🕒 最終更新:", mtime.strftime("%Y/%m/%d %H:%M:%S"))
        try:
            with open(csv_abs, "r", encoding=ENCODING) as f:
                tail_lines = f.read().splitlines()[-5:]
            st.code("\n".join(tail_lines), language="text")
        except Exception as e:
            st.warning(f"末尾テキストの読込に失敗: {e}")
        try:
            tail_df = pd.read_csv(csv_abs, encoding=ENCODING).tail(3)
            st.write("🧪 pandasでの末尾3行:", tail_df)
        except Exception as e:
            st.warning(f"pandas読込に失敗: {e}")
    else:
        st.error("❌ CSVファイルが存在しません。パスの誤認かも。")

# ==============================
# 起動時のデータ読み込み
# ==============================
df = safe_read_csv(CSV_PATH)

# ==============================
# 画面構成
# ==============================
# --- パスワードチェック（Secrets） ---
if not check_password(MAIN_PASSWORD):
    st.stop()

# --- ロゴ（存在時のみ）とタイトル ---
try:
    if os.path.exists("logo.jpg"):
        st.sidebar.image("logo.jpg", width=200)
except Exception:
    pass

st.markdown(
    """
    <style>
    .main { padding-top: 10px; }
    </style>
    <div style='margin-top: -40px; text-align: center;'>
      <h1 style='color: darkred;'>🚨 トラブル対策支援システム 🚨</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# --- 診断モードトグル ---
diagnostics_enabled = st.sidebar.checkbox("🧰 診断モード（CSV末尾・パス表示）", value=False)

# --- ページ選択 ---
page = st.sidebar.radio("ページを選択", ["🔍 トラブル検索", "📝 新規登録"])

# ==============================
# 🔍 トラブル検索ページ
# ==============================
if page == "🔍 トラブル検索":
    st.subheader("🔍 トラブル検索")
    equipment_options = ["すべて"] + sorted(df["設備名"].dropna().unique().tolist())
    selected_equipment = st.selectbox("🏭 設備名でフィルター", equipment_options)
    input_trouble = st.text_input("💬 トラブル内容を入力してください")

    filtered_df = df.copy()
    if selected_equipment != "すべて":
        filtered_df = filtered_df[filtered_df["設備名"] == selected_equipment]

    # 類似検索
    if st.button("検索") and input_trouble.strip():
        similar_df = find_similar_troubles_bert(input_trouble, filtered_df)
        if similar_df.empty:
            st.info("該当する類似トラブルが見つかりませんでした。")
        else:
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
                st.write(f"🔎 **調査過程**: {row['調査過程']}")
                st.write(f"⚠️ **調査時の注意点**: {row['調査時の注意点']}")
                st.markdown("---")

    # エクスポート
    if not df.empty:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="TroubleList")
        output.seek(0)
        st.download_button(
            label="📥 トラブルリストダウンロード",
            data=output,
            file_name="trouble_list.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # 診断
    if diagnostics_enabled:
        st.markdown("#### 🔎 診断情報")
        show_diagnostics(CSV_PATH)

# ==============================
# 📝 新規登録ページ
# ==============================
elif page == "📝 新規登録":
    if check_register_password(REGISTER_PASSWORD):
        st.subheader("📝 トラブル新規登録")
        with st.form("trouble_form"):
            new_location = st.selectbox("発生拠点", ["防府", "大津", "アスコ", "美土里", "MNAC", "MAM"], key="location")
            new_date = st.date_input("発生年月日", value=datetime.date.today(), key="date")
            new_machine = st.text_input("成形機No.", key="machine")
            new_equipment = st.selectbox("設備名", [
                "成形機", "取出機", "オートローダ", "温調器", "ホットランナー",
                "バルブゲート", "金型交換台車", "ASSY機", "溶着機", "リークテスト機", "その他"
            ], key="equipment")
            new_content = st.text_area("トラブル内容", key="content")
            new_cause = st.text_area("原因", key="cause")
            new_action = st.text_area("是正内容", key="action")
            # 0時間を許容
            new_time = st.number_input("対応時間(h)", min_value=0.0, step=0.5, key="time")
            new_person = st.text_input("対応者", key="person")
            new_process = st.text_area("調査過程", key="process")
            new_notes = st.text_area("調査時の注意点", key="notes")

            submitted = st.form_submit_button("登録")

        if submitted:
            try:
                # --- バリデーション（0以上を許容） ---
                if not all([
                    new_machine.strip(),
                    new_content.strip(),
                    new_cause.strip(),
                    new_action.strip(),
                    (new_time >= 0),
                    new_person.strip(),
                    new_process.strip(),
                    new_notes.strip()
                ]):
                    st.error("⚠ 全ての必須項目を正しく入力してください。対応時間は0以上にしてください。")
                    st.stop()

                # --- 新規行の作成（列完全一致＆strip） ---
                new_row = {
                    "発生拠点": new_location,
                    "発生年月日": new_date.strftime("%Y/%m/%d"),
                    "成形機No.": new_machine.strip(),
                    "設備名": new_equipment,
                    "トラブル内容": new_content.strip(),
                    "原因": new_cause.strip(),
                    "是正内容": new_action.strip(),
                    "対応時間(h)": float(new_time),
                    "対応者": new_person.strip(),
                    "調査過程": new_process.strip(),
                    "調査時の注意点": new_notes.strip()
                }
                new_df = pd.DataFrame([new_row], columns=COLUMNS)

                # --- ロック取得＆読み込み→結合→上書き ---
                csv_abs = os.path.abspath(CSV_PATH)
                lock_abs = os.path.abspath(LOCK_PATH)
                with FileLock(lock_abs, timeout=10):
                    existing = safe_read_csv(csv_abs)
                    # 既存の列順に合わせて結合
                    for c in existing.columns:
                        if c not in new_df.columns:
                            new_df[c] = ""
                    combined = pd.concat([existing, new_df[existing.columns]], ignore_index=True)
                    # 上書き保存（ヘッダーは常に1回）
                    combined.to_csv(csv_abs, index=False, encoding=ENCODING, lineterminator="\n")

                # --- 書き込み直後の確認（任意） ---
                if diagnostics_enabled:
                    st.markdown("#### 🈺 登録直後の確認")
                    try:
                        tail = pd.read_csv(csv_abs, encoding=ENCODING).tail(3)
                        st.write("末尾3行:", tail)
                    except Exception as e:
                        st.warning(f"登録後の確認読み込みに失敗: {e}")

                st.toast("✅ 登録が完了しました！")
                time.sleep(0.3)
                st.rerun()

            except PermissionError as e:
                st.error("❌ 書き込み権限エラー: Excel等でCSVを開きっぱなしにしていないか確認してね。詳細: " + str(e))
            except TimeoutError as e:
                st.error("❌ ロック取得タイムアウト: 同時編集が続いていないか確認してね。詳細: " + str(e))
            except Exception as e:
                st.error(f"❌ 予期せぬエラー: {e}")

        # 診断
        if diagnostics_enabled:
            st.markdown("#### 🔎 診断情報")
            show_diagnostics(CSV_PATH)

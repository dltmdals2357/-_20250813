
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="MBTI by Country Explorer", layout="wide")

st.title("MBTI by Country — Quick Explorer")

# ---- Data loading ----
st.sidebar.header("1) 데이터 불러오기")
uploaded = st.sidebar.file_uploader("CSV 업로드 (columns: Country, 16 MBTI types)", type=["csv"])

def load_data(file):
    df = pd.read_csv(file)
    # Basic sanitization: keep expected columns if present
    df.columns = [c.strip() for c in df.columns]
    if "Country" not in df.columns:
        raise ValueError("CSV에 'Country' 열이 없습니다.")
    # Keep only Country + 16 MBTI columns if possible
    mbti_types = ["INFJ","ISFJ","INTP","ISFP","ENTP","INFP","ENTJ","ISTP","INTJ",
                  "ESFP","ESTJ","ENFP","ESTP","ISTJ","ENFJ","ESFJ"]
    keep_cols = ["Country"] + [c for c in mbti_types if c in df.columns]
    df = df[keep_cols].copy()
    # Convert numeric
    for c in df.columns:
        if c != "Country":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

if uploaded is not None:
    df = load_data(uploaded)
else:
    # Fallback to known path for this session
    default_path = "/mnt/data/countriesMBTI_16types.csv"
    df = load_data(default_path)

st.sidebar.success(f"불러온 국가 수: {len(df)}")

mbti_cols = [c for c in df.columns if c != "Country"]

# ---- Overview ----
st.subheader("개요")
left, right = st.columns(2)
with left:
    st.write(f"**행(국가):** {len(df)}  |  **MBTI 유형 열:** {len(mbti_cols)}")
    st.dataframe(df.head(10), use_container_width=True)

with right:
    st.write("**결측치 개수(열별):**")
    st.write(df[mbti_cols].isna().sum())

# ---- Global average distribution ----
st.subheader("전세계 평균 분포")
avg = df[mbti_cols].mean().sort_values(ascending=False)
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(avg.index, avg.values)
ax1.set_ylabel("평균 비율")
ax1.set_xticklabels(avg.index, rotation=45, ha="right")
st.pyplot(fig1, clear_figure=True)

# ---- Country-level view ----
st.subheader("국가별 분포")
country = st.selectbox("국가 선택", df["Country"].tolist())
row = df.loc[df["Country"] == country, mbti_cols].iloc[0]

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.bar(mbti_cols, row.values)
ax2.set_ylabel("비율")
ax2.set_xticklabels(mbti_cols, rotation=45, ha="right")
st.pyplot(fig2, clear_figure=True)

# ---- Top-N countries by a selected MBTI ----
st.subheader("특정 MBTI 기준 상위 국가")
colA, colB = st.columns([2,1])
with colA:
    chosen_type = st.selectbox("MBTI 선택", mbti_cols, index=mbti_cols.index("ENFP") if "ENFP" in mbti_cols else 0)
with colB:
    top_n = st.slider("상위 N", min_value=3, max_value=20, value=10, step=1)

top_df = df[["Country", chosen_type]].dropna().sort_values(chosen_type, ascending=False).head(top_n)
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.barh(top_df["Country"], top_df[chosen_type])
ax3.invert_yaxis()
ax3.set_xlabel("비율")
st.pyplot(fig3, clear_figure=True)
st.dataframe(top_df.reset_index(drop=True), use_container_width=True)

# ---- Similar countries (cosine similarity) ----
st.subheader("유사 국가 찾기 (코사인 유사도)")
def cosine_sim(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.allclose(a, 0) or np.allclose(b, 0):
        return np.nan
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

target = df.loc[df["Country"] == country, mbti_cols].iloc[0].values
sims = []
for _, r in df.iterrows():
    if r["Country"] == country:
        continue
    sim = cosine_sim(target, r[mbti_cols].values)
    sims.append((r["Country"], sim))
sim_df = pd.DataFrame(sims, columns=["Country", "CosineSim"]).sort_values("CosineSim", ascending=False).head(10)
st.dataframe(sim_df, use_container_width=True)

# ---- Correlation heatmap ----
st.subheader("MBTI 유형 간 상관관계(국가 단위)")
corr = df[mbti_cols].corr()
fig4, ax4 = plt.subplots(figsize=(8, 6))
im = ax4.imshow(corr.values, aspect="auto")
ax4.set_xticks(range(len(mbti_cols)))
ax4.set_yticks(range(len(mbti_cols)))
ax4.set_xticklabels(mbti_cols, rotation=45, ha="right")
ax4.set_yticklabels(mbti_cols)
ax4.set_title("상관계수 행렬")
st.pyplot(fig4, clear_figure=True)

# ---- Download filtered data ----
st.subheader("데이터 내보내기")
csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button("전체 데이터 CSV 다운로드", data=csv_bytes, file_name="mbti_countries_clean.csv", mime="text/csv")

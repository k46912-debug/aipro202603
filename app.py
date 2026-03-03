import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# 設定網頁標題與版面
st.set_page_config(page_title="Wine Classifier Web", layout="wide")

# 1. 載入資料集
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return wine, df

wine_data, df_wine = load_data()

# 2. 側邊欄 (Sidebar)
st.sidebar.title("🛠️ 模型選擇與資訊")

# 模型選單
model_option = st.sidebar.selectbox(
    "請選擇預測模型:",
    ("KNN", "Logistic Regression", "XGBoost", "Random Forest")
)

# 資料集資訊
st.sidebar.write("---")
st.sidebar.subheader("🍷 酒類資料集資訊")
st.sidebar.write(f"**特徵數量:** {len(wine_data.feature_names)}")
st.sidebar.write(f"**資料總筆數:** {len(df_wine)}")
st.sidebar.write(f"**分類類別:** {', '.join(wine_data.target_names)}")
st.sidebar.write("**描述:** 該資料集包含了對義大利同一地區生長但來自三個不同品種類型的酒的化學分析結果。")

# 3. 主區域 (Main Area)
st.title("🍷 酒類 (Wine) 資料集分類預測")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 前 5 筆資料預覽")
    st.dataframe(df_wine.head())

with col2:
    st.subheader("📊 特徵統計資訊")
    st.dataframe(df_wine.describe())

# 4. 按鈕預測與結果顯示
st.write("---")
st.subheader(f"🚀 使用 {model_option} 進行預測")

if st.button("開始執行預測"):
    # 準備資料
    X = df_wine.drop('target', axis=1)
    y = df_wine['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 對應模型檔案名稱 (偵測到目錄中已有這些檔案)
    model_files = {
        "KNN": "KNN_model.joblib",
        "Logistic Regression": "Logistic_Regression_model.joblib",
        "XGBoost": "XGBoost_model.joblib",
        "Random Forest": "Random_Forest_model.joblib"
    }
    
    model_path = model_files.get(model_option)
    full_path = os.path.join("c:/Users/User/Downloads/aipro01", model_path)
    
    if os.path.exists(full_path):
        try:
            # 載入現有模型
            model = joblib.load(full_path)
            
            # 執行預測
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            # 顯示結果
            st.success(f"🎉 預測完成！撰用模型: **{model_option}**")
            
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("準確度 (Accuracy)", f"{acc:.2%}")
            
            st.write("**預測結果 (測試集前10筆):**")
            comparison_df = pd.DataFrame({
                '實際分類': [wine_data.target_names[i] for i in y_test[:10]],
                '預測分類': [wine_data.target_names[int(i)] for i in y_pred[:10]]
            })
            st.table(comparison_df)
            
        except Exception as e:
            st.error(f"載入或運行模型時發生錯誤: {e}")
            st.info("提示：若模型檔案不完整，可能需要重新訓練後儲存。")
    else:
        st.error(f"找不到模型檔案: {model_path}")
        st.info("請確認模型檔案已放在正確的目錄中。")

# SEO 與頁尾
st.markdown("---")
st.caption("Developed by Antigravity AI | Streamlit + Scikit-learn Application")

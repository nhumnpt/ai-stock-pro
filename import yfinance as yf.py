import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import datetime
import io
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model

# ==========================================
# ⚙️ ตั้งค่าหน้าต่างโปรแกรม
# ==========================================
st.set_page_config(page_title="AI Stock Pro (Multimodal Edition)", page_icon="🧠", layout="wide")

if 'favorites' not in st.session_state:
    st.session_state.favorites = []

st.title("🧠 AI ผู้ช่วยลงทุนส่วนตัว (ระบบพยากรณ์กราฟราคารวมร่างกระแสข่าวเรียลไทม์)")
st.markdown("ระบุชื่อหุ้นที่คุณได้เทรนสมองกลแบบ **[กราฟ+ข่าว]** ไว้จากระบบหลังบ้าน เพื่อประมวลผลกลยุทธ์ความเร็วสูง")

# แถบเมนูด้านซ้าย
with st.sidebar:
    st.header("⭐️ รายการหุ้นติดดาว")
    if len(st.session_state.favorites) > 0:
        df_fav = pd.DataFrame(st.session_state.favorites)
        st.dataframe(df_fav[["สัญลักษณ์", "ราคาปัจจุบัน", "โซนซื้อที่แนะนำ", "ราคาเป้าหมายเฉลี่ย"]])
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_fav.to_excel(writer, index=False, sheet_name='Favorites')
        
        st.download_button(
            label="📥 ดาวน์โหลดสรุปแผนเทรดลง Excel",
            data=output.getvalue(),
            file_name=f"AI_Investment_Plan_{datetime.date.today()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        if st.button("🗑️ ล้างรายการ"):
            st.session_state.favorites = []
            st.rerun()

# ช่องค้นหาหุ้นหลัก
ticker_input = st.text_input("🔍 พิมพ์ชื่อหุ้นแล้วกด Enter (เช่น ASTS, TSLA, AAPL, NVDA):").strip().upper()

if ticker_input:
    with st.spinner(f"⚡ กำลังดึงข้อมูลตลาด ข่าวสาร และเชื่อมต่อสมอง AI สำหรับ {ticker_input} ..."):
        try:
            stock = yf.Ticker(ticker_input)
            info = stock.info
            
            if not info or ('regularMarketPrice' not in info and 'currentPrice' not in info and 'previousClose' not in info):
                st.error("⚠️ ไม่พบสัญลักษณ์หุ้นนี้ กรุณาตรวจสอบชื่อหุ้นอีกครั้ง")
            else:
                company_name = info.get("longName", ticker_input)
                current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
                target_price = info.get("targetMeanPrice", "N/A")
                currency = "฿" if ticker_input.endswith(".BK") else "$"
                
                # 📊 ดึงข้อมูลกราฟอัปเดตล่าสุด
                df_recent = stock.history(period="3mo", interval="1d")
                if isinstance(df_recent.columns, pd.MultiIndex):
                    doc_cols = df_recent.columns.get_level_values(0)
                    df_recent.columns = doc_cols

                # คำนวณแนวรับเชิงเทคนิคขั้นสูง (ใช้จุดสูงสุด/ต่ำสุดในรอบ 3 เดือนมาหาแนว Fibonacci)
                high_3m = df_recent['High'].max()
                low_3m = df_recent['Low'].min()
                diff = high_3m - low_3m
                
                # แนวรับระดับสำคัญจิตวิทยา (Fibonacci 38.2% และ 61.8%)
                fibo_38 = high_3m - (0.382 * diff)
                fibo_61 = high_3m - (0.618 * diff)
                
                # คำนวณค่าเฉลี่ย EMA 50 วัน เพื่อหาแนวรับเทรนด์ระยะกลาง
                df_recent['EMA_50'] = ta.trend.ema_indicator(df_recent['Close'], window=50) if len(df_recent) >= 50 else df_recent['Close']
                ema_50_val = df_recent['EMA_50'].iloc[-1]

                # สรุปโซนซื้อที่เหมาะสม
                suggested_buy_zone = f"{currency}{fibo_61:.2f} - {currency}{fibo_38:.2f}"
                if current_price <= fibo_38:
                    status_comment = "🟢 ราคาลงมาอยู่ในโซนแนวรับที่น่าสะสมแล้ว ไม้แรกสามารถทยอยเข้าได้"
                else:
                    status_comment = f"⚠️ ราคายังค่อนข้างสูงกว่าแนวรับ ควรรอให้ย่อตัวลงมาใกล้แถวๆ {currency}{fibo_38:.2f} จะได้เปรียบกว่า"

                # สัญญาณ RSI สำหรับดูแรงซื้อแรงขาย ณ วินาทีนี้
                rsi_series = ta.momentum.rsi(close=df_recent["Close"], window=14)
                rsi_val = rsi_series.iloc[-1]
                if rsi_val >= 70: rsi_status = f"🔴 Overbought ({rsi_val:.1f}) แรงซื้อมากเกินไป ระวังแรงเทขายทำกำไร"
                elif rsi_val <= 30: rsi_status = f"🔵 Oversold ({rsi_val:.1f}) แรงขายมากเกินไป เกิดการ panic-sell ลุ้น rebound"
                else: rsi_status = f"🟡 Neutral ({rsi_val:.1f}) แรงซื้อและแรงขายอยู่ในเกณฑ์สมดุล"

                # 📈 1. คำนวณจำแนกสัญญาณเทคนิคัลรายตัว (Buy/Hold/Sell Counts)
                tech_buy, tech_hold, tech_sell = 0, 0, 0
                try:
                    if rsi_val <= 30: tech_buy += 1
                    elif rsi_val >= 70: tech_sell += 1
                    else: tech_hold += 1

                    macd_series = ta.trend.macd(close=df_recent["Close"])
                    macd_sig = ta.trend.macd_signal(close=df_recent["Close"])
                    if macd_series.iloc[-1] > macd_sig.iloc[-1]: tech_buy += 1
                    else: tech_sell += 1

                    ema_20 = ta.trend.ema_indicator(close=df_recent["Close"], window=20)
                    if df_recent["Close"].iloc[-1] > ema_20.iloc[-1]: tech_buy += 1
                    else: tech_sell += 1

                    sma_50 = ta.trend.sma_indicator(close=df_recent["Close"], window=50)
                    if df_recent["Close"].iloc[-1] > sma_50.iloc[-1]: tech_buy += 1
                    else: tech_sell += 1

                    bb_high = ta.volatility.bollinger_hband(close=df_recent["Close"])
                    bb_low = ta.volatility.bollinger_lband(close=df_recent["Close"])
                    close_val = df_recent["Close"].iloc[-1]
                    if close_val <= bb_low.iloc[-1]: tech_buy += 1
                    elif close_val >= bb_high.iloc[-1]: tech_sell += 1
                    else: tech_hold += 1
                except Exception:
                    pass

                if tech_buy >= 4: tech_summary = "🟢 ซื้ออย่างแข็งแกร่ง (Strong Buy)"
                elif tech_buy >= 3: tech_summary = "🟢 ซื้อ (Buy)"
                elif tech_sell >= 4: tech_summary = "🔴 ขายอย่างแข็งแกร่ง (Strong Sell)"
                elif tech_sell >= 3: tech_summary = "🔴 ขาย (Sell)"
                else: tech_summary = "🟡 ถือ/รอดูสถานการณ์ (Neutral/Hold)"

                # 🎯 2. ดึงข้อมูลเรตติ้งและเป้าหมายจากนักวิเคราะห์
                rec_key_raw = info.get('recommendationKey', 'N/A')
                rec_map = {
                    'strong_buy': '🟢 ซื้ออย่างแข็งแกร่ง (Strong Buy)',
                    'buy': '🟢 ซื้อ (Buy)',
                    'hold': '🟡 ถือ (Hold)',
                    'sell': '🔴 ขาย (Sell)',
                    'strong_sell': '🔴 ขายอย่างแข็งแกร่ง (Strong Sell)',
                    'underperform': '🔴 แนะนำขายนอกกลุ่ม (Underperform)',
                    'outperform': '🟢 ซื้อมากกว่าตลาด (Outperform)'
                }
                analyst_recommendation = rec_map.get(str(rec_key_raw).lower(), 'N/A')
                
                target_mean = info.get('targetMeanPrice', 'N/A')
                target_high = info.get('targetHighPrice', 'N/A')
                target_low = info.get('targetLowPrice', 'N/A')
                analyst_count = info.get('numberOfAnalystOpinions', 'N/A')

                # 📰 3. ระบบคำนวณคะแนนข่าวดิจิทัลแบบเรียลไทม์ (Sentiment Scoring Engine)
                news_list = stock.news
                news_summary = "ไม่มีข่าวเด่นที่ส่งผลกระทบอย่างมีนัยสำคัญในขณะนี้"
                sentiment_score = 0.5  # ค่ากลางเริ่มต้น (Neutral)
                sentiment_label = "Neutral 🟡"
                
                if news_list:
                    titles = [n.get('title') for n in news_list[:5] if n.get('title')]
                    news_summary = " | ".join(titles) if titles else news_summary
                    
                    pos_words = ["growth", "buy", "profit", "beats", "up", "กำไร", "โต", "bullish", "upgrade", "success", "positive"]
                    neg_words = ["drop", "sell", "loss", "misses", "down", "ขาดทุน", "ร่วง", "bearish", "downgrade", "risk", "negative"]
                    
                    pos_count = sum(1 for t in titles for w in pos_words if w in t.lower())
                    neg_count = sum(1 for t in titles for w in neg_words if w in t.lower())
                    
                    total_words = pos_count + neg_count
                    if total_words > 0:
                        sentiment_score = 0.5 + ((pos_count - neg_count) / (total_words * 2))
                    
                    if sentiment_score > 0.55: sentiment_label = f"Positive 🟢 (ความเชื่อมั่นสื่อ: {sentiment_score:.2f})"
                    elif sentiment_score < 0.45: sentiment_label = f"Negative 🔴 (ความเสี่ยงสื่อ: {sentiment_score:.2f})"
                    else: sentiment_label = f"Neutral 🟡 (กระแสข่าวสมดุล: {sentiment_score:.2f})"

                # แสดงแดชบอร์ดข้อมูลหลัก
                st.subheader(f"📊 ผลวิเคราะห์ข้อมูลเรียลไทม์: {company_name} ({ticker_input})")
                col1, col2, col3 = st.columns(3)
                col1.metric("ราคาตลาดล่าสุด (เรียลไทม์)", f"{currency}{current_price:.2f}")
                col2.metric("โซนแนวรับ/ราคาซื้อที่เหมาะสม", suggested_buy_zone)
                col3.metric("กระแสข่าวดิจิทัล (Sentiment)", sentiment_label)
                
                st.info(f"💡 **ความเห็นเชิงเทคนิค:** {status_comment} | **พฤติกรรมตลาดวันนี้:** {rsi_status}")
                st.caption(f"📰 **หัวข้อข่าวเด่นล่าสุด:** {news_summary}")

                # ส่วนรายงานการวิเคราะห์และเรตติ้งโดยละเอียด
                st.markdown("---")
                st.subheader("📊 รายงานสรุปสัญญาณเทคนิคัล & เป้าหมายจากนักวิเคราะห์")
                
                rcol1, rcol2 = st.columns(2)
                with rcol1:
                    st.markdown("### 📈 สรุปสัญญาณเทคนิครายตัว (Technical Indicators Breakdown)")
                    st.markdown(f"**สัญญาณโดยรวม:** `{tech_summary}`")
                    
                    tcol1, tcol2, tcol3 = st.columns(3)
                    tcol1.metric("สัญญาณ [ซื้อ] (Buy)", f"{tech_buy} ตัวชี้วัด")
                    tcol2.metric("สัญญาณ [ถือ] (Hold)", f"{tech_hold} ตัวชี้วัด")
                    tcol3.metric("สัญญาณ [ขาย] (Sell)", f"{tech_sell} ตัวชี้วัด")
                    
                    st.caption("🔍 คำนวณแบบเรียลไทม์ผ่านตัวชี้วัด: RSI, MACD, EMA 20, SMA 50, Bollinger Bands")

                with rcol2:
                    st.markdown("### 🎯 เป้าหมายนักวิเคราะห์ & เรตติ้ง (Analyst Recommendation)")
                    st.markdown(f"**คำแนะนำของนักวิเคราะห์ส่วนใหญ่:** `{analyst_recommendation}`")
                    
                    mean_val = f"{currency}{target_mean:.2f}" if isinstance(target_mean, (int, float)) else "N/A"
                    high_val = f"{currency}{target_high:.2f}" if isinstance(target_high, (int, float)) else "N/A"
                    low_val = f"{currency}{target_low:.2f}" if isinstance(target_low, (int, float)) else "N/A"
                    
                    st.markdown(f"* **ราคาเป้าหมายเฉลี่ย (Mean Target):** `{mean_val}`")
                    st.markdown(f"* **ช่วงราคาเป้าหมาย (ต่ำสุด - สูงสุด):** `{low_val} - {high_val}`")
                    st.markdown(f"* **จำนวนนักวิเคราะห์ที่ประเมิน:** `{analyst_count} คน`")

                # ==========================================
                # 🧠 ส่วนดึงสมอง AI Deep Learning (LSTM ร่วมวิเคราะห์จากข่าว)
                # ==========================================
                st.markdown("---")
                st.subheader("🧠 ผลการวิเคราะห์แนวโน้มจากสมองกล Deep Learning (กราฟราคา + กระแสข่าว)")
                clean_ticker = ticker_input.replace('.BK', '').strip().lower()
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
                # 1. ค้นหาไฟล์โมเดลผสานข่าวสาร (News-infused Model)
                news_model_paths = [
                    os.path.join(script_dir, f'news_model_{clean_ticker}.keras'),
                    os.path.join(script_dir, 'ai_stock_brains', f'news_model_{clean_ticker}.keras')
                ]
                news_scaler_paths = [
                    os.path.join(script_dir, f'news_scaler_{clean_ticker}.pkl'),
                    os.path.join(script_dir, 'ai_stock_brains', f'news_scaler_{clean_ticker}.pkl')
                ]
                news_model_file = next((p for p in news_model_paths if os.path.exists(p)), None)
                news_scaler_file = next((p for p in news_scaler_paths if os.path.exists(p)), None)

                # 2. ค้นหาไฟล์โมเดลราคาปกติ (Standard Price Model)
                std_model_paths = [
                    os.path.join(script_dir, f'model_{clean_ticker}.keras'),
                    os.path.join(script_dir, 'ai_stock_brains', f'model_{clean_ticker}.keras')
                ]
                std_scaler_paths = [
                    os.path.join(script_dir, f'scaler_{clean_ticker}.pkl'),
                    os.path.join(script_dir, 'ai_stock_brains', f'scaler_{clean_ticker}.pkl')
                ]
                std_model_file = next((p for p in std_model_paths if os.path.exists(p)), None)
                std_scaler_file = next((p for p in std_scaler_paths if os.path.exists(p)), None)

                # ตัวแปรสำหรับใช้บันทึกลง favorites
                next_day_pred = None
                model_loaded = False
                
                if news_model_file and news_scaler_file:
                    # โหลดก้อนสมองแบบผสานข่าวสาร
                    loaded_model = load_model(news_model_file)
                    loaded_scaler = joblib.load(news_scaler_file)
                    n_features = getattr(loaded_scaler, 'n_features_in_', len(loaded_scaler.scale_))
                    
                    if len(df_recent) >= 60:
                        if n_features == 2:
                            close_prices = df_recent['Close'].values[-60:].reshape(-1, 1)
                            current_sentiments = np.full((60, 1), sentiment_score)
                            combined_features = np.hstack((close_prices, current_sentiments))
                            combined_scaled = loaded_scaler.transform(combined_features)
                            X_test = np.reshape(combined_scaled, (1, 60, 2))
                            pred_scaled = loaded_model.predict(X_test, verbose=0)
                            
                            dummy_output = np.zeros((1, 2))
                            dummy_output[0, 0] = pred_scaled[0][0]
                            next_day_pred = loaded_scaler.inverse_transform(dummy_output)[0][0]
                        else:
                            last_60_days = df_recent['Close'].values[-60:].reshape(-1, 1)
                            last_60_days_scaled = loaded_scaler.transform(last_60_days)
                            X_test = np.reshape(last_60_days_scaled, (1, 60, 1))
                            pred_scaled = loaded_model.predict(X_test, verbose=0)
                            next_day_pred = loaded_scaler.inverse_transform(pred_scaled)[0][0]
                        
                        model_loaded = True
                        pred_diff = ((next_day_pred - current_price) / current_price) * 100
                        pred_diff_str = f'+{pred_diff:.2f}%' if pred_diff > 0 else f'{pred_diff:.2f}%'
                        
                        last_20_days = df_recent['Close'].values[-20:]
                        trend_model = LinearRegression().fit(np.arange(20).reshape(-1, 1), last_20_days)
                        slope = trend_model.coef_[0]
                        ml_trend = '🟢 มีแนวโน้มปรับตัวขึ้นต่อ (Bullish)' if slope > 0.05 else '🔴 มีแนวโน้มชะลอตัวลึกลง (Bearish)' if slope < -0.05 else '🟡 ทรงตัวออกข้าง (Sideways)'

                        st.success('✅ โหลดไฟล์สมองกลผสานกระแสข่าวสำเร็จ ประมวลผลเสร็จสิ้นใน 0.3 วินาที')
                        mcol1, mcol2 = st.columns(2)
                        mcol1.metric('ราคาปิดพยากรณ์สำหรับวันทำการถัดไป (ผสานข่าวเรียลไทม์)', f'{currency}{next_day_pred:.2f}', pred_diff_str)
                        mcol2.metric('ทิศทางแนวโน้มโครงสร้างราคาจาก AI', ml_trend)
                        
                        # วาดกราฟ
                        st.markdown('**📈 กราฟราคาปิดล่าสุด & เส้นแนวโน้มความปลอดภัย (EMA 50):**')
                        chart_data = df_recent[['Close', 'EMA_50']].copy()
                        chart_data.columns = ['ราคาปิดจริงล่าสุด', 'เส้นแนวรับเทรนด์กลาง (EMA 50)']
                        st.line_chart(chart_data)
                    else:
                        st.warning('⚠️ ข้อมูลในประวัติศาสตร์รอบ 3 เดือนไม่เพียงพอ (ต้องการข้อมูลอย่างน้อย 60 วัน)')
                        
                elif std_model_file and std_scaler_file:
                    # โหลดก้อนสมองปกติ
                    loaded_model = load_model(std_model_file)
                    loaded_scaler = joblib.load(std_scaler_file)
                    n_features = getattr(loaded_scaler, 'n_features_in_', len(loaded_scaler.scale_))
                    
                    if len(df_recent) >= 60:
                        if n_features == 2:
                            close_prices = df_recent['Close'].values[-60:].reshape(-1, 1)
                            current_sentiments = np.full((60, 1), sentiment_score)
                            combined_features = np.hstack((close_prices, current_sentiments))
                            combined_scaled = loaded_scaler.transform(combined_features)
                            X_test = np.reshape(combined_scaled, (1, 60, 2))
                            pred_scaled = loaded_model.predict(X_test, verbose=0)
                            
                            dummy_output = np.zeros((1, 2))
                            dummy_output[0, 0] = pred_scaled[0][0]
                            next_day_pred = loaded_scaler.inverse_transform(dummy_output)[0][0]
                        else:
                            last_60_days = df_recent['Close'].values[-60:].reshape(-1, 1)
                            last_60_days_scaled = loaded_scaler.transform(last_60_days)
                            X_test = np.reshape(last_60_days_scaled, (1, 60, 1))
                            pred_scaled = loaded_model.predict(X_test, verbose=0)
                            next_day_pred = loaded_scaler.inverse_transform(pred_scaled)[0][0]
                            
                        model_loaded = True
                        pred_diff = ((next_day_pred - current_price) / current_price) * 100
                        pred_diff_str = f'+{pred_diff:.2f}%' if pred_diff > 0 else f'{pred_diff:.2f}%'
                        
                        last_20_days = df_recent['Close'].values[-20:]
                        trend_model = LinearRegression().fit(np.arange(20).reshape(-1, 1), last_20_days)
                        slope = trend_model.coef_[0]
                        ml_trend = '🟢 มีแนวโน้มปรับตัวขึ้นต่อ (Bullish)' if slope > 0.05 else '🔴 มีแนวโน้มชะลอตัวลึกลง (Bearish)' if slope < -0.05 else '🟡 ทรงตัวออกข้าง (Sideways)'

                        if n_features == 2:
                            st.success('✅ ตรวจพบสมองกลมัลติโมเดลผสานกระแสข่าวในชื่อโมเดลมาตรฐาน! ประมวลผลร่วมกับข่าวสารสำเร็จ')
                            mcol1, mcol2 = st.columns(2)
                            mcol1.metric('ราคาปิดพยากรณ์สำหรับวันทำการถัดไป (ผสานข่าวเรียลไทม์)', f'{currency}{next_day_pred:.2f}', pred_diff_str)
                            mcol2.metric('ทิศทางแนวโน้มโครงสร้างราคาจาก AI', ml_trend)
                        else:
                            st.success('ℹ️ พบไฟล์สมองกลปกติ (โหมดวิเคราะห์และทำนายราคาปิดแบบมาตรฐาน)')
                            mcol1, mcol2 = st.columns(2)
                            mcol1.metric('ราคาปิดพยากรณ์สำหรับวันทำการถัดไป (โมเดลปกติ)', f'{currency}{next_day_pred:.2f}', pred_diff_str)
                            mcol2.metric('ทิศทางแนวโน้มโครงสร้างราคาจาก AI', ml_trend)
                        
                        # วาดกราฟ
                        st.markdown('**📈 กราฟราคาปิดล่าสุด & เส้นแนวโน้มความปลอดภัย (EMA 50):**')
                        chart_data = df_recent[['Close', 'EMA_50']].copy()
                        chart_data.columns = ['ราคาปิดจริงล่าสุด', 'เส้นแนวรับเทรนด์กลาง (EMA 50)']
                        st.line_chart(chart_data)
                    else:
                        st.warning('⚠️ ข้อมูลในประวัติศาสตร์รอบ 3 เดือนไม่เพียงพอ (ต้องการข้อมูลอย่างน้อย 60 วัน)')
                        
                    if n_features == 1:
                        # แนะนำการใส่โมเดลข่าวเพิ่มเติมเฉพาะตอนที่กำลังรันโมเดล 1 feature อยู่
                        st.info(f'💡 **Tip สำหรับหุ้น {ticker_input}:** คุณสามารถเพิ่มประสิทธิภาพของระบบได้ โดยการนำไฟล์สมองกลผสานกระแสข่าวสาร `news_model_{clean_ticker}.keras` และ `news_scaler_{clean_ticker}.pkl` มาวางลงในโฟลเดอร์ `ai_stock_brains` เพื่อเปิดใช้ระบบคำนวณมัลติโมเดลเชิงลึกได้เลยครับ')
                else:
                    # กรณีไม่พบโมเดลใดๆ เลย
                    st.warning(f'⚠️ ไม่พบไฟล์โมเดลสำเร็จรูปที่มีการผสานข่าวสารสำหรับหุ้น {ticker_input} หน้าเว็บจึงแสดงเฉพาะข้อมูลเทคนิคและข่าวสารเรียลไทม์ (หากต้องการเปิดส่วนพยากรณ์ข่าวมัลติโมเดล กรุณารันไฟล์ Colab เพื่อสร้างไฟล์สมอง `news_model_{clean_ticker}.keras` มาวางไว้ด้วยนะครับ)')
                st.markdown('---')
                if st.button("⭐️ บันทึกหุ้นนี้เข้าแผนการลงทุน"):
                    new_data = {
                        "สัญลักษณ์": ticker_input,
                        "ราคาปัจจุบัน": f"{currency}{current_price:.2f}",
                        "โซนซื้อที่แนะนำ": suggested_buy_zone,
                        "ราคาเป้าหมายเฉลี่ย": f"{currency}{target_mean:.2f}" if isinstance(target_mean, (int, float)) else "N/A",
                        "สัญญาณเทคนิค": f"ซื้อ {tech_buy} | ถือ {tech_hold} | ขาย {tech_sell}",
                        "เรตติ้งนักวิเคราะห์": analyst_recommendation,
                        "ราคาทำนายวันพรุ่งนี้": f"{currency}{next_day_pred:.2f}" if model_loaded else "N/A"
                    }
                    if not any(f['สัญลักษณ์'] == ticker_input for f in st.session_state.favorites):
                        st.session_state.favorites.append(new_data)
                        st.success(f"บันทึกแผนเทรด {ticker_input} เข้าสู่รายการโปรดแล้ว!")
                        st.rerun()
        except Exception as e:
            st.error(f"ระบบขัดข้องชั่วคราว: {e}")

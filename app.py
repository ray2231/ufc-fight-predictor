import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    # Coba joblib dulu, fallback ke pickle
    try:
        model = joblib.load('ufc_fight_model.pkl')
    except:
        with open('ufc_fight_model.pkl', 'rb') as f:
            model = pickle.load(f)

    try:
        scaler = joblib.load('ufc_scaler.pkl')
    except:
        with open('ufc_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

    with open('feature_cols.pkl', 'rb') as f:
        feat_cols = pickle.load(f)
    fighters = pd.read_csv('fighter_database.csv')
    with open('model_performance.pkl', 'rb') as f:
        perf = pickle.load(f)
    return model, scaler, feat_cols, fighters, perf

model, scaler, feat_cols, fighters, perf = load_model()

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="🥊 UFC Fight Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 { color: #e94560; font-size: 3rem; }
    .main-header p { color: #ffffff; font-size: 1.2rem; }
    .winner-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        border: 2px solid #e94560;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #e94560, #c23152);
        color: white;
        font-size: 1.3rem;
        font-weight: bold;
        padding: 0.8rem;
        border-radius: 10px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("""
<div class="main-header">
    <h1>🥊 UFC FIGHT PREDICTOR</h1>
    <p>AI-Powered Fight Outcome Prediction | Ultimate Edition</p>
</div>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## 📊 Model Performance")
    st.metric("🎯 Accuracy", f"{perf['accuracy']:.1%}")
    st.metric("📈 AUC Score", f"{perf['auc']:.4f}")
    st.metric("🔄 Walk-Forward AUC", f"{perf.get('walk_forward_auc', 0):.4f}")
    st.markdown("---")
    st.metric("👤 Total Fighters", f"{perf['n_fighters']:,}")
    st.metric("🥊 Training Fights", f"{perf['n_training_fights']:,}")
    st.metric("📋 Features Used", perf['n_features'])
    st.markdown("---")
    st.markdown("""
    ### ⚠️ Disclaimer
    This tool is for **analysis & entertainment** only.
    - Never bet more than you can afford to lose
    - Past performance ≠ future results
    """)

# ========== HELPER: GET FIGHTER STAT ==========
def safe_get(fighter_row, col, default=0):
    """Safely get a value from fighter row"""
    try:
        val = fighter_row.get(col, default)
        if pd.isna(val):
            return default
        return float(val)
    except:
        return default

# ========== HELPER: BUILD ALL FEATURES ==========
def build_matchup_features(f1, f2):
    """Build ALL matchup features matching the trained model"""

    matchup = {
        # === RANKING ===
        'rank_diff': safe_get(f2, 'current_rank', 99) - safe_get(f1, 'current_rank', 99),

        # === CAREER ===
        'fight_exp_diff': safe_get(f1, 'total_fights') - safe_get(f2, 'total_fights'),
        'winrate_diff': safe_get(f1, 'win_rate') - safe_get(f2, 'win_rate'),
        'f1_win_rate': safe_get(f1, 'win_rate'),
        'f2_win_rate': safe_get(f2, 'win_rate'),
        'finish_rate_diff': safe_get(f1, 'finish_rate') - safe_get(f2, 'finish_rate'),
        'ko_rate_diff': safe_get(f1, 'ko_rate') - safe_get(f2, 'ko_rate'),
        'sub_rate_diff': safe_get(f1, 'sub_rate') - safe_get(f2, 'sub_rate'),
        'streak_diff': safe_get(f1, 'current_streak') - safe_get(f2, 'current_streak'),
        'f1_streak': safe_get(f1, 'current_streak'),
        'f2_streak': safe_get(f2, 'current_streak'),
        'best_streak_diff': safe_get(f1, 'best_win_streak') - safe_get(f2, 'best_win_streak'),

        # === STRIKING ===
        'slpm_diff': safe_get(f1, 'slpm', 3.0) - safe_get(f2, 'slpm', 3.0),
        'f1_slpm': safe_get(f1, 'slpm', 3.0),
        'f2_slpm': safe_get(f2, 'slpm', 3.0),
        'str_acc_diff': safe_get(f1, 'str_acc', 0.45) - safe_get(f2, 'str_acc', 0.45),
        'f1_str_acc': safe_get(f1, 'str_acc', 0.45),
        'f2_str_acc': safe_get(f2, 'str_acc', 0.45),
        'sapm_diff': safe_get(f1, 'sapm', 3.0) - safe_get(f2, 'sapm', 3.0),
        'f1_sapm': safe_get(f1, 'sapm', 3.0),
        'f2_sapm': safe_get(f2, 'sapm', 3.0),

        # === STRIKING DEFENSE ===
        'str_def_diff': safe_get(f1, 'str_def', 0.5) - safe_get(f2, 'str_def', 0.5),
        'f1_str_def': safe_get(f1, 'str_def', 0.5),
        'f2_str_def': safe_get(f2, 'str_def', 0.5),

        # === STRIKING EFFICIENCY ===
        'f1_strike_eff': safe_get(f1, 'slpm', 3.0) - safe_get(f1, 'sapm', 3.0),
        'f2_strike_eff': safe_get(f2, 'slpm', 3.0) - safe_get(f2, 'sapm', 3.0),
        'strike_eff_diff': (safe_get(f1, 'slpm', 3.0) - safe_get(f1, 'sapm', 3.0)) - \
                           (safe_get(f2, 'slpm', 3.0) - safe_get(f2, 'sapm', 3.0)),

        # === GRAPPLING ===
        'td_avg_diff': safe_get(f1, 'td_avg', 1.0) - safe_get(f2, 'td_avg', 1.0),
        'f1_td_avg': safe_get(f1, 'td_avg', 1.0),
        'f2_td_avg': safe_get(f2, 'td_avg', 1.0),
        'td_acc_diff': safe_get(f1, 'td_acc', 0.4) - safe_get(f2, 'td_acc', 0.4),
        'f1_td_acc': safe_get(f1, 'td_acc', 0.4),
        'f2_td_acc': safe_get(f2, 'td_acc', 0.4),
        'td_def_diff': safe_get(f1, 'td_def', 0.6) - safe_get(f2, 'td_def', 0.6),
        'f1_td_def': safe_get(f1, 'td_def', 0.6),
        'f2_td_def': safe_get(f2, 'td_def', 0.6),

        # === SUBMISSION ===
        'sub_avg_diff': safe_get(f1, 'sub_avg', 0.5) - safe_get(f2, 'sub_avg', 0.5),
        'f1_sub_avg': safe_get(f1, 'sub_avg', 0.5),
        'f2_sub_avg': safe_get(f2, 'sub_avg', 0.5),

        # === MATCHUP SPECIFIC ===
        'f1_grapple_vs_f2_td_def': safe_get(f1, 'td_avg', 1.0) * (1 - safe_get(f2, 'td_def', 0.6)),
        'f2_grapple_vs_f1_td_def': safe_get(f2, 'td_avg', 1.0) * (1 - safe_get(f1, 'td_def', 0.6)),
        'grapple_advantage': (safe_get(f1, 'td_avg', 1.0) * (1 - safe_get(f2, 'td_def', 0.6))) - \
                             (safe_get(f2, 'td_avg', 1.0) * (1 - safe_get(f1, 'td_def', 0.6))),

        'f1_strike_vs_f2_def': safe_get(f1, 'slpm', 3.0) * safe_get(f1, 'str_acc', 0.45) * (1 - safe_get(f2, 'str_def', 0.5)),
        'f2_strike_vs_f1_def': safe_get(f2, 'slpm', 3.0) * safe_get(f2, 'str_acc', 0.45) * (1 - safe_get(f1, 'str_def', 0.5)),
        'strike_advantage': (safe_get(f1, 'slpm', 3.0) * safe_get(f1, 'str_acc', 0.45) * (1 - safe_get(f2, 'str_def', 0.5))) - \
                            (safe_get(f2, 'slpm', 3.0) * safe_get(f2, 'str_acc', 0.45) * (1 - safe_get(f1, 'str_def', 0.5))),

        # === PHYSICAL ===
        'height_diff': safe_get(f1, 'height', 70) - safe_get(f2, 'height', 70),
        'reach_diff': safe_get(f1, 'reach', 70) - safe_get(f2, 'reach', 70),
        'f1_height': safe_get(f1, 'height', 70),
        'f2_height': safe_get(f2, 'height', 70),
        'f1_reach': safe_get(f1, 'reach', 70),
        'f2_reach': safe_get(f2, 'reach', 70),
        'age_diff': safe_get(f1, 'age', 30) - safe_get(f2, 'age', 30),
        'f1_age': safe_get(f1, 'age', 30),
        'f2_age': safe_get(f2, 'age', 30),
        'stance_diff': abs(safe_get(f1, 'stance', 0) - safe_get(f2, 'stance', 0)),
    }

    return matchup

# ========== TABS ==========
tab1, tab2, tab3 = st.tabs(["🥊 Fight Predictor", "📊 Fighter Database", "📈 Model Info"])

# ==========================================
# TAB 1: FIGHT PREDICTOR
# ==========================================
with tab1:
    fighter_names = sorted(fighters['fighter'].unique())

    col1, col_vs, col2 = st.columns([2, 1, 2])

    with col1:
        st.markdown("### 🔴 RED CORNER")
        f1_name = st.selectbox("Select Fighter", fighter_names, key="f1", index=0)
        f1_data = fighters[fighters['fighter'] == f1_name].iloc[0]
        st.markdown(f"""
        **Record:** {safe_get(f1_data, 'wins'):.0f}W - {safe_get(f1_data, 'losses'):.0f}L |
        **Win Rate:** {safe_get(f1_data, 'win_rate'):.1%}
        """)
        with st.expander("📊 Full Stats"):
            st.markdown(f"""
            | Stat | Value |
            |------|-------|
            | Finish Rate | {safe_get(f1_data, 'finish_rate'):.1%} |
            | KO Rate | {safe_get(f1_data, 'ko_rate'):.1%} |
            | Sub Rate | {safe_get(f1_data, 'sub_rate'):.1%} |
            | Strikes/Min | {safe_get(f1_data, 'slpm', 0):.1f} |
            | Strike Acc | {safe_get(f1_data, 'str_acc', 0):.1%} |
            | Strike Def | {safe_get(f1_data, 'str_def', 0):.1%} |
            | TD Avg/15min | {safe_get(f1_data, 'td_avg', 0):.1f} |
            | TD Accuracy | {safe_get(f1_data, 'td_acc', 0):.1%} |
            | TD Defense | {safe_get(f1_data, 'td_def', 0):.1%} |
            | Sub Avg/15min | {safe_get(f1_data, 'sub_avg', 0):.1f} |
            | Streak | {safe_get(f1_data, 'current_streak'):.0f} |
            """)

    with col_vs:
        st.markdown("<h1 style='text-align:center; padding-top:80px; color:#e94560;'>VS</h1>",
                     unsafe_allow_html=True)

    with col2:
        st.markdown("### 🔵 BLUE CORNER")
        available = [f for f in fighter_names if f != f1_name]
        f2_name = st.selectbox("Select Fighter", available, key="f2",
                                index=min(1, len(available)-1))
        f2_data = fighters[fighters['fighter'] == f2_name].iloc[0]
        st.markdown(f"""
        **Record:** {safe_get(f2_data, 'wins'):.0f}W - {safe_get(f2_data, 'losses'):.0f}L |
        **Win Rate:** {safe_get(f2_data, 'win_rate'):.1%}
        """)
        with st.expander("📊 Full Stats"):
            st.markdown(f"""
            | Stat | Value |
            |------|-------|
            | Finish Rate | {safe_get(f2_data, 'finish_rate'):.1%} |
            | KO Rate | {safe_get(f2_data, 'ko_rate'):.1%} |
            | Sub Rate | {safe_get(f2_data, 'sub_rate'):.1%} |
            | Strikes/Min | {safe_get(f2_data, 'slpm', 0):.1f} |
            | Strike Acc | {safe_get(f2_data, 'str_acc', 0):.1%} |
            | Strike Def | {safe_get(f2_data, 'str_def', 0):.1%} |
            | TD Avg/15min | {safe_get(f2_data, 'td_avg', 0):.1f} |
            | TD Accuracy | {safe_get(f2_data, 'td_acc', 0):.1%} |
            | TD Defense | {safe_get(f2_data, 'td_def', 0):.1%} |
            | Sub Avg/15min | {safe_get(f2_data, 'sub_avg', 0):.1f} |
            | Streak | {safe_get(f2_data, 'current_streak'):.0f} |
            """)

    st.markdown("")

    # ========== PREDICT ==========
    if st.button("🥊 PREDICT FIGHT OUTCOME", use_container_width=True, type="primary"):

        f1 = fighters[fighters['fighter'] == f1_name].iloc[0]
        f2 = fighters[fighters['fighter'] == f2_name].iloc[0]

        # Build features using helper function
        matchup = build_matchup_features(f1, f2)

        # Predict
        X_pred = pd.DataFrame([matchup])

        # Ensure column order matches training
        missing_cols = [c for c in feat_cols if c not in X_pred.columns]
        for c in missing_cols:
            X_pred[c] = 0

        X_pred = X_pred[feat_cols]
        X_pred_scaled = scaler.transform(X_pred)

        prob = model.predict_proba(X_pred_scaled)[0]
        f1_prob = prob[1]
        f2_prob = prob[0]

        winner = f1_name if f1_prob > 0.5 else f2_name
        confidence = max(f1_prob, f2_prob)

        if confidence > 0.75:
            conf_emoji, conf_text = "🟢", "HIGH CONFIDENCE"
        elif confidence > 0.60:
            conf_emoji, conf_text = "🟡", "MEDIUM CONFIDENCE"
        else:
            conf_emoji, conf_text = "🔴", "LOW CONFIDENCE - TOSS UP"

        # Winner display
        st.markdown("---")
        st.markdown(f"""
        <div class="winner-box">
            <h2>{conf_emoji} PREDICTED WINNER</h2>
            <h1 style="color: #e94560; font-size: 2.5rem;">🏆 {winner}</h1>
            <h3>Confidence: {confidence:.1%} ({conf_text})</h3>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### 🔴 {f1_name}: {f1_prob:.1%}")
            st.progress(f1_prob)
        with col2:
            st.markdown(f"#### 🔵 {f2_name}: {f2_prob:.1%}")
            st.progress(f2_prob)

        # Tale of the tape
        st.markdown("### 📊 Tale of the Tape")

        tape = pd.DataFrame({
            'Stat': ['Win Rate', 'Finish Rate', 'KO Rate', 'Sub Rate',
                     'Strikes/Min', 'Str Accuracy', 'Str Defense',
                     'TD Avg/15min', 'TD Accuracy', 'TD Defense',
                     'Sub Avg/15min', 'Strike Efficiency',
                     'Current Streak', 'Total Fights'],
            f1_name: [
                f"{safe_get(f1, 'win_rate'):.1%}",
                f"{safe_get(f1, 'finish_rate'):.1%}",
                f"{safe_get(f1, 'ko_rate'):.1%}",
                f"{safe_get(f1, 'sub_rate'):.1%}",
                f"{safe_get(f1, 'slpm', 0):.1f}",
                f"{safe_get(f1, 'str_acc', 0):.1%}",
                f"{safe_get(f1, 'str_def', 0):.1%}",
                f"{safe_get(f1, 'td_avg', 0):.1f}",
                f"{safe_get(f1, 'td_acc', 0):.1%}",
                f"{safe_get(f1, 'td_def', 0):.1%}",
                f"{safe_get(f1, 'sub_avg', 0):.1f}",
                f"{safe_get(f1, 'slpm', 0) - safe_get(f1, 'sapm', 0):.2f}",
                f"{safe_get(f1, 'current_streak'):.0f}",
                f"{safe_get(f1, 'total_fights'):.0f}",
            ],
            '🏆': [
                '→' if safe_get(f1,'win_rate') > safe_get(f2,'win_rate') else '←' if safe_get(f1,'win_rate') < safe_get(f2,'win_rate') else '=',
                '→' if safe_get(f1,'finish_rate') > safe_get(f2,'finish_rate') else '←' if safe_get(f1,'finish_rate') < safe_get(f2,'finish_rate') else '=',
                '→' if safe_get(f1,'ko_rate') > safe_get(f2,'ko_rate') else '←' if safe_get(f1,'ko_rate') < safe_get(f2,'ko_rate') else '=',
                '→' if safe_get(f1,'sub_rate') > safe_get(f2,'sub_rate') else '←' if safe_get(f1,'sub_rate') < safe_get(f2,'sub_rate') else '=',
                '→' if safe_get(f1,'slpm') > safe_get(f2,'slpm') else '←' if safe_get(f1,'slpm') < safe_get(f2,'slpm') else '=',
                '→' if safe_get(f1,'str_acc') > safe_get(f2,'str_acc') else '←' if safe_get(f1,'str_acc') < safe_get(f2,'str_acc') else '=',
                '→' if safe_get(f1,'str_def') > safe_get(f2,'str_def') else '←' if safe_get(f1,'str_def') < safe_get(f2,'str_def') else '=',
                '→' if safe_get(f1,'td_avg') > safe_get(f2,'td_avg') else '←' if safe_get(f1,'td_avg') < safe_get(f2,'td_avg') else '=',
                '→' if safe_get(f1,'td_acc') > safe_get(f2,'td_acc') else '←' if safe_get(f1,'td_acc') < safe_get(f2,'td_acc') else '=',
                '→' if safe_get(f1,'td_def') > safe_get(f2,'td_def') else '←' if safe_get(f1,'td_def') < safe_get(f2,'td_def') else '=',
                '→' if safe_get(f1,'sub_avg') > safe_get(f2,'sub_avg') else '←' if safe_get(f1,'sub_avg') < safe_get(f2,'sub_avg') else '=',
                '→' if (safe_get(f1,'slpm')-safe_get(f1,'sapm')) > (safe_get(f2,'slpm')-safe_get(f2,'sapm')) else '←',
                '→' if safe_get(f1,'current_streak') > safe_get(f2,'current_streak') else '←' if safe_get(f1,'current_streak') < safe_get(f2,'current_streak') else '=',
                '→' if safe_get(f1,'total_fights') > safe_get(f2,'total_fights') else '←' if safe_get(f1,'total_fights') < safe_get(f2,'total_fights') else '=',
            ],
            f2_name: [
                f"{safe_get(f2, 'win_rate'):.1%}",
                f"{safe_get(f2, 'finish_rate'):.1%}",
                f"{safe_get(f2, 'ko_rate'):.1%}",
                f"{safe_get(f2, 'sub_rate'):.1%}",
                f"{safe_get(f2, 'slpm', 0):.1f}",
                f"{safe_get(f2, 'str_acc', 0):.1%}",
                f"{safe_get(f2, 'str_def', 0):.1%}",
                f"{safe_get(f2, 'td_avg', 0):.1f}",
                f"{safe_get(f2, 'td_acc', 0):.1%}",
                f"{safe_get(f2, 'td_def', 0):.1%}",
                f"{safe_get(f2, 'sub_avg', 0):.1f}",
                f"{safe_get(f2, 'slpm', 0) - safe_get(f2, 'sapm', 0):.2f}",
                f"{safe_get(f2, 'current_streak'):.0f}",
                f"{safe_get(f2, 'total_fights'):.0f}",
            ]
        })
        st.table(tape.set_index('Stat'))

        # Key matchup insights
        st.markdown("### 🔍 Key Matchup Insights")

        insights = []
        if matchup.get('strike_advantage', 0) > 0.3:
            insights.append(f"🔴 **{f1_name}** has significant STRIKING advantage")
        elif matchup.get('strike_advantage', 0) < -0.3:
            insights.append(f"🔵 **{f2_name}** has significant STRIKING advantage")

        if matchup.get('grapple_advantage', 0) > 0.3:
            insights.append(f"🔴 **{f1_name}** has significant GRAPPLING advantage")
        elif matchup.get('grapple_advantage', 0) < -0.3:
            insights.append(f"🔵 **{f2_name}** has significant GRAPPLING advantage")

        if abs(matchup.get('reach_diff', 0)) > 3:
            longer = f1_name if matchup['reach_diff'] > 0 else f2_name
            insights.append(f"📏 **{longer}** has {abs(matchup['reach_diff']):.0f}\" reach advantage")

        if abs(matchup.get('age_diff', 0)) > 5:
            younger = f1_name if matchup['age_diff'] < 0 else f2_name
            insights.append(f"📅 **{younger}** is {abs(matchup['age_diff']):.0f} years younger")

        if abs(matchup.get('streak_diff', 0)) >= 3:
            hotter = f1_name if matchup['streak_diff'] > 0 else f2_name
            insights.append(f"🔥 **{hotter}** has stronger momentum (streak diff: {abs(matchup['streak_diff']):.0f})")

        if not insights:
            insights.append("⚖️ This is a closely matched fight!")

        for insight in insights:
            st.markdown(f"- {insight}")

# ==========================================
# TAB 2: FIGHTER DATABASE
# ==========================================
with tab2:
    st.markdown("### 👤 Fighter Database")

    search = st.text_input("🔍 Search fighter:")

    display_cols = ['fighter', 'total_fights', 'wins', 'losses', 'win_rate',
                    'ko_rate', 'sub_rate', 'finish_rate', 'current_streak']

    # Add new cols if they exist
    for col in ['slpm', 'str_acc', 'str_def', 'td_avg', 'td_acc', 'td_def', 'sub_avg']:
        if col in fighters.columns:
            display_cols.append(col)

    display_df = fighters[display_cols].copy()

    if search:
        display_df = display_df[display_df['fighter'].str.lower().str.contains(search.lower())]

    # Format percentages
    pct_cols = [c for c in display_df.columns if display_df[c].dtype == 'float64' and display_df[c].max() <= 1.0]
    for col in pct_cols:
        if col in ['win_rate', 'ko_rate', 'sub_rate', 'finish_rate', 'str_acc', 'str_def', 'td_acc', 'td_def']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    st.dataframe(display_df.sort_values('total_fights', ascending=False),
                  use_container_width=True, height=600)

    st.markdown(f"**Total fighters: {len(display_df)}**")

# ==========================================
# TAB 3: MODEL INFO
# ==========================================
with tab3:
    st.markdown("### 📈 Model Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", perf['model_name'])
    with col2:
        st.metric("Accuracy", f"{perf['accuracy']:.1%}")
    with col3:
        st.metric("AUC", f"{perf['auc']:.4f}")

    st.markdown(f"""
    ### 🔍 Features Used ({perf['n_features']} total)

    | Category | Features |
    |----------|----------|
    | Career | Win rate, Finish rate, KO rate, Sub rate, Streaks |
    | Striking | SLpM, Strike Accuracy, SApM, Strike Defense |
    | Grappling | TD Avg/15min, TD Accuracy, TD Defense |
    | Submission | Sub Avg/15min |
    | Physical | Height, Reach, Age, Stance |
    | Matchup | Strike advantage, Grapple advantage, Efficiency diffs |
    | Rankings | Current UFC ranking position |

    ### ✅ Validation
    - Time-based split (no data leakage)
    - Walk-Forward AUC: **{perf.get('walk_forward_auc', 0):.4f}**
    - Consistent across multiple time periods
    """)

# Footer
st.markdown("---")
st.markdown("⚠️ **DISCLAIMER:** Analysis tool only. Not financial advice. Use responsibly.")

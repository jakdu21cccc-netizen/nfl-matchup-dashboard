import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import plotly.express as px

# 1. 페이지 레이아웃 설정
st.set_page_config(layout="wide", page_title="NFL 팀 스탯 양자 비교", page_icon="🏈")

# 🌟 사이드바 가독성 문제 해결을 위한 CSS 보완 🌟
st.markdown("""
    <style>
    /* 전체 배경에 풋볼 필드 이미지와 85% 농도의 어두운 필터를 겹쳐서 적용 */
    .stApp {
        background-image: linear-gradient(rgba(15, 25, 30, 0.85), rgba(15, 25, 30, 0.85)), 
                          url("https://images.unsplash.com/photo-1566577739112-5180d4bf9390?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    
    /* 상단 여백 조정 */
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    
    /* 메인 화면의 텍스트들을 하얀색 계열로 강제 지정하여 가독성 확보 */
    h1, h2, h3, p, label {
        color: #F8FAFC !important;
    }
    
    /* ✅ 해결 파트: 왼쪽 사이드바 내부의 글자색만 진한 회색/검정으로 덮어쓰기 */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label {
        color: #333333 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏈 NFL Matchup: Team Stats Comparison")

# 2. 스탯 카테고리 정의
STAT_CATEGORIES = {
    "공격 스탯(경기당)": [
        "평균득점", "토탈평균획득야드", "플레이 당 획득 거리", "터치다운 수",
        "평균 러싱플레이 시도 횟수", "러싱 야드", "러싱 터치다운 수", "러싱 플레이당 거리",
        "평균 패싱플레이 시도 횟수", "패싱 야드", "패싱 터치다운 수", "패싱 플레이당 거리"
    ],
    "수비 스탯(경기당)": [
        "허용득점", "토탈허용평균야드", "플레이 당 허용 거리", "허용 터치다운 수",
        "허용 평균 러싱플레이 시도 횟수", "허용 러싱 야드", "허용 러싱 터치다운 수", "허용 러싱 플레이당 거리",
        "허용 평균 패싱플레이 시도 횟수", "허용 패싱 야드", "허용 패싱터치다운 수", "허용 패싱플레이당 거리"
    ],
    "턴오버 스탯(시즌 누적)": [
        "턴오버 마진", "턴오버 창출", "턴오버 허용", 
        "펌블 리커버 수", "인터셉트 수", "펌블 허용 수", "인터셉트 허용 수"
    ]
}

# 3. 데이터 로드 및 전처리
@st.cache_data(show_spinner="NFLverse 데이터를 분석하고 디자인 리소스를 불러오는 중입니다...", ttl=3600)
def get_nfl_data(season=2025):
    try:
        team_meta = nfl.import_team_desc()[['team_abbr', 'team_name', 'team_logo_espn', 'team_color']]
        
        sched = nfl.import_schedules([season])
        sched = sched[sched['game_type'] == 'REG']
        
        home_sched = sched.groupby('home_team').agg(경기수=('game_id','count'), 득점=('home_score','sum'), 실점=('away_score','sum'))
        away_sched = sched.groupby('away_team').agg(경기수=('game_id','count'), 득점=('away_score','sum'), 실점=('home_score','sum'))
        team_games = home_sched.add(away_sched, fill_value=0)
        
        pbp = nfl.import_pbp_data([season])
        pbp = pbp[(pbp['season_type'] == 'REG') & pbp['posteam'].notna()]
        
        off = pbp.groupby('posteam').agg(
            총플레이수=('play_id', 'count'), 총야드=('yards_gained', 'sum'), 패스시도=('pass_attempt', 'sum'), 패스야드=('passing_yards', 'sum'), 패스TD=('pass_touchdown', 'sum'),
            러싱시도=('rush_attempt', 'sum'), 러싱야드=('rushing_yards', 'sum'), 러싱TD=('rush_touchdown', 'sum'), 인터셉트허용=('interception', 'sum'), 펌블허용=('fumble_lost', 'sum')
        )
        def_pbp = pbp.groupby('defteam').agg(
            허용총플레이수=('play_id', 'count'), 허용총야드=('yards_gained', 'sum'), 허용패스시도=('pass_attempt', 'sum'), 허용패스야드=('passing_yards', 'sum'), 허용패스TD=('pass_touchdown', 'sum'),
            허용러싱시도=('rush_attempt', 'sum'), 허용러싱야드=('rushing_yards', 'sum'), 허용러싱TD=('rush_touchdown', 'sum'), 인터셉트유도=('interception', 'sum'), 펌블리커버리=('fumble_lost', 'sum')
        )
        
        stats = team_games.join(off).join(def_pbp).fillna(0)
        final_rows = []
        for team, row in stats.iterrows():
            g = max(row['경기수'], 1)
            off_map = {
                "평균득점": row['득점'] / g, "토탈평균획득야드": row['총야드'] / g, "플레이 당 획득 거리": row['총야드'] / max(row['총플레이수'], 1),
                "터치다운 수": (row['패스TD'] + row['러싱TD']) / g, "평균 러싱플레이 시도 횟수": row['러싱시도'] / g,
                "러싱 야드": row['러싱야드'] / g, "러싱 터치다운 수": row['러싱TD'] / g, "러싱 플레이당 거리": row['러싱야드'] / max(row['러싱시도'], 1),
                "평균 패싱플레이 시도 횟수": row['패스시도'] / g, "패싱 야드": row['패스야드'] / g,
                "패싱 터치다운 수": row['패스TD'] / g, "패싱 플레이당 거리": row['패스야드'] / max(row['패스시도'], 1)
            }
            for k, v in off_map.items(): final_rows.append({"팀명": team, "카테고리": "공격 스탯(경기당)", "스탯명": k, "수치": v})
                
            def_map = {
                "허용득점": row['실점'] / g, "토탈허용평균야드": row['허용총야드'] / g, "플레이 당 허용 거리": row['허용총야드'] / max(row['허용총플레이수'], 1),
                "허용 터치다운 수": (row['허용패스TD'] + row['허용러싱TD']) / g, "허용 평균 러싱플레이 시도 횟수": row['허용러싱시도'] / g,
                "허용 러싱 야드": row['허용러싱야드'] / g, "허용 러싱 터치다운 수": row['허용러싱TD'] / g, "허용 러싱 플레이당 거리": row['허용러싱야드'] / max(row['허용러싱시도'], 1),
                "허용 평균 패싱플레이 시도 횟수": row['허용패스시도'] / g, "허용 패싱 야드": row['허용패스야드'] / g,
                "허용 패싱터치다운 수": row['허용패스TD'] / g, "허용 패싱플레이당 거리": row['허용패스야드'] / max(row['허용패스시도'], 1)
            }
            for k, v in def_map.items(): final_rows.append({"팀명": team, "카테고리": "수비 스탯(경기당)", "스탯명": k, "수치": v})
                
            to_map = {
                "턴오버 마진": row['인터셉트유도'] + row['펌블리커버리'] - row['인터셉트허용'] - row['펌블허용'],
                "턴오버 창출": row['인터셉트유도'] + row['펌블리커버리'], "턴오버 허용": row['인터셉트허용'] + row['펌블허용'],
                "펌블 리커버 수": row['펌블리커버리'], "인터셉트 수": row['인터셉트유도'], "펌블 허용 수": row['펌블허용'], "인터셉트 허용 수": row['인터셉트허용']
            }
            for k, v in to_map.items(): final_rows.append({"팀명": team, "카테고리": "턴오버 스탯(시즌 누적)", "스탯명": k, "수치": v})

        df_result = pd.DataFrame(final_rows)
        rank_dfs = []
        for stat_name, group in df_result.groupby('스탯명'):
            cat = group['카테고리'].iloc[0]
            asc = False 
            if cat == "수비 스탯(경기당)": asc = True
            elif cat == "턴오버 스탯(시즌 누적)" and stat_name in ["턴오버 허용", "펌블 허용 수", "인터셉트 허용 수"]: asc = True
            group = group.copy()
            group['순위'] = group['수치'].rank(ascending=asc, method='min').astype(int)
            rank_dfs.append(group)
            
        return pd.concat(rank_dfs).reset_index(drop=True), team_meta
    except Exception as e:
        st.error(f"데이터 로드 중 에러가 발생했습니다: {e}")
        return pd.DataFrame(), pd.DataFrame()

# 4. 앱 구동 및 UI 렌더링
season_input = st.sidebar.number_input("시즌 선택", min_value=2020, max_value=2026, value=2025)
df_nfl, team_meta = get_nfl_data(season=season_input)

if not df_nfl.empty:
    all_teams = sorted(df_nfl['팀명'].unique())
    
    st.sidebar.header("⚙️ 대시보드 설정")
    team1 = st.sidebar.selectbox("첫 번째 팀 선택 (Home)", all_teams, index=all_teams.index('WAS') if 'WAS' in all_teams else 0)
    team2 = st.sidebar.selectbox("두 번째 팀 선택 (Away)", all_teams, index=all_teams.index('SEA') if 'SEA' in all_teams else 1)
    
    t1_info = team_meta[team_meta['team_abbr'] == team1].iloc[0] if not team_meta[team_meta['team_abbr'] == team1].empty else None
    t2_info = team_meta[team_meta['team_abbr'] == team2].iloc[0] if not team_meta[team_meta['team_abbr'] == team2].empty else None
    
    t1_color = f"#{str(t1_info['team_color']).strip('#')}" if t1_info is not None else "#1f77b4"
    t2_color = f"#{str(t2_info['team_color']).strip('#')}" if t2_info is not None else "#ff7f0e"

    st.write("---")
    
    col1, col2, col3 = st.columns([1, 0.5, 1])
    with col1:
        if t1_info is not None: 
            st.markdown(f"<div style='text-align: center;'><img src='{t1_info['team_logo_espn']}' width='150'></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='text-align: center; color: #E2E8F0; margin-top: 30px;'>VS</h1>", unsafe_allow_html=True)
    with col3:
        if t2_info is not None: 
            st.markdown(f"<div style='text-align: center;'><img src='{t2_info['team_logo_espn']}' width='150'></div>", unsafe_allow_html=True)
        
    st.write("---")
    
    category = st.sidebar.selectbox("스탯 대분류", list(STAT_CATEGORIES.keys()))
    all_stats = STAT_CATEGORIES[category]
    selected_stats = st.sidebar.multiselect("비교 대상 세부 지표 선택", options=all_stats, default=all_stats)
    
    if len(selected_stats) == 0:
        st.warning("⚠️ 지표를 하나 이상 선택하셔야 비교 데이터를 볼 수 있습니다.")
    else:
        filtered = df_nfl[(df_nfl['팀명'].isin([team1, team2])) & (df_nfl['카테고리'] == category) & (df_nfl['스탯명'].isin(selected_stats))]
        numeric_table = filtered.pivot(index='스탯명', columns='팀명', values='수치').reindex(selected_stats)
        
        st.subheader("💡 Key Metrics Highlight")
        
        def get_stat_info(s_name, t_name):
            val = filtered[(filtered['스탯명'] == s_name) & (filtered['팀명'] == t_name)]
            if not val.empty: return int(val.iloc[0]['순위']), val.iloc[0]['수치']
            return 0, 0
            
        def make_custom_kpi_card(stat_title, r1, r2, v1, v2, team1_name, team2_name, color1, color2, is_turnover):
            str_v1 = f"{int(v1)}" if is_turnover else f"{v1:.2f}"
            str_v2 = f"{int(v2)}" if is_turnover else f"{v2:.2f}"
            
            html_content = f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); text-align: center;">
                <h4 style="color: #333333; font-weight: 700; margin-top: 0; margin-bottom: 15px; letter-spacing: 0.5px;">{stat_title}</h4>
                <div style="display: flex; justify-content: space-around; align-items: center;">
                    <div style="flex: 1;">
                        <div style="font-size: 2.4rem; font-weight: 800; color: {color1};">{r1}위</div>
                        <div style="font-size: 0.9rem; color: #718096; margin-top: 5px; font-weight: 700;">{team1_name} ({str_v1})</div>
                    </div>
                    <div style="font-size: 1.2rem; font-weight: 900; color: #CBD5E0; padding: 0 10px;">VS</div>
                    <div style="flex: 1;">
                        <div style="font-size: 2.4rem; font-weight: 800; color: {color2};">{r2}위</div>
                        <div style="font-size: 0.9rem; color: #718096; margin-top: 5px; font-weight: 700;">{team2_name} ({str_v2})</div>
                    </div>
                </div>
            </div>
            """
            return html_content

        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        is_to_category = "턴오버" in category
        
        try:
            if len(selected_stats) > 0:
                s_name = selected_stats[0]
                r1, v1 = get_stat_info(s_name, team1); r2, v2 = get_stat_info(s_name, team2)
                kpi_col1.markdown(make_custom_kpi_card(s_name, r1, r2, v1, v2, team1, team2, t1_color, t2_color, is_to_category), unsafe_allow_html=True)
            if len(selected_stats) > 1:
                s_name = selected_stats[1]
                r1, v1 = get_stat_info(s_name, team1); r2, v2 = get_stat_info(s_name, team2)
                kpi_col2.markdown(make_custom_kpi_card(s_name, r1, r2, v1, v2, team1, team2, t1_color, t2_color, is_to_category), unsafe_allow_html=True)
            if len(selected_stats) > 2:
                s_name = selected_stats[2]
                r1, v1 = get_stat_info(s_name, team1); r2, v2 = get_stat_info(s_name, team2)
                kpi_col3.markdown(make_custom_kpi_card(s_name, r1, r2, v1, v2, team1, team2, t1_color, t2_color, is_to_category), unsafe_allow_html=True)
        except Exception: pass

        st.write("---")
        st.subheader(f"📊 {category} 상세 분석 테이블")
        
        display_df = filtered.copy()
        if "턴오버" in category:
            display_df['출력값'] = display_df.apply(lambda x: f"{int(x['수치'])} ({int(x['순위'])}위)", axis=1)
        else:
            display_df['출력값'] = display_df.apply(lambda x: f"{x['수치']:.2f} ({int(x['순위'])}위)", axis=1)
            
        display_table = display_df.pivot(index='스탯명', columns='팀명', values='출력값').reindex(selected_stats)
        gap_values = numeric_table[team1] - numeric_table[team2]
        if "턴오버" in category: display_table[f"격차 ({team1} - {team2})"] = gap_values.apply(lambda x: f"{int(x)}")
        else: display_table[f"격차 ({team1} - {team2})"] = gap_values.apply(lambda x: f"{x:.2f}")
            
        st.dataframe(
            display_table, 
            use_container_width=True, 
            height=(len(display_table) + 1) * 35  # 데이터 개수 * 행 높이(35px)만큼 높이 자동 할당
        )
        
        st.write("---")
        st.subheader("📉 스탯별 나란히 비교 그래프")
        
        df_plot = numeric_table[[team1, team2]].reset_index()
        df_melted = df_plot.melt(id_vars='스탯명', value_vars=[team1, team2], var_name='팀명', value_name='수치')
        
        fig = px.bar(
            df_melted, x='수치', y='스탯명', color='팀명', barmode='group', orientation='h',
            color_discrete_map={team1: t1_color, team2: t2_color}, 
            text_auto='.2f' if "턴오버" not in category else '.0f'
        )
        
        fig.update_layout(
            xaxis_title="기록", yaxis_title="", legend_title="팀",
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            margin=dict(t=30, b=0, l=0, r=0),
            height=max(400, len(selected_stats) * 65)
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("NFLverse 데이터를 받아오는 중이거나 정제된 내역이 없습니다.")

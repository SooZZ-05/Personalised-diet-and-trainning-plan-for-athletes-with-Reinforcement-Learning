
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Stable-Baselines3 / Gymnasium imports ---
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
except Exception as e:
    st.error("Missing dependencies. Make sure requirements.txt is installed on Streamlit Cloud.")
    st.stop()

# =============================
# Athlete Environment (minimal)
# =============================
class AthleteEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, render_mode=None):
        super().__init__()
        self.np_random = np.random.default_rng()
        self.render_mode = render_mode
        # Action space: Dict with 5 meal picks, 1 exercise, 1 sleep hour (0-12)
        self.action_space = spaces.Dict({
            "meal_choice": spaces.MultiDiscrete([25, 25, 25, 25, 25]),
            "exercise_choice": spaces.Discrete(10),
            "sleep_duration": spaces.Box(low=0.0, high=12.0, shape=(1,), dtype=np.float32)
        })
        self.observation_space = spaces.Dict({
            "gender": spaces.Discrete(2),
            "weight": spaces.Box(low=40.0, high=150.0, shape=(1,), dtype=np.float32),
            "height": spaces.Box(low=140.0, high=210.0, shape=(1,), dtype=np.float32),
            "age": spaces.Discrete(100),
            "activity_level": spaces.Discrete(5),
            "calories_today": spaces.Box(low=0.0, high=7000.0, shape=(1,), dtype=np.float32),
            "satiety_level": spaces.Box(low=0.0, high=1000.0, shape=(1,), dtype=np.float32),
            "fatigue_level": spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            "days_since_rest": spaces.Discrete(10),
            "sleep_hours": spaces.Box(low=0.0, high=24.0, shape=(1,), dtype=np.float32),
            "day_of_cycle": spaces.Discrete(30),
            "next_segment": spaces.Discrete(3)
        })
        self._define_food_data()
        self._define_exercise_data()
        self.state = None
        self.current_step_in_episode = 0
        self.episode_length = 30
        self.target_weight = None

    def _define_food_data(self):
        food_data = {
            'index': list(range(1, 26)),
            'name': [
                'White bread','Eggs','Cornflakes','Croissant','Grain bread',
                'Whole meal bread','White rice','Brown rice','White pasta','Brown pasta',
                'Porridge','Potatoes','Baked beans','Lentils','Cheese',
                'Beef','Ling fish','Peanuts','Yogurt','Ice cream',
                'Cake','Cookies','Apples','Grapes','Bananas'
            ],
            'satiety': [100,150,150,47,154,157,138,132,119,188,209,323,168,133,146,176,225,84,88,96,65,120,197,162,118],
            'calories': [132,156,100,272,162,138,206,216,169,238,125,177,155,116,80,265,87,161,174,137,129,80,96,40,111]
        }
        self.food_df = pd.DataFrame(food_data).set_index('index')

    def _define_exercise_data(self):
        exercise_data = {
            'index': list(range(1, 11)),
            'name': [
                'Aerobics','High intensity aerobics','Calisthenics','High intensity calisthenics',
                'Running','Swimming','Weight lifting','High intensity weight lifting','Walking','None'
            ],
            'calories_burnt': [422,493,317,563,563,563,211,422,493,0],
            'fatigue_effect': [1,1,1,1.5,1.5,1.5,0.5,1,1,-3]
        }
        self.exercise_df = pd.DataFrame(exercise_data).set_index('index')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        g = rng.integers(0, 2)
        if g == 1:
            w = rng.uniform(65.0, 95.0); h = rng.uniform(170.0, 195.0)
        else:
            w = rng.uniform(55.0, 75.0); h = rng.uniform(155.0, 180.0)
        a = rng.integers(18, 35); act = rng.integers(1, 6)
        bmi = w / ((h/100)**2)
        if 18.5 <= bmi <= 24.9:
            tbmi = rng.uniform(max(18.5, bmi-1.0), min(24.9, bmi+1.0))
        elif bmi < 18.5:
            tbmi = rng.uniform(18.5, 21.0)
        else:
            tbmi = rng.uniform(22.0, 24.9)
        self.target_weight = tbmi * ((h/100)**2)
        self.target_weight = max(60.0 if g==1 else 50.0, min(95.0 if g==1 else 80.0, self.target_weight))
        self.state = {
            "gender": np.array([g], dtype=np.int32),
            "weight": np.array([w], dtype=np.float32),
            "height": np.array([h], dtype=np.float32),
            "age": np.array([a], dtype=np.int32),
            "activity_level": np.array([act], dtype=np.int32),
            "calories_today": np.array([0.0], dtype=np.float32),
            "satiety_level": np.array([0.0], dtype=np.float32),
            "fatigue_level": np.array([0.0], dtype=np.float32),
            "days_since_rest": 0,
            "sleep_hours": np.array([0.0], dtype=np.float32),
            "day_of_cycle": 0,
            "next_segment": 0
        }
        self.current_step_in_episode = 0
        return self.state, {}

    def _get_next_state_copy(self):
        s = self.state
        return {
            "gender": s["gender"].copy(),
            "weight": s["weight"].copy(),
            "height": s["height"].copy(),
            "age": s["age"].copy(),
            "activity_level": s["activity_level"].copy(),
            "calories_today": s["calories_today"].copy(),
            "satiety_level": s["satiety_level"].copy(),
            "fatigue_level": s["fatigue_level"].copy(),
            "days_since_rest": s["days_since_rest"],
            "sleep_hours": s["sleep_hours"].copy(),
            "day_of_cycle": s["day_of_cycle"],
            "next_segment": s["next_segment"]
        }

    def _apply_meal_action(self, meal_action):
        total_cals = 0.0; total_satiety = 0.0
        for idx in meal_action:
            item = self.food_df.loc[int(idx)+1]
            total_cals += float(item['calories']); total_satiety += float(item['satiety'])
        return total_cals, total_satiety

    def _apply_exercise_action(self, exercise_action):
        ex = self.exercise_df.loc[int(exercise_action)+1]
        return float(ex['calories_burnt']), float(ex['fatigue_effect'])

    def _simulate_sleep_recovery(self, state, sleep_hours):
        recovery_factor = min(1.0, float(sleep_hours) / 8.0)
        state["fatigue_level"] = np.maximum(0, state["fatigue_level"] - (1.0 * recovery_factor))
        state["satiety_level"] *= 0.3
        state["sleep_hours"] = np.array([sleep_hours], dtype=np.float32)

    def _advance_day(self, state):
        tdee = state["weight"][0] * 30
        net = state["calories_today"][0] - tdee
        state["weight"] += np.array([net / 7000.0], dtype=np.float32)
        state["calories_today"] = np.array([0.0], dtype=np.float32)
        state["day_of_cycle"] = (state["day_of_cycle"] + 1) % 30

    def _calculate_reward(self, state, segment_name, exercise_action):
        r = 0.0
        if segment_name == "morning" and int(exercise_action) != 9:
            r += 50.0
        if state["fatigue_level"][0] > 8.0: r -= 15.0
        r += 30.0 if state["days_since_rest"] <= 5 else -25.0
        if segment_name == "evening" and state["sleep_hours"][0] >= 7.0: r += 40.0
        r -= abs(state["weight"][0] - self.target_weight) * 5.0

        # calorie adherence (Mifflin-St Jeor style maintenance * activity_level)
        if state["gender"][0] == 1:
            maintenance = 10*state["weight"][0] + 6.25*state["height"][0] - 5*state["age"][0] + 5
        else:
            maintenance = 10*state["weight"][0] + 6.25*state["height"][0] - 5*state["age"][0] - 161
        maintenance *= max(1, int(state["activity_level"][0]))
        recommend = maintenance - 500 if state["weight"][0] > self.target_weight else maintenance + 500
        cal_dev = (state["calories_today"][0] - recommend)
        r += float(- (cal_dev ** 2) / (2 * (recommend * 0.1) ** 2))
        return float(r)

    def step(self, action):
        meal = action["meal_choice"]; exercise = action["exercise_choice"]; sleep = float(action["sleep_duration"][0])
        seg = int(self.state["next_segment"])
        seg_name = {0:"morning",1:"midday",2:"evening"}[seg]
        ns = self._get_next_state_copy()

        cals, sat = self._apply_meal_action(meal)
        ns["calories_today"] += cals
        ns["satiety_level"] = np.array([sat], dtype=np.float32)

        if seg_name == "morning":
            burn, fat = self._apply_exercise_action(exercise)
            ns["calories_today"] -= burn
            ns["fatigue_level"] += fat
            if int(exercise) != 9: ns["days_since_rest"] += 1
            else: ns["days_since_rest"] = 0
        else:
            pass

        if seg_name == "evening":
            self._simulate_sleep_recovery(ns, sleep)
        else:
            ns["satiety_level"] *= 0.7

        ns["next_segment"] = (seg + 1) % 3
        if seg_name == "evening":
            self._advance_day(ns)

        reward = self._calculate_reward(ns, seg_name, exercise)

        self.state = ns
        self.current_step_in_episode += 1
        terminated = False
        truncated = self.current_step_in_episode >= (self.episode_length * 3)
        return self.state, reward, terminated, truncated, {}

# =============================
# Wrapper to make PPO-compatible
# =============================
class AthleteEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        original = env.action_space
        self.meal_size = original['meal_choice'].nvec
        self.exercise_size = original['exercise_choice'].n
        self.sleep_size = 13
        self.action_space = spaces.MultiDiscrete([*self.meal_size, self.exercise_size, self.sleep_size])
        self.sleep_bins = np.linspace(0.0, 12.0, self.sleep_size)

        # Flatten observation
        total_dim = 0
        for _, sp in env.observation_space.spaces.items():
            total_dim += sp.shape[0] if isinstance(sp, spaces.Box) else 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

        self.obs_structure = []
        for k, sp in env.observation_space.spaces.items():
            if isinstance(sp, spaces.Box): self.obs_structure.append(('box', k, sp.shape[0]))
            else: self.obs_structure.append(('discrete', k, 1))

    def _flatten_observation(self, obs):
        flat = []
        for typ, k, _ in self.obs_structure:
            v = obs[k]
            if typ == 'box':
                flat.extend(np.array(v).flatten().tolist())
            else:
                # ensure scalar
                if hasattr(v, "item"):
                    flat.append(float(v.item()))
                elif isinstance(v, (np.ndarray, list)):
                    flat.append(float(np.array(v).reshape(-1)[0]))
                else:
                    flat.append(float(v))
        return np.array(flat, dtype=np.float32)

    def step(self, action):
        meal_action = action[:5]
        exercise_action = int(action[5])
        sleep_idx = int(action[6])
        dict_action = {
            "meal_choice": meal_action,
            "exercise_choice": exercise_action,
            "sleep_duration": np.array([self.sleep_bins[sleep_idx]], dtype=np.float32)
        }
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        return self._flatten_observation(obs), float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_observation(obs), info

# =============================
# Utilities
# =============================
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    try:
        model = PPO.load(model_path, device="cpu")
        return model
    except Exception as e:
        st.exception(e)
        st.stop()

def simulate_days(model, days=30, seed=None):
    """Rollout a full episode deterministically using the PPO model"""
    base_env = AthleteEnv(render_mode=None)
    env = AthleteEnvWrapper(base_env)
    if seed is not None:
        np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    records = []
    day = 1
    seg_map = {0:"Morning", 1:"Midday", 2:"Evening"}
    food_names = base_env.food_df['name'].to_dict()
    ex_names = base_env.exercise_df['name'].to_dict()

    for t in range(days*3):
        action, _ = model.predict(obs, deterministic=True)
        # decode for logging
        meal = list(map(int, action[:5]))
        exercise = int(action[5])
        sleep_idx = int(action[6])
        sleep_hours = float(env.sleep_bins[sleep_idx])
        # before step: which segment?
        # we need to peek from base_env.state["next_segment"]
        seg = int(base_env.state["next_segment"])
        seg_name = seg_map[seg]

        obs, reward, terminated, truncated, info = env.step(action)

        row = {
            "t": t,
            "day": (t // 3) + 1,
            "segment": seg_name,
            "weight": float(base_env.state["weight"][0]),
            "calories_today": float(base_env.state["calories_today"][0]),
            "satiety": float(base_env.state["satiety_level"][0]),
            "fatigue": float(base_env.state["fatigue_level"][0]),
            "sleep_hours": float(base_env.state["sleep_hours"][0]),
            "days_since_rest": int(base_env.state["days_since_rest"]),
            "reward": float(reward),
            "meal_items": ", ".join([food_names[i+1] for i in meal]),
            "exercise": ex_names[exercise+1] if seg_name=="Morning" else "-",
            "sleep_planned": sleep_hours if seg_name=="Evening" else 0.0,
            "target_weight": float(base_env.target_weight),
        }
        records.append(row)
        if truncated: break
    df = pd.DataFrame(records)
    return df

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Athlete Diet & Training — PPO Simulator", layout="wide", page_icon="🏃‍♂️")

st.title("🏃‍♂️ Athlete Diet & Training (PPO) — Day-by-Day Animation")
st.caption("Visualize the athlete's status after meals and training across days using the **final PPO model**.")

# Sidebar — Model file
st.sidebar.header("Model")
default_model_path = st.sidebar.text_input("PPO model path (.zip)", "athlete_ppo_model.zip",
                                           help="Upload the final PPO model with Stable-Baselines3 format.")
uploaded = st.sidebar.file_uploader("...or upload model file", type=["zip"])

model_path = default_model_path
if uploaded is not None:
    # Save uploaded model to a temp path for SB3 to load
    model_path = os.path.join("/tmp", uploaded.name)
    with open(model_path, "wb") as f:
        f.write(uploaded.getbuffer())

# Controls
st.sidebar.header("Simulation")
days = st.sidebar.slider("Days to simulate", min_value=7, max_value=60, value=30, step=1)
seed = st.sidebar.number_input("Random seed (optional)", value=0, step=1)
run_btn = st.sidebar.button("Run Simulation")

# Session state for animation
if "df" not in st.session_state:
    st.session_state.df = None
if "frame" not in st.session_state:
    st.session_state.frame = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

# Load and simulate
if run_btn:
    model = load_model(model_path)
    with st.spinner("Simulating..."):
        st.session_state.df = simulate_days(model, days=days, seed=int(seed) if seed else None)
        st.session_state.frame = 0
        st.session_state.playing = True

df = st.session_state.df
if df is None:
    st.info("Load your **final PPO model** and click **Run Simulation** to generate the animated visualization.")
    st.stop()

# =============================
# Animated Dashboard
# =============================
# High-level KPI cards
latest = df.iloc[st.session_state.frame]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Day", int(latest["day"]))
col2.metric("Segment", latest["segment"])
col3.metric("Weight (kg)", f"{latest['weight']:.2f}")
col4.metric("Target (kg)", f"{latest['target_weight']:.2f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Calories Today", f"{latest['calories_today']:.0f}")
col6.metric("Satiety", f"{latest['satiety']:.1f}")
col7.metric("Fatigue /10", f"{latest['fatigue']:.1f}")
col8.metric("Training streak (days)", int(latest["days_since_rest"]))

col9, = st.columns(1)
col9.metric("Sleep (h)", f"{latest['sleep_hours']:.1f}")

# Plots (weight line, calories, fatigue/satiety)
fig1 = px.line(
    df,
    x="day",
    y="weight",
    color="segment",
    markers=True,
    title="Weight over Time (by day & segment)",
)
# highlight the current day instead of a single segment index
cur_day = int(latest["day"])
fig1.add_vrect(x0=cur_day - 0.5, x1=cur_day + 0.5, line_width=0, opacity=0.15)
fig1.add_vline(x=st.session_state.frame, line_dash="dash")

# Daily aggregation for calories
daily = df.groupby("day", as_index=False).agg(
    weight=("weight","last"),
    calories_today=("calories_today","last"),
    fatigue=("fatigue","last"),
    satiety=("satiety","last"),
)
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=daily["day"], y=daily["calories_today"], name="Calories Today"))
fig2.add_trace(go.Scatter(x=daily["day"], y=daily["weight"], yaxis="y2", mode="lines+markers", name="Weight"))
fig2.update_layout(
    title="Daily Calories (bar) & Weight (line)",
    yaxis_title="Calories",
    yaxis2=dict(title="Weight (kg)", overlaying='y', side='right')
)

# fig3 = go.Figure()
# fig3.add_trace(go.Scatter(x=df["t"], y=df["fatigue"], mode="lines", name="Fatigue"))
# fig3.add_trace(go.Scatter(x=df["t"], y=df["satiety"], mode="lines", name="Satiety"))
# fig3.update_layout(title="Fatigue and Satiety (per segment)", xaxis_title="t")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df["day"], y=df["fatigue"], mode="lines+markers", name="Fatigue"))
fig3.add_trace(go.Scatter(x=df["day"], y=df["satiety"], mode="lines+markers", name="Satiety"))
fig3.update_layout(title="Fatigue and Satiety (per day, 3 segments)", xaxis_title="day")


left, right = st.columns([2,1])
with left:
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

with right:
    st.subheader("Current Actions")
    st.write(f"**Segment:** {latest['segment']}")
    st.write(f"**Meal items:** {latest['meal_items']}")
    st.write(f"**Exercise:** {latest['exercise']}")
    if latest['segment'] == "Evening":
        st.write(f"**Sleep planned:** {latest['sleep_planned']:.1f} h")
    st.caption("Actions are decoded from PPO outputs for the current frame.")

# Playback controls
st.markdown("---")
c1, c2, c3, c4 = st.columns([1,2,4,3])
with c1:
    if st.button("⏮️ Restart"):
        st.session_state.frame = 0
with c2:
    if st.session_state.playing:
        if st.button("⏸️ Pause"):
            st.session_state.playing = False
    else:
        if st.button("▶️ Play"):
            st.session_state.playing = True

with c3:
    st.session_state.frame = st.slider("Scrub timeline", 0, len(df)-1, st.session_state.frame, key="scrubber")

with c4:
    speed = st.selectbox("Speed", options=[0.05, 0.1, 0.25, 0.5], index=2, format_func=lambda x: f"{x}s/frame")

# Auto-advance animation using reruns
if st.session_state.playing:
    next_frame = (st.session_state.frame + 1) % len(df)
    st.session_state.frame = next_frame
    time.sleep(float(speed))
    st.rerun()

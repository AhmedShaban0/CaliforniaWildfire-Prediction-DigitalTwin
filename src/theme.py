"""
Theme and Design System for California Wildfire Prediction System.
Implements a polished, cohesive visual identity inspired by the project logo and colors:
- Deep oceanic navy and slate palette with fire-red, warm-orange, amber-gold, and emerald-green semantic accents.
- Seamless transparent integration of the horizontal project logo asset without harsh white boxes.
- Distinct, high-contrast active page navigation states with glowing indicator accents.
- Unified dashboard cards, hero banners, operational disclosures, and interactive legends.
- Strict LTR layout enforcement and responsive styles for desktop and mobile viewports.
"""
import streamlit as st
import os
import base64

# Exact Color Tokens matching reference design (#06111D palette)
COLOR_CANVAS_BG = "#06111D"        # Deep near-black California navy
COLOR_DARK_NAVY = "#081827"        # Sidebar surface
COLOR_DEEP_NAVY = "#0D2235"        # Primary card surface
COLOR_ELEVATED = "#102A40"         # Secondary panel / elevated surface
COLOR_BORDER = "#1C3A50"           # Subtle border
COLOR_BORDER_LIGHT = "rgba(255, 52, 52, 0.4)"  # Focused / Highlighted border
COLOR_TEXT_PRIMARY = "#F4F7FB"     # High-contrast primary text
COLOR_TEXT_SECONDARY = "#9FB1C5"   # Slate muted text
COLOR_TEXT_MUTED = "#64748B"       # Subtle auxiliary text

# Wildfire Risk & Accent Semantics
COLOR_PRIMARY_RED = "#FF3434"      # Primary fire red accent
COLOR_ORANGE_ACCENT = "#FF6A24"    # Bright orange accent
COLOR_CRIMSON = "#EF4444"          # Extreme risk (>= 45%)
COLOR_FIRE_RED = "#FF3434"         # Fire red (High risk / Alert cutoff >= 25.07%)
COLOR_FIRE_ORANGE = "#FF6A24"      # Fire orange (Fire emphasis / detections)
COLOR_WARM_ORANGE = "#FF7A24"      # Warm orange
COLOR_GOLD = "#F59E0B"             # Amber Gold (Moderate risk 15% - 25.07%)
COLOR_GREEN = "#22C55E"            # Emerald green (Low risk < 15%)
COLOR_RISK_GREEN = "#22C55E"
COLOR_RISK_AMBER = "#F59E0B"
COLOR_RISK_RED = "#EF4444"
COLOR_RISK_CRIMSON = "#DC2626"

# Exact Gradients
GRADIENT_HERO = "linear-gradient(135deg, #081827 0%, #0D2235 50%, #102A40 100%)"
GRADIENT_FIRE = "linear-gradient(110deg, #FF3434 0%, #FF6A24 100%)"
GRADIENT_CARD = "linear-gradient(145deg, #0D2235 0%, #081827 100%)"
GRADIENT_ACTIVE_NAV = "linear-gradient(90deg, rgba(255, 52, 52, 0.25) 0%, rgba(13, 34, 53, 0.85) 100%)"

# Compatibility Aliases
COLOR_CRIMSON_RED = COLOR_CRIMSON
COLOR_CARD_BG = COLOR_DEEP_NAVY
COLOR_MAIN_BG = COLOR_CANVAS_BG
COLOR_PRIMARY_TEXT = COLOR_TEXT_PRIMARY
COLOR_SECONDARY_TEXT = COLOR_TEXT_SECONDARY
COLOR_SUCCESS_LOW = COLOR_GREEN
COLOR_WARNING_MODERATE = COLOR_GOLD
COLOR_BRAND_NAVY = COLOR_DEEP_NAVY

RISK_TIER_COLORS = {
    "Low": COLOR_GREEN,
    "Moderate": COLOR_GOLD,
    "High": COLOR_FIRE_ORANGE,
    "Extreme": COLOR_CRIMSON,
}


@st.cache_data(show_spinner=False)
def get_logo_b64() -> str:
    """Returns base64 encoded string of wildfire.png."""
    logo_path = "wildfire.png"
    if not os.path.exists(logo_path):
        alt = os.path.join("assets", "wildfire.png")
        if os.path.exists(alt):
            logo_path = alt
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


REGION_IMAGE_MAP = {
    "Napa": "assets/regions/napa_valley.jpg",
    "Tahoe": "assets/regions/lake_tahoe.jpg",
    "Paradise": "assets/regions/paradise_butte.jpg",
    "Angeles": "assets/regions/angeles_forest.jpg",
    "Shasta": "assets/regions/shasta_cascade.jpg",
    "Yosemite": "assets/regions/yosemite_mariposa.jpg",
    "Santa Barbara": "assets/regions/santa_barbara.jpg",
    "San Diego": "assets/regions/san_diego_cleveland.jpg",
    "Cleveland": "assets/regions/san_diego_cleveland.jpg",
    "Mendocino": "assets/regions/mendocino_coast.jpg",
    "Central Valley": "assets/regions/central_valley.jpg",
    "Fresno": "assets/regions/central_valley.jpg",
}


@st.cache_data(show_spinner=False)
def get_region_thumb_b64(region_name: str = None) -> str:
    """Returns base64 encoded string of location thumbnail matching region or fallback."""
    path = os.path.join("assets", "location_thumb.jpg")
    if region_name:
        for k, p in REGION_IMAGE_MAP.items():
            if k.lower() in region_name.lower() and os.path.exists(p):
                path = p
                break
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


@st.cache_data(show_spinner=False)
def get_location_thumb_b64(region_name: str = None) -> str:
    """Returns base64 encoded string of location thumbnail."""
    return get_region_thumb_b64(region_name)


@st.cache_data(show_spinner=False)
def get_fire_thumb_b64() -> str:
    """Returns base64 encoded string of wildfire flame thumbnail."""
    path = os.path.join("assets", "fire_thumb.jpg")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


def render_html(html_str: str):
    """
    Safely renders user-facing HTML in Streamlit.
    Strips leading and trailing whitespace from every line to ensure CommonMark
    never mistakenly interprets HTML tags as indented markdown code blocks.
    """
    lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    cleaned = "\n".join(lines)
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)


CUSTOM_DASHBOARD_CSS = f"""
<style>
/* Force strict Left-to-Right direction across the entire application and all controls */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], div[data-baseweb="slider"], div[data-testid="stSlider"] {{
    direction: ltr !important;
    text-align: left !important;
}}

div[data-testid="stSlider"],
div[data-baseweb="slider"],
div[data-baseweb="slider"] * {{
    direction: ltr !important;
}}

/* Slider Component Layout & Polish */
div[data-testid="stSlider"] {{
    direction: ltr !important;
    text-align: left !important;
    margin-bottom: 0.65rem !important;
}}

div[data-testid="stSlider"] [data-testid="stWidgetLabel"] {{
    min-height: 24px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
}}

div[data-testid="stSlider"] [data-testid="stWidgetLabel"] label {{
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: {COLOR_TEXT_PRIMARY} !important;
}}

div[data-testid="stSliderTickBar"] {{
    direction: ltr !important;
    font-family: ui-monospace, SFMono-Regular, "Roboto Mono", Menlo, monospace !important;
    font-size: 0.74rem !important;
    color: {COLOR_TEXT_SECONDARY} !important;
    margin-top: 4px !important;
}}

/* Main container styling */
.stApp {{
    background-color: {COLOR_CANVAS_BG};
    color: {COLOR_TEXT_PRIMARY};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}}

/* Main app padding for pleasant spacing */
.block-container {{
    padding-top: 1.75rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1440px !important;
}}

/* Sidebar styling - Full height, ~285px wide, dark navy */
[data-testid="stSidebar"] {{
    min-width: 285px !important;
    max-width: 285px !important;
    background-color: {COLOR_DARK_NAVY} !important;
    border-right: 1px solid {COLOR_BORDER} !important;
}}

[data-testid="stSidebarContent"] {{
    padding-top: 0.5rem !important;
    padding-bottom: 0.75rem !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
}}

[data-testid="stSidebarUserContent"] {{
    padding: 0 !important;
}}

[data-testid="stSidebar"] .stMarkdown {{
    color: {COLOR_TEXT_SECONDARY};
}}

/* Typography */
h1, h2, h3, h4, h5, h6 {{
    color: {COLOR_TEXT_PRIMARY} !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}}

p, span, label, div {{
    color: {COLOR_TEXT_PRIMARY};
}}

.app-mono {{
    font-family: ui-monospace, SFMono-Regular, "Roboto Mono", Menlo, Consolas, monospace;
    letter-spacing: 0.04em;
}}

.app-muted {{
    color: {COLOR_TEXT_SECONDARY} !important;
    font-size: 0.88rem;
    line-height: 1.55;
}}

/* Main Page Hero Banner */
.app-hero-panel {{
    background: {GRADIENT_HERO};
    border: 1px solid {COLOR_BORDER};
    border-left: 4px solid {COLOR_FIRE_RED};
    border-radius: 14px;
    padding: 1.5rem 1.85rem;
    margin-bottom: 1.75rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
}}

.app-hero-top-row {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
    flex-wrap: wrap;
}}

/* Seamless, glassmorphic logo container without white background box */
.app-hero-tile {{
    background: rgba(16, 42, 67, 0.55);
    border: 1px solid rgba(51, 92, 128, 0.6);
    border-radius: 10px;
    padding: 6px 14px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    height: 52px;
    backdrop-filter: blur(6px);
}}

.app-hero-tile img {{
    height: 40px;
    width: auto;
    object-fit: contain;
    display: block;
}}

.app-hero-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background-color: rgba(24, 59, 86, 0.7);
    color: {COLOR_WARM_ORANGE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 20px;
    padding: 0.35rem 0.85rem;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: ui-monospace, SFMono-Regular, "Roboto Mono", Menlo, monospace;
}}

.app-hero-title {{
    font-size: 2.15rem !important;
    font-weight: 750 !important;
    color: {COLOR_TEXT_PRIMARY} !important;
    margin: 0 0 0.45rem 0 !important;
    letter-spacing: -0.025em;
    line-height: 1.15;
}}

.app-hero-subtitle {{
    color: {COLOR_TEXT_SECONDARY};
    font-size: 0.95rem;
    line-height: 1.55;
    margin: 0;
    max-width: 980px;
}}

/* Section Titles with Emoji Prefix */
.app-section-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1.6rem 0 0.9rem 0;
    border-bottom: 1px solid rgba(36, 69, 99, 0.5);
    padding-bottom: 0.5rem;
}}

.app-section-title {{
    font-size: 1.28rem !important;
    font-weight: 700 !important;
    color: {COLOR_TEXT_PRIMARY} !important;
    margin: 0 !important;
    letter-spacing: -0.015em;
}}

.app-section-tag {{
    font-family: ui-monospace, SFMono-Regular, "Roboto Mono", Menlo, monospace;
    font-size: 0.72rem;
    color: {COLOR_WARM_ORANGE};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}}

/* Dashboard Intelligence Cards */
.app-intel-card {{
    background: {GRADIENT_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    padding: 1.25rem 1.35rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
}}

.app-intel-card:hover {{
    border-color: {COLOR_WARM_ORANGE};
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}}

.app-card-top {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
}}

.app-icon-tile {{
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background-color: {COLOR_ELEVATED};
    border: 1px solid {COLOR_BORDER};
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}}

.app-card-eyebrow {{
    font-family: ui-monospace, SFMono-Regular, "Roboto Mono", Menlo, monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {COLOR_TEXT_SECONDARY};
}}

.app-card-value {{
    font-size: 1.55rem;
    font-weight: 700;
    color: {COLOR_TEXT_PRIMARY};
    line-height: 1.2;
    margin-bottom: 0.35rem;
}}

.app-card-sub {{
    font-size: 0.82rem;
    color: {COLOR_TEXT_SECONDARY};
    line-height: 1.45;
}}

/* Workflow Card */
.app-workflow-card {{
    background: {GRADIENT_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    padding: 1.25rem 1.35rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    border-top: 3px solid {COLOR_WARM_ORANGE};
}}

.app-workflow-step {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.72rem;
    font-weight: 700;
    color: {COLOR_WARM_ORANGE};
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
}}

.app-workflow-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {COLOR_TEXT_PRIMARY};
    margin-bottom: 0.45rem;
}}

.app-workflow-desc {{
    font-size: 0.82rem;
    color: {COLOR_TEXT_SECONDARY};
    line-height: 1.5;
}}

/* Sidebar Header & Brand */
.app-sidebar-header {{
    padding: 0.4rem 0.2rem 0.85rem 0.2rem;
    border-bottom: 1px solid {COLOR_BORDER};
    margin-bottom: 0.85rem;
    text-align: center;
}}

.app-sidebar-logo-tile {{
    background: rgba(16, 42, 67, 0.55);
    border: 1px solid rgba(51, 92, 128, 0.6);
    border-radius: 10px;
    padding: 8px 12px;
    margin: 0 auto 0.65rem auto;
    width: 100%;
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(6px);
}}

.app-sidebar-logo-tile img {{
    max-height: 64px;
    max-width: 100%;
    width: auto;
    object-fit: contain;
    display: block;
}}

.app-sidebar-status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 0.7rem;
    color: {COLOR_GREEN};
    background: rgba(53, 183, 121, 0.12);
    border: 1px solid rgba(53, 183, 121, 0.3);
    border-radius: 12px;
    padding: 0.2rem 0.65rem;
    font-weight: 600;
}}

.app-sidebar-status-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: {COLOR_GREEN};
    display: inline-block;
    box-shadow: 0 0 6px {COLOR_GREEN};
}}

.app-sidebar-nav-label {{
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 0.68rem;
    color: {COLOR_TEXT_SECONDARY};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0.85rem 0 0.35rem 0.35rem;
    font-weight: 700;
}}

/* Form elements & inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {{
    background-color: {COLOR_DARK_NAVY} !important;
    border-color: {COLOR_BORDER} !important;
    color: {COLOR_TEXT_PRIMARY} !important;
    border-radius: 8px !important;
}}

div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {{
    border-color: {COLOR_WARM_ORANGE} !important;
    box-shadow: 0 0 0 1px {COLOR_WARM_ORANGE} !important;
}}

/* Streamlit buttons - Global styling */
div.stButton > button {{
    background-color: {COLOR_ELEVATED};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 0.45rem 1.15rem;
    font-size: 0.88rem;
    font-weight: 550;
    transition: all 0.15s ease-in-out;
}}

div.stButton > button:hover {{
    border-color: {COLOR_WARM_ORANGE};
    background-color: {COLOR_DEEP_NAVY};
    color: #FFFFFF;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
}}

div.stButton > button[kind="primary"] {{
    background: {GRADIENT_FIRE};
    color: #FFFFFF;
    border: none;
    font-weight: 650;
}}

div.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 0 16px rgba(229, 43, 36, 0.5);
    transform: translateY(-1px);
}}

/* High-contrast Sidebar Navigation Buttons */
[data-testid="stSidebar"] div.stButton {{
    margin-bottom: 4px !important;
}}

[data-testid="stSidebar"] div.stButton > button {{
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    text-align: left !important;
    padding: 0.42rem 0.85rem !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    font-weight: 550 !important;
    min-height: 38px !important;
    height: 38px !important;
    line-height: 1.25 !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
    background-color: transparent !important;
    border: 1px solid transparent !important;
    border-left: 3px solid transparent !important;
    color: #CBD5E1 !important;
}}

[data-testid="stSidebar"] div.stButton > button:hover {{
    background-color: rgba(16, 42, 64, 0.5) !important;
    color: #FFFFFF !important;
    border-color: transparent !important;
    border-left: 3px solid rgba(255, 106, 36, 0.6) !important;
    transform: none !important;
}}

/* Highly identifiable ACTIVE navigation button in sidebar */
[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
    background: linear-gradient(90deg, rgba(255, 52, 52, 0.22) 0%, rgba(13, 34, 53, 0.95) 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 52, 52, 0.45) !important;
    border-left: 3.5px solid #FF3434 !important;
    font-weight: 700 !important;
    box-shadow: 0 0 14px rgba(255, 52, 52, 0.2) !important;
}}

[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(90deg, rgba(255, 52, 52, 0.32) 0%, rgba(13, 34, 53, 1.0) 100%) !important;
    border-color: #FF3434 !important;
}}

/* Prediction Method Selector Buttons */
div[data-testid="column"]:has(#pred-method-anchor) div[data-testid="stHorizontalBlock"] {{
    gap: 14px !important;
}}

div[data-testid="column"]:has(#pred-method-anchor) div.stButton > button,
button[aria-label*="Real-time Prediction"],
button[aria-label*="Manual Prediction"] {{
    height: 44px !important;
    min-height: 42px !important;
    max-height: 44px !important;
    width: 100% !important;
    min-width: 180px !important;
    max-width: none !important;
    padding: 0 14px !important;
    font-size: 14.5px !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease-in-out !important;
    box-sizing: border-box !important;
}}

div[data-testid="column"]:has(#pred-method-anchor) div.stButton > button div[data-testid="stMarkdownContainer"],
div[data-testid="column"]:has(#pred-method-anchor) div.stButton > button div[data-testid="stMarkdownContainer"] > p,
button[aria-label*="Real-time Prediction"] div[data-testid="stMarkdownContainer"],
button[aria-label*="Real-time Prediction"] div[data-testid="stMarkdownContainer"] > p,
button[aria-label*="Manual Prediction"] div[data-testid="stMarkdownContainer"],
button[aria-label*="Manual Prediction"] div[data-testid="stMarkdownContainer"] > p {{
    font-size: 14.5px !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
    margin: 0 !important;
    line-height: 1.2 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
}}

/* Active Prediction Method Button: Radiant Wildfire Red-Orange Gradient */
div[data-testid="column"]:has(#pred-method-anchor) div.stButton > button[kind="primary"],
button[aria-label*="Real-time Prediction"][kind="primary"],
button[aria-label*="Manual Prediction"][kind="primary"],
button[aria-label*="Forecast"] {{
    background: linear-gradient(135deg, #FF3434 0%, #FF6A24 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 106, 36, 0.5) !important;
    box-shadow: 0 2px 14px rgba(255, 52, 52, 0.35) !important;
    font-size: 14.5px !important;
    font-weight: 600 !important;
    padding: 0 14px !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
    height: 44px !important;
    min-height: 42px !important;
}}

button[aria-label*="Forecast"] {{
    font-size: 13.5px !important;
    padding: 0 8px !important;
}}

button[aria-label*="Forecast"] p {{
    font-size: 13.5px !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
}}

div[data-testid="column"]:has(#pred-method-anchor) div.stButton > button[kind="primary"]:hover,
button[aria-label*="Real-time Prediction"][kind="primary"]:hover,
button[aria-label*="Manual Prediction"][kind="primary"]:hover {{
    box-shadow: 0 0 18px rgba(255, 52, 52, 0.55) !important;
    transform: translateY(-1px) !important;
}}

/* Inactive Prediction Method Button: Dark Navy with Border */
div[data-testid="column"]:has(#pred-method-anchor) div.stButton > button[kind="secondary"],
button[aria-label*="Real-time Prediction"][kind="secondary"],
button[aria-label*="Manual Prediction"][kind="secondary"] {{
    background-color: #0D2235 !important;
    color: #9FB1C5 !important;
    border: 1px solid #1C3A50 !important;
    box-shadow: none !important;
}}

div[data-testid="column"]:has(#pred-method-anchor) div.stButton > button[kind="secondary"]:hover,
button[aria-label*="Real-time Prediction"][kind="secondary"]:hover,
button[aria-label*="Manual Prediction"][kind="secondary"]:hover {{
    background-color: #102A40 !important;
    color: #FFFFFF !important;
    border-color: rgba(255, 106, 36, 0.5) !important;
    transform: translateY(-1px) !important;
}}

@media (max-width: 992px) {{
    div[data-testid="column"]:has(#pred-method-anchor) div[data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
    }}
    div[data-testid="column"]:has(#pred-method-anchor) div.stButton > button,
    button[aria-label*="Real-time Prediction"],
    button[aria-label*="Manual Prediction"] {{
        width: 100% !important;
        max-width: 100% !important;
    }}
}}

/* Responsive Environmental Factors Grid */
.env-factors-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
}}

@media (max-width: 992px) {{
    .env-factors-grid {{
        grid-template-columns: repeat(2, 1fr);
    }}
}}

@media (max-width: 640px) {{
    .env-factors-grid {{
        grid-template-columns: 1fr;
    }}
}}

/* Sidebar Footer Card */
.app-sidebar-bottom-card {{
    background: {GRADIENT_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 10px;
    padding: 0.75rem 0.85rem;
    text-align: left;
    margin-top: 1rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}}

.app-sidebar-bottom-card .title {{
    font-size: 0.78rem;
    font-weight: 700;
    color: {COLOR_TEXT_PRIMARY};
    margin-bottom: 0.25rem;
}}

.app-sidebar-bottom-card .meta {{
    font-size: 0.72rem;
    color: {COLOR_TEXT_SECONDARY};
    line-height: 1.45;
}}

/* Prediction Result Panel */
.app-prediction-panel {{
    background: {GRADIENT_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    margin-bottom: 1.25rem;
}}

/* Operational notice card */
.app-notice-card {{
    background-color: {COLOR_DARK_NAVY};
    border: 1px solid {COLOR_BORDER};
    border-left: 4px solid {COLOR_GOLD};
    border-radius: 8px;
    padding: 0.95rem 1.25rem;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}}

.app-notice-title {{
    font-size: 0.85rem;
    font-weight: 600;
    color: {COLOR_GOLD};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.25rem;
    font-family: ui-monospace, SFMono-Regular, "Roboto Mono", Menlo, monospace;
}}

.app-notice-body {{
    font-size: 0.84rem;
    color: {COLOR_TEXT_SECONDARY};
    line-height: 1.5;
    margin: 0;
}}

/* Color Legend Bar */
.app-risk-legend-bar {{
    background: {COLOR_DARK_NAVY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-top: 0.65rem;
    margin-bottom: 0.85rem;
}}

.app-legend-items {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 1.25rem;
    margin-bottom: 0.4rem;
}}

.app-legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.82rem;
    color: {COLOR_TEXT_PRIMARY};
}}

.app-legend-color-box {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid rgba(255, 255, 255, 0.25);
    flex-shrink: 0;
}}

.app-legend-caption {{
    font-size: 0.75rem;
    color: {COLOR_TEXT_SECONDARY};
    line-height: 1.4;
    border-top: 1px solid rgba(36, 69, 99, 0.4);
    padding-top: 0.35rem;
}}
</style>
"""


def apply_theme():
    """Inject the centralized California Wildfire Dashboard design system."""
    st.markdown(CUSTOM_DASHBOARD_CSS, unsafe_allow_html=True)
    if hasattr(st, "html"):
        st.html("""
        <script>
        (function enforceLTR() {
            try {
                var sym = Symbol.for("react-aria.i18n.locale");
                window[sym] = "en-US";
                if (window.parent && window.parent !== window) {
                    try {
                        window.parent[sym] = "en-US";
                        window.parent.dispatchEvent(new Event("languagechange"));
                    } catch(e) {}
                }
                window.dispatchEvent(new Event("languagechange"));
                document.documentElement.dir = "ltr";
                document.documentElement.lang = "en";
                document.body.dir = "ltr";
            } catch(err) {}
        })();
        </script>
        """)


def render_sidebar_header():
    """Renders the branded header in the sidebar with the exact original logo inside a white background container."""
    b64 = get_logo_b64()
    img_tag = f'<img src="data:image/png;base64,{b64}" alt="California Wildfire Prediction Logo" style="width: 100%; max-width: 210px; height: auto; max-height: 90px; object-fit: contain; display: block; margin: 0 auto;" />' if b64 else '<span style="font-size: 2.2rem;">🔥</span>'

    html = f"""
    <div style="padding: 4px 2px 8px 2px; text-align: center;">
        <div style="display: flex; align-items: center; justify-content: center; padding: 10px 14px; background: #FFFFFF; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.25); box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);">
            {img_tag}
        </div>
        <div style="height: 1px; background: linear-gradient(90deg, transparent 0%, #1C3A50 25%, #1C3A50 75%, transparent 100%); margin: 12px 0 8px 0;"></div>
    </div>
    """
    render_html(html)


def render_sidebar_project_card():
    """Renders a compact, refined operational card in the sidebar that never forces scrolling."""
    html = f"""
    <div style="background: rgba(13, 34, 53, 0.65); border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 8px 10px; margin-top: 10px; text-align: left; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2px;">
            <span style="font-size: 0.75rem; font-weight: 750; color: #FFFFFF;">California Fire Atlas</span>
            <span style="font-size: 0.65rem; color: #22C55E; font-family: ui-monospace, monospace; font-weight: 600;">ACTIVE</span>
        </div>
        <div style="font-size: 0.68rem; color: #9FB1C5; line-height: 1.35;">
            24h Horizon &bull; 0.25° (~25 km) Grid &bull; ERA5
        </div>
    </div>
    """
    render_html(html)


def render_wildfire_header(
    title: str = "Wildfire Prediction",
    subtitle: str = "Predict wildfire risk using satellite observations, weather conditions, and machine learning.",
    timestamp_str: str = "Live Forecast (UTC)",
):
    """
    Renders the primary header matching the reference image:
    'Wildfire Prediction' with 'Prediction' in red-orange gradient,
    subtitle below, and a compact UTC timestamp pill on the top right.
    """
    html = f"""
    <div style="display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 1.25rem; flex-wrap: wrap; gap: 0.75rem;">
        <div>
            <h1 style="font-size: 2.25rem; font-weight: 800; color: #FFFFFF; margin: 0 0 0.25rem 0; letter-spacing: -0.025em; line-height: 1.15;">
                Wildfire <span style="background: linear-gradient(90deg, #FF3434 0%, #FF6A24 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Prediction</span>
            </h1>
            <p style="font-size: 0.92rem; color: #9FB1C5; margin: 0; max-width: 820px; line-height: 1.5;">
                {subtitle}
            </p>
        </div>
        <div style="display: inline-flex; align-items: center; gap: 0.55rem; background: #0D2235; border: 1px solid #1C3A50; border-radius: 20px; padding: 0.4rem 0.95rem; font-family: ui-monospace, monospace; font-size: 0.75rem; color: #F4F7FB; box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
            <span>📅 {timestamp_str}</span>
            <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #22C55E; box-shadow: 0 0 6px #22C55E; margin-left: 2px;"></span>
        </div>
    </div>
    """
    render_html(html)


def render_prediction_result_gauge(score: float, tier: str, pct_str: str, horizon_hours: int = 24):
    """
    Renders the circular radial arc gauge card matching the reference design.
    """
    clamped = max(0.0, min(1.0, float(score)))
    total_arc = 141.37  # pi * r (r=45)
    offset = total_arc * (1.0 - clamped)

    if tier == "Low":
        stroke_color = "#22C55E"
        pill_bg = "rgba(34, 197, 94, 0.16)"
        pill_border = "rgba(34, 197, 94, 0.45)"
        pill_text = "#22C55E"
    elif tier == "Moderate":
        stroke_color = "#F59E0B"
        pill_bg = "rgba(245, 158, 11, 0.16)"
        pill_border = "rgba(245, 158, 11, 0.45)"
        pill_text = "#F59E0B"
    elif tier == "High":
        stroke_color = "#FF3434"
        pill_bg = "rgba(255, 52, 52, 0.2)"
        pill_border = "rgba(255, 52, 52, 0.55)"
        pill_text = "#FF3434"
    else:  # Extreme
        stroke_color = "#DC2626"
        pill_bg = "rgba(220, 38, 38, 0.28)"
        pill_border = "rgba(220, 38, 38, 0.65)"
        pill_text = "#FF5555"

    html = f"""
    <div style="background: {COLOR_DEEP_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 12px; padding: 1.15rem 1.25rem; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35); height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.95rem; font-weight: 700; color: #F4F7FB; margin-bottom: 0.75rem;">
            <span style="font-size: 1.15rem;">🔥</span>
            <span>Prediction Result ({horizon_hours} Hours)</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 0.75rem;">
            <!-- Radial SVG Gauge -->
            <div style="display: flex; flex-direction: column; align-items: center; width: 110px; flex-shrink: 0;">
                <svg viewBox="0 0 120 75" style="width: 110px; overflow: visible;">
                    <!-- Background Track -->
                    <path d="M 15 65 A 45 45 0 0 1 105 65" fill="none" stroke="#162D44" stroke-width="11" stroke-linecap="round"/>
                    <!-- Active Value Stroke -->
                    <path d="M 15 65 A 45 45 0 0 1 105 65" fill="none" stroke="{stroke_color}" stroke-width="11" stroke-dasharray="141.37" stroke-dashoffset="{offset:.2f}" stroke-linecap="round" style="filter: drop-shadow(0 0 6px {stroke_color});"/>
                    <text x="60" y="58" text-anchor="middle" font-size="20" font-weight="800" fill="#F4F7FB" font-family="-apple-system, BlinkMacSystemFont, sans-serif">{score:.2f}</text>
                </svg>
                <div style="margin-top: -6px; background: {pill_bg}; border: 1px solid {pill_border}; border-radius: 12px; padding: 2px 10px; font-size: 0.72rem; font-weight: 750; color: {pill_text}; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap;">
                    {tier} Risk
                </div>
            </div>
            <!-- Probability Text & Explanation -->
            <div style="flex: 1; text-align: left; padding-left: 0.4rem;">
                <div style="font-size: 2.3rem; font-weight: 850; color: {stroke_color}; line-height: 1.05; letter-spacing: -0.02em;">
                    {pct_str}
                </div>
                <div style="font-size: 0.8rem; color: #9FB1C5; margin-top: 0.35rem; line-height: 1.4;">
                    Probability of spaceborne active fire detection within next {horizon_hours} hours
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid rgba(28, 58, 80, 0.6); padding-top: 0.55rem; margin-top: 0.85rem; font-size: 0.72rem; color: #64748B;">
            <b>Contract:</b> VIIRS/MODIS 0.25° detection target (<code>fire_next_24h</code>). Statistical model inference.
        </div>
    </div>
    """
    render_html(html)


def render_selected_location_card(
    name: str,
    lat: float,
    lon: float,
    temp_c: float,
    rh_pct: float,
    wind_ms: float,
    ndvi: float,
    elev_m: float,
    land_cover: str,
    region_key: str = None,
):
    """
    Renders the 'Selected Location' card with dynamic location thumbnail image and 6 parameter rows.
    """
    loc_b64 = get_region_thumb_b64(region_key or name)
    img_tag = f'<img src="data:image/jpeg;base64,{loc_b64}" alt="{name}" style="width: 92px; height: 62px; border-radius: 8px; object-fit: cover; border: 1px solid rgba(255, 106, 36, 0.4); box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35); flex-shrink: 0;" />' if loc_b64 else ''

    html = f"""
    <div style="background: {COLOR_DEEP_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 12px; padding: 1.15rem 1.25rem; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35); height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
        <div style="display: flex; align-items: flex-start; justify-content: space-between; gap: 0.75rem; margin-bottom: 0.85rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.75rem; font-weight: 700; color: #9FB1C5; text-transform: uppercase; letter-spacing: 0.06em;">
                    <span>📍</span>
                    <span>Selected Location</span>
                </div>
                <div style="font-size: 1.15rem; font-weight: 750; color: #F4F7FB; margin-top: 0.15rem; line-height: 1.25;">
                    {name}
                </div>
                <div style="font-family: ui-monospace, monospace; font-size: 0.75rem; color: #64748B;">
                    {lat:.4f}° N, {abs(lon):.4f}° W
                </div>
            </div>
            {img_tag}
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.55rem 1rem; border-top: 1px solid rgba(28, 58, 80, 0.6); padding-top: 0.75rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.82rem;">
                <span style="color: #9FB1C5; display: flex; align-items: center; gap: 0.35rem;">🌡️ Temperature</span>
                <span style="font-weight: 700; color: #F4F7FB; font-family: ui-monospace, monospace;">{temp_c:.1f} °C</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.82rem;">
                <span style="color: #9FB1C5; display: flex; align-items: center; gap: 0.35rem;">💧 Rel. Humidity</span>
                <span style="font-weight: 700; color: #F4F7FB; font-family: ui-monospace, monospace;">{rh_pct:.0f} %</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.82rem;">
                <span style="color: #9FB1C5; display: flex; align-items: center; gap: 0.35rem;">💨 Wind Speed</span>
                <span style="font-weight: 700; color: #F4F7FB; font-family: ui-monospace, monospace;">{wind_ms:.1f} m/s</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.82rem;">
                <span style="color: #9FB1C5; display: flex; align-items: center; gap: 0.35rem;">🍃 NDVI</span>
                <span style="font-weight: 700; color: #F4F7FB; font-family: ui-monospace, monospace;">{ndvi:.2f}</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.82rem;">
                <span style="color: #9FB1C5; display: flex; align-items: center; gap: 0.35rem;">⛰️ Elevation</span>
                <span style="font-weight: 700; color: #F4F7FB; font-family: ui-monospace, monospace;">{elev_m:.0f} m</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.82rem;">
                <span style="color: #9FB1C5; display: flex; align-items: center; gap: 0.35rem;">🌲 Land Cover</span>
                <span style="font-weight: 700; color: #F4F7FB; font-size: 0.78rem; text-align: right; max-width: 140px; word-break: break-word;">{land_cover}</span>
            </div>
        </div>
    </div>
    """
    render_html(html)


def render_key_environmental_factors(
    temp_c: float,
    rh_pct: float,
    wind_ms: float,
    ndvi: float,
    evi: float = 0.12,
    elev_m: float = 0.0,
    sp_hpa: float = 1013.2,
    tp_mm: float = 0.0,
    land_cover: str = "Forest",
):
    """
    Renders the enlarged, spacious responsive grid of 9 key environmental factors
    without clipping or overflow.
    """
    items = [
        ("🌡️", f"{temp_c:.1f} °C", "Temperature"),
        ("💧", f"{rh_pct:.0f} %", "Rel. Humidity"),
        ("💨", f"{wind_ms:.1f} m/s", "Wind Speed"),
        ("🍃", f"{ndvi:.2f}", "NDVI Greenness"),
        ("🌿", f"{evi:.2f}", "EVI Greenness"),
        ("⛰️", f"{elev_m:.0f} m", "Surface Elevation"),
        ("⏱️", f"{sp_hpa:.1f} hPa", "Surface Pressure"),
        ("🌧️", f"{tp_mm:.1f} mm", "24h Precipitation"),
        ("🌲", f"{land_cover}", "Land Cover"),
    ]

    tiles_html = "".join([
        f"""
        <div style="background: {COLOR_ELEVATED}; border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 0.8rem 0.95rem; display: flex; align-items: flex-start; gap: 0.75rem; min-height: 68px;">
            <div style="font-size: 1.35rem; line-height: 1; flex-shrink: 0; padding-top: 2px;">{icon}</div>
            <div style="flex: 1; min-width: 0;">
                <div style="font-size: 0.72rem; color: {COLOR_TEXT_SECONDARY}; text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em; margin-bottom: 2px;">
                    {lbl}
                </div>
                <div style="font-size: 1.05rem; font-weight: 750; color: {COLOR_TEXT_PRIMARY}; font-family: ui-monospace, SFMono-Regular, monospace; line-height: 1.25; word-break: break-word;">
                    {val}
                </div>
            </div>
        </div>
        """
        for icon, val, lbl in items
    ])

    html = f"""
    <div style="background: {COLOR_DEEP_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 12px; padding: 1.15rem 1.35rem; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35); height: 100%;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.85rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.98rem; font-weight: 700; color: #F4F7FB;">
                <span style="font-size: 1.15rem;">🔑</span>
                <span>Key Environmental Factors</span>
            </div>
            <span style="font-size: 0.72rem; color: #9FB1C5; font-family: ui-monospace, monospace;">9 Atmospheric & Terrestrial Parameters</span>
        </div>
        <div class="env-factors-grid">
            {tiles_html}
        </div>
    </div>
    """
    render_html(html)


def render_recent_firms_cards(detections: list):
    """
    Renders recent genuine NASA FIRMS active fire satellite detections in a horizontal strip.
    """
    fire_b64 = get_fire_thumb_b64()
    cards_html = ""
    for d in (detections or [])[:4]:
        ts_str = d.get("timestamp", "Aug 15, 2024 10:24 UTC")
        lat = d.get("lat", 36.842)
        lon = d.get("lon", -119.412)
        cards_html += f"""
        <div style="flex: 1; min-width: 105px; background: rgba(16, 42, 64, 0.45); border: 1px solid rgba(28, 58, 80, 0.8); border-radius: 8px; overflow: hidden;">
            <img src="data:image/jpeg;base64,{fire_b64}" alt="Thermal anomaly" style="width: 100%; height: 50px; object-fit: cover; display: block;" />
            <div style="padding: 0.45rem 0.55rem;">
                <div style="font-size: 0.68rem; font-family: ui-monospace, monospace; color: #CBD5E1; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {ts_str}
                </div>
                <div style="font-size: 0.65rem; font-family: ui-monospace, monospace; color: #FF6A24; margin-top: 2px;">
                    {lat:.3f}° N, {abs(lon):.3f}° W
                </div>
            </div>
        </div>
        """

    html = f"""
    <div style="background: {COLOR_DEEP_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 12px; padding: 1.15rem 1.25rem; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35); height: 100%;">
        <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.95rem; font-weight: 700; color: #F4F7FB; margin-bottom: 0.85rem;">
            <div style="display: flex; align-items: center; gap: 0.45rem;">
                <span>🔥</span>
                <span>Recent Fire Detections (FIRMS)</span>
            </div>
            <span style="font-size: 0.72rem; color: #FF6A24; font-weight: 600; background: rgba(255, 106, 36, 0.12); padding: 2px 8px; border-radius: 10px; border: 1px solid rgba(255, 106, 36, 0.3);">Observed FIRMS</span>
        </div>
        <div style="display: flex; gap: 0.55rem; overflow-x: auto;">
            {cards_html}
        </div>
    </div>
    """
    render_html(html)


def render_main_hero(
    title: str = "California Wildfire Prediction",
    subtitle: str = "Hourly environmental forecasting estimating satellite active fire detection (fire_next_24h) across California's 0.25° grid (~25 km) using Copernicus ERA5 reanalysis and multi-sensor Earth observation.",
    eyebrow: str = "🔥 CALIFORNIA WILDFIRE INTELLIGENCE",
):
    """Renders the hero banner with horizontal logo badge and pill."""
    b64 = get_logo_b64()
    img_tag = f'<img src="data:image/png;base64,{b64}" alt="Project Logo" />' if b64 else '<span style="font-size: 1.3rem;">🔥</span>'

    html = f"""
    <div class="app-hero-panel">
        <div class="app-hero-top-row">
            <div class="app-hero-tile">
                {img_tag}
            </div>
            <div class="app-hero-pill">{eyebrow}</div>
        </div>
        <h1 class="app-hero-title">{title}</h1>
        <p class="app-hero-subtitle">{subtitle}</p>
    </div>
    """
    render_html(html)


def render_section_header(title: str, tag: str = ""):
    """Renders a major section heading starting with an emoji."""
    tag_html = f'<span class="app-section-tag">// {tag}</span>' if tag else ""
    html = f"""
    <div class="app-section-header">
        <h2 class="app-section-title">{title}</h2>
        {tag_html}
    </div>
    """
    render_html(html)


def render_intelligence_card(
    icon: str,
    eyebrow: str,
    value: str,
    sub: str,
    border_color: str = COLOR_BORDER,
):
    """Renders a dashboard card with icon tile, small uppercase label, value, and supporting text."""
    html = f"""
    <div class="app-intel-card" style="border-color: {border_color};">
        <div>
            <div class="app-card-top">
                <div class="app-icon-tile">{icon}</div>
                <div class="app-card-eyebrow">{eyebrow}</div>
            </div>
            <div class="app-card-value">{value}</div>
        </div>
        <div class="app-card-sub">{sub}</div>
    </div>
    """
    render_html(html)


def render_workflow_card(step: str, title: str, description: str):
    """Renders a numbered pipeline/workflow card for process sections."""
    html = f"""
    <div class="app-workflow-card">
        <div class="app-workflow-step">{step}</div>
        <div class="app-workflow-title">{title}</div>
        <div class="app-workflow-desc">{description}</div>
    </div>
    """
    render_html(html)


def render_risk_legend(show_note: bool = True):
    """
    Renders an accessible, standard wildfire risk legend explaining the
    0.25° (~25 km) grid shapes, color semantics, thresholds, and distinct
    observed FIRMS detections.
    """
    note_html = ""
    if show_note:
        note_html = """
        <div class="app-legend-caption" style="border-top: 1px solid rgba(36, 69, 99, 0.4); padding-top: 0.45rem; margin-top: 0.45rem; font-size: 0.76rem; color: #94A3B8; line-height: 1.45;">
            <b>Grid & Prediction Scope:</b> Each shape represents a 0.25° (~25 km × 25 km) grid cell showing estimated probability of a spaceborne fire detection within the next 24 hours (<code>fire_next_24h</code>). Grid cells are <u>model-estimated statistical probabilities</u>, not confirmed fire locations or flame spread simulations. Observed detections represent verified thermal anomalies detected during satellite overpasses.
        </div>
        """

    html = f"""
    <div class="app-risk-legend-bar" style="background: {COLOR_DARK_NAVY}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 0.75rem 1rem; margin-top: 0.65rem; margin-bottom: 0.65rem;">
        <div style="font-size: 0.72rem; font-family: ui-monospace, monospace; color: {COLOR_TEXT_SECONDARY}; text-transform: uppercase; font-weight: 700; margin-bottom: 0.4rem;">
            MAP SYMBOLOGY // MODEL ESTIMATED 24H RISK CELLS (0.25° GRID) vs OBSERVED DETECTIONS:
        </div>
        <div class="app-legend-items" style="display: flex; flex-wrap: wrap; align-items: center; gap: 1rem; font-size: 0.82rem;">
            <div class="app-legend-item">
                <span class="app-legend-color-box" style="background-color: {COLOR_GREEN}; width: 14px; height: 14px; border-radius: 3px; display: inline-block;"></span>
                <span><b>Low Risk</b> (&lt;15%)</span>
            </div>
            <div class="app-legend-item">
                <span class="app-legend-color-box" style="background-color: {COLOR_GOLD}; width: 14px; height: 14px; border-radius: 3px; display: inline-block;"></span>
                <span><b>Moderate</b> (15%–25.1%)</span>
            </div>
            <div class="app-legend-item">
                <span class="app-legend-color-box" style="background-color: {COLOR_FIRE_ORANGE}; width: 14px; height: 14px; border-radius: 3px; display: inline-block;"></span>
                <span><b>High / Alert</b> (≥25.1%)</span>
            </div>
            <div class="app-legend-item">
                <span class="app-legend-color-box" style="background-color: {COLOR_CRIMSON}; width: 14px; height: 14px; border-radius: 3px; display: inline-block;"></span>
                <span><b>Extreme</b> (≥45%)</span>
            </div>
            <div class="app-legend-item" style="margin-left: auto; background: rgba(255, 59, 48, 0.12); border: 1px solid rgba(255, 59, 48, 0.35); border-radius: 14px; padding: 2px 10px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #FF3B30; box-shadow: 0 0 6px #FF3B30; margin-right: 4px;"></span>
                <span style="font-size: 0.78rem; font-weight: 700; color: #FCA5A5;">Observed FIRMS Detection (Satellite Overpass)</span>
            </div>
        </div>
        {note_html}
    </div>
    """
    render_html(html)


def render_operational_notice():
    """Renders scientific and physical scope boundary disclosures."""
    html = f"""
    <div class="app-notice-card">
        <div class="app-notice-title">Operational Scope & Scientific Governance Notice</div>
        <p class="app-notice-body">
            <b>Target Definition:</b> The model estimates whether a satellite active fire detection (VIIRS/MODIS via NASA FIRMS) occurs within the same 0.25° (~25 km × 25 km) grid cell during the subsequent 24 hours (<code>fire_next_24h</code>).<br/>
            <b>Advisory Restriction:</b> Predictions represent statistical probability based on Copernicus ERA5 reanalysis and spaceborne observations. They do <u>not</u> constitute confirmed active fires, emergency evacuation orders, or physical combustion simulations. Always refer to CAL FIRE and local emergency authorities for official wildfire guidance.
        </p>
    </div>
    """
    render_html(html)

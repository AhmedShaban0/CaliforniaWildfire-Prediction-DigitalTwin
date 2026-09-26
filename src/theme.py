"""
Theme and Design System for California Wildfire Prediction System.
Implements the exact layout, visual treatment, and color tokens:
- Strict Left-to-Right (LTR) numeric controls and layout enforcement.
- Left sidebar with button-style navigation and enlarged branded project card.
- Safe HTML rendering eliminating CommonMark 4-space code block leaks.
- Main hero banner, emoji section headings, and dashboard cards.
- Exact brand tokens (#081827, #071A2C, #102A43, #183B56, #29465E, #F8FAFC, #A9BACB, etc.).
"""
import streamlit as st
import os
import base64
import textwrap

# Exact Color Tokens
COLOR_CANVAS_BG = "#081827"        # Main background
COLOR_DARK_NAVY = "#071A2C"        # Deep background / Sidebar
COLOR_DEEP_NAVY = "#102A43"        # Main navy surface
COLOR_ELEVATED = "#183B56"         # Elevated surface / Active states
COLOR_BORDER = "#29465E"           # Border
COLOR_TEXT_PRIMARY = "#F8FAFC"     # Primary text
COLOR_TEXT_SECONDARY = "#A9BACB"   # Secondary text

# Wildfire Risk & Accent Semantics
COLOR_CRIMSON = "#C51F32"          # Crimson (Extreme risk)
COLOR_FIRE_RED = "#E52B24"         # Fire red (High risk / Alert threshold)
COLOR_FIRE_ORANGE = "#F4511E"      # Fire orange (Fire emphasis / detections)
COLOR_WARM_ORANGE = "#FF7A24"      # Warm orange (Warning / accent)
COLOR_GREEN = "#35B779"            # Green (Low risk / baseline)
COLOR_GOLD = "#F6B73C"             # Gold (Moderate risk / informational)

# Exact Gradients
GRADIENT_HERO = "linear-gradient(110deg, #071A2C 0%, #102A43 55%, #183B56 100%)"
GRADIENT_FIRE = "linear-gradient(110deg, #C51F32 0%, #E52B24 50%, #FF7A24 100%)"
GRADIENT_CARD = "linear-gradient(135deg, #102A43 0%, #071A2C 100%)"

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


def render_html(html_str: str):
    """
    Safely renders user-facing HTML in Streamlit.
    Strips leading and trailing whitespace from every line to ensure CommonMark
    never mistakenly interprets HTML tags as indented markdown code blocks,
    and delegates to st.html for native HTML parsing without markdown interference.
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

/* Slider Component Unified Layout & Polish */
div[data-testid="stSlider"] {{
    direction: ltr !important;
    text-align: left !important;
    margin-bottom: 0.65rem !important;
}}

div[data-testid="stSlider"] [data-testid="stWidgetLabel"] {{
    min-height: 26px !important;
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

/* Sidebar styling - Full height, ~290px wide, dark navy */
[data-testid="stSidebar"] {{
    min-width: 290px !important;
    max-width: 290px !important;
    background-color: {COLOR_DARK_NAVY} !important;
    border-right: 1px solid {COLOR_BORDER} !important;
}}

[data-testid="stSidebarContent"] {{
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    padding-left: 0.85rem !important;
    padding-right: 0.85rem !important;
    overflow-y: hidden !important;
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
    font-weight: 600 !important;
    letter-spacing: -0.015em;
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

/* Main Page Hero Banner - Wide horizontal gradient panel */
.app-hero-panel {{
    background: {GRADIENT_HERO};
    border: 1px solid {COLOR_BORDER};
    border-left: 4px solid {COLOR_FIRE_RED};
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.75rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}}

.app-hero-top-row {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.85rem;
    flex-wrap: wrap;
}}

.app-hero-tile {{
    background-color: #FFFFFF;
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 6px 12px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    height: 54px;
}}

.app-hero-tile img {{
    height: 42px;
    width: auto;
    object-fit: contain;
    display: block;
}}

.app-hero-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background-color: {COLOR_ELEVATED};
    color: {COLOR_WARM_ORANGE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 20px;
    padding: 0.35rem 0.85rem;
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: ui-monospace, SFMono-Regular, "Roboto Mono", Menlo, monospace;
}}

.app-hero-title {{
    font-size: 2.25rem !important;
    font-weight: 700 !important;
    color: {COLOR_TEXT_PRIMARY} !important;
    margin: 0 0 0.45rem 0 !important;
    letter-spacing: -0.025em;
    line-height: 1.15;
}}

.app-hero-subtitle {{
    color: {COLOR_TEXT_SECONDARY};
    font-size: 0.98rem;
    line-height: 1.55;
    margin: 0;
    max-width: 950px;
}}

/* Section Titles with Emoji Prefix */
.app-section-header {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.75rem 0 1rem 0;
}}

.app-section-title {{
    font-size: 1.35rem !important;
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
}}

/* Dashboard Cards */
.app-intel-card {{
    background: {GRADIENT_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    padding: 1.25rem 1.35rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.15s ease, border-color 0.15s ease;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
}}

.app-intel-card:hover {{
    border-color: {COLOR_WARM_ORANGE};
    transform: translateY(-2px);
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

/* Process / Workflow Card */
.app-workflow-card {{
    background: {GRADIENT_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    padding: 1.25rem 1.35rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}}

.app-workflow-step {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.74rem;
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

/* Branded Project Card in Sidebar with Larger Logo and Clean Spacing */
.app-sidebar-bottom-card {{
    background: {GRADIENT_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    padding: 0.65rem 0.65rem;
    text-align: center;
    margin-top: 0.75rem;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
}}

.app-sidebar-logo-tile {{
    background-color: #FFFFFF;
    border: 1px solid {COLOR_BORDER};
    border-radius: 10px;
    padding: 8px 12px;
    margin: 0 auto;
    width: 100%;
    max-width: 260px;
    height: 115px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}}

.app-sidebar-logo-tile img {{
    max-height: 98px;
    max-width: 100%;
    width: auto;
    object-fit: contain;
    display: block;
}}

/* Form elements & inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {{
    background-color: {COLOR_DARK_NAVY} !important;
    border-color: {COLOR_BORDER} !important;
    color: {COLOR_TEXT_PRIMARY} !important;
    border-radius: 6px !important;
}}

/* Streamlit buttons - Global styling */
div.stButton > button {{
    background-color: {COLOR_ELEVATED};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    padding: 0.45rem 1.15rem;
    font-size: 0.88rem;
    font-weight: 500;
    transition: all 0.15s ease-in-out;
}}

div.stButton > button:hover {{
    border-color: {COLOR_FIRE_RED};
    background-color: {COLOR_DEEP_NAVY};
    color: #FFFFFF;
}}

div.stButton > button[kind="primary"] {{
    background: {GRADIENT_FIRE};
    color: #FFFFFF;
    border: none;
    font-weight: 600;
}}

div.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 0 14px rgba(229, 43, 36, 0.45);
}}

/* Compact Button-style Navigation in Sidebar */
[data-testid="stSidebar"] div.stButton {{
    margin-bottom: 0.2rem !important;
}}

[data-testid="stSidebar"] div.stButton > button {{
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    text-align: left !important;
    padding: 0.38rem 0.75rem !important;
    border-radius: 6px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    min-height: 35px !important;
    line-height: 1.25 !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
    background-color: transparent !important;
    border: 1px solid transparent !important;
    color: {COLOR_TEXT_SECONDARY} !important;
}}

[data-testid="stSidebar"] div.stButton > button:hover {{
    background-color: rgba(24, 59, 86, 0.5) !important;
    color: #FFFFFF !important;
    border-color: {COLOR_BORDER} !important;
    transform: none !important;
}}

/* Active navigation button in sidebar */
[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
    background: {COLOR_ELEVATED} !important;
    color: #FFFFFF !important;
    border: 1px solid {COLOR_WARM_ORANGE} !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
}}

[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {{
    background: {COLOR_ELEVATED} !important;
    border-color: {COLOR_WARM_ORANGE} !important;
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


def render_sidebar_project_card():
    """
    Renders the branded project card near the bottom of the sidebar.
    Presents the authentic project PNG logo in an elevated light tile.
    Text beneath the logo (title & tagline) is cleanly removed per user request.
    """
    b64 = get_logo_b64()
    img_tag = f'<img src="data:image/png;base64,{b64}" alt="California Wildfire Prediction Logo" />' if b64 else '<span style="font-size: 2rem;">🔥</span>'

    html = f"""
    <div class="app-sidebar-bottom-card">
        <div class="app-sidebar-logo-tile">
            {img_tag}
        </div>
    </div>
    """
    render_html(html)


def render_main_hero(
    title: str = "California Wildfire Prediction",
    subtitle: str = "Hourly environmental forecasting estimating satellite active fire detection (fire_next_24h) across California's 0.25° grid (~25 km) using Copernicus ERA5 reanalysis and multi-sensor Earth observation.",
    eyebrow: str = "🔥 CALIFORNIA WILDFIRE INTELLIGENCE",
):
    """
    Renders the prominent wide landing hero matching the reference screenshot:
    - Wide horizontal panel spanning the main content width
    - Project logo in a compact light tile at the upper left of the hero content
    - Pill beside the logo with label
    - Large bold white title below the logo row
    - One/two-line supporting description in muted blue-gray text
    """
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
    """Renders a refined dashboard card with icon tile, small uppercase label, value, and supporting text."""
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

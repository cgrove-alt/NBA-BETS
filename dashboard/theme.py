"""
NBA Betting Dashboard - Centralized Theme Configuration

Single source of truth for colors, spacing, and design tokens.
"""

# Prop types available
PROP_TYPES = ["Points", "Rebounds", "Assists", "3PM", "PRA"]

# Dark theme colors
COLORS = {
    # Backgrounds
    "bg_primary": "#0d1117",
    "bg_secondary": "#161b22",
    "bg_tertiary": "#21262d",
    "bg_card": "#1c2128",
    "bg_card_hover": "#262c36",
    "bg_hover": "#30363d",

    # Text
    "text_primary": "#e6edf3",
    "text_secondary": "#8b949e",
    "text_muted": "#6e7681",

    # Accents
    "accent_primary": "#58a6ff",
    "accent_primary_hover": "#79b8ff",
    "accent_success": "#3fb950",
    "accent_warning": "#d29922",
    "accent_danger": "#f85149",

    # Borders
    "border_color": "#30363d",
    "border_light": "#21262d",

    # Semi-transparent for backgrounds
    "success_light": "rgba(63, 185, 80, 0.15)",
    "warning_light": "rgba(210, 153, 34, 0.15)",
    "danger_light": "rgba(248, 81, 73, 0.15)",
    "primary_light": "rgba(88, 166, 255, 0.15)",
}

# Spacing system (in pixels)
SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
    "xxl": "48px",
}

# Border radius
RADIUS = {
    "sm": "6px",
    "md": "10px",
    "lg": "12px",
    "xl": "16px",
    "full": "9999px",
}

# Font sizes
FONT_SIZES = {
    "xs": "11px",
    "sm": "13px",
    "md": "15px",
    "lg": "18px",
    "xl": "24px",
    "xxl": "32px",
}

# Gradients
GRADIENTS = {
    "success": "linear-gradient(135deg, #238636 0%, #3fb950 100%)",
    "danger": "linear-gradient(135deg, #da3633 0%, #f85149 100%)",
    "primary": "linear-gradient(135deg, #1f6feb 0%, #58a6ff 100%)",
    "warning": "linear-gradient(135deg, #9e6a03 0%, #d29922 100%)",
    "neutral": "linear-gradient(135deg, #30363d 0%, #484f58 100%)",
}

# Shadows
SHADOWS = {
    "card": "0 3px 6px rgba(0, 0, 0, 0.4)",
    "hover": "0 8px 24px rgba(0, 0, 0, 0.5)",
    "glow_primary": "0 0 15px rgba(88, 166, 255, 0.3)",
    "glow_success": "0 0 15px rgba(63, 185, 80, 0.3)",
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "high": 70,
    "medium": 50,
    "low": 0,
}

# Confidence colors
def get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level."""
    if confidence >= CONFIDENCE_THRESHOLDS["high"]:
        return COLORS["accent_success"]
    elif confidence >= CONFIDENCE_THRESHOLDS["medium"]:
        return COLORS["accent_warning"]
    return COLORS["text_muted"]

def get_confidence_gradient(confidence: float) -> str:
    """Get gradient based on confidence level."""
    if confidence >= CONFIDENCE_THRESHOLDS["high"]:
        return GRADIENTS["success"]
    elif confidence >= CONFIDENCE_THRESHOLDS["medium"]:
        return GRADIENTS["warning"]
    return GRADIENTS["neutral"]

# Pick colors
def get_pick_color(pick: str) -> str:
    """Get color for OVER/UNDER pick."""
    if pick == "OVER":
        return COLORS["accent_success"]
    elif pick == "UNDER":
        return COLORS["accent_danger"]
    return COLORS["text_muted"]

def get_pick_bg(pick: str) -> str:
    """Get background color for OVER/UNDER pick."""
    if pick == "OVER":
        return COLORS["success_light"]
    elif pick == "UNDER":
        return COLORS["danger_light"]
    return "transparent"

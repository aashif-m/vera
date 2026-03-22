"""
Vera - Lightweight Fact Checker
Streamlit Frontend with CoT (Chain-of-Thought) Mode Toggle
"""

import streamlit as st
import requests
import os

# Configuration
API_URL = os.getenv("API_URL", "http://api:8000")

# Page config
st.set_page_config(
    page_title="Vera - Fact Checker",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .claim-card {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .supported {
        border-left: 4px solid #22c55e;
        background-color: rgba(34, 197, 94, 0.1);
    }
    .refuted {
        border-left: 4px solid #ef4444;
        background-color: rgba(239, 68, 68, 0.1);
    }
    .nei {
        border-left: 4px solid #6b7280;
        background-color: rgba(107, 114, 128, 0.1);
    }
    .opinion {
        border-left: 4px solid #8b5cf6;
        background-color: rgba(139, 92, 246, 0.1);
    }
    .score-container {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .big-score {
        font-size: 4rem;
        font-weight: 700;
    }
    .reasoning-box {
        background-color: rgba(99, 102, 241, 0.1);
        border-left: 3px solid #6366f1;
        padding: 0.75rem;
        margin-top: 0.5rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    .reason-tag {
        background-color: rgba(99, 102, 241, 0.2);
        color: #4f46e5;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .mode-indicator {
        text-align: center;
        padding: 0.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .mode-standard {
        background-color: rgba(34, 197, 94, 0.1);
        color: #16a34a;
    }
    .mode-reasoning {
        background-color: rgba(99, 102, 241, 0.1);
        color: #4f46e5;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔍 Vera</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Lightweight AI Fact-Checker • Decompose → Retrieve → Verify</p>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Reasoning toggle
    use_reasoning = st.toggle(
        "🧠 Enable Reasoning (CoT)",
        value=False,
        help="When enabled, the AI will explain its reasoning for each classification and verdict."
    )
    
    st.markdown("---")
    
    # Mode indicator
    if use_reasoning:
        st.markdown("""
        <div class="mode-indicator mode-reasoning">
            <strong>🧠 Reasoning Mode</strong><br>
            <small>Shows AI reasoning for classifications</small>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Models in use:**")
        st.markdown("- 🔹 Decomposer: LFM 2.5 (CoT)")
        st.markdown("- 🔹 Verifier: LFM 2.5 (CoT)")
    else:
        st.markdown("""
        <div class="mode-indicator mode-standard">
            <strong>⚡ Standard Mode</strong><br>
            <small>Fast verification without reasoning</small>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Models in use:**")
        st.markdown("- 🔹 Decomposer: LFM 2.5")
        st.markdown("- 🔹 Verifier: ModernBERT-Zeroshot")
    
    st.markdown("---")
    st.markdown("**Search:** Tavily AI")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    text_input = st.text_area(
        "Paste text to fact-check",
        height=150,
        placeholder="Enter a paragraph, article, or claim to analyze...\n\nExample: The Eiffel Tower was built in 1889 by Gustave Eiffel. It is the tallest structure in France."
    )

with col2:
    st.markdown("### Quick Stats")
    if use_reasoning:
        st.info("🧠 **Reasoning Mode**\nAI explanations enabled")
    else:
        st.success("⚡ **Standard Mode**\nFast verification")

# Check button
check_clicked = st.button("✨ Check Facts", type="primary", use_container_width=True)

# Divider
st.markdown("---")

# Results section
if check_clicked and text_input:
    with st.spinner("🔄 Analyzing text..." + (" (with reasoning)" if use_reasoning else "")):
        try:
            response = requests.post(
                f"{API_URL}/check",
                json={
                    "text": text_input,
                    "use_reasoning": use_reasoning
                },
                timeout=180  # Longer timeout for reasoning mode
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Mode indicator
                mode = data.get("mode", "standard")
                if mode == "reasoning":
                    st.markdown("""
                    <div class="mode-indicator mode-reasoning">
                        🧠 Results generated with <strong>Reasoning Mode</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Score display
                col_score, col_stats = st.columns([1, 2])
                
                with col_score:
                    score = data["factuality_score"]
                    if score >= 70:
                        score_color = "#22c55e"  # Green
                    elif score >= 40:
                        score_color = "#f59e0b"  # Orange
                    else:
                        score_color = "#ef4444"  # Red
                    
                    st.markdown(f"""
                    <div class="score-container">
                        <div class="big-score">{score}%</div>
                        <div>Factuality Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_stats:
                    st.markdown("### Breakdown")
                    
                    stat_cols = st.columns(4)
                    with stat_cols[0]:
                        st.metric("✅ Supported", data["supported_count"])
                    with stat_cols[1]:
                        st.metric("❌ Refuted", data["refuted_count"])
                    with stat_cols[2]:
                        st.metric("⚪ Unclear", data["nei_count"])
                    with stat_cols[3]:
                        st.metric("💭 Opinions", data["opinion_count"])
                    
                    total = len(data["claims"])
                    st.progress(data["supported_count"] / max(total, 1) if total > 0 else 0)
                
                st.markdown("---")
                
                # Claims list
                st.markdown(f"### 📋 Claims Found ({len(data['claims'])})")
                
                for i, claim in enumerate(data["claims"]):
                    verdict = claim["verdict"]
                    
                    # Emoji and styling based on verdict
                    if verdict == "SUPPORTED":
                        emoji = "✅"
                        css_class = "supported"
                    elif verdict == "REFUTED":
                        emoji = "❌"
                        css_class = "refuted"
                    elif verdict == "OPINION":
                        emoji = "💭"
                        css_class = "opinion"
                    else:
                        emoji = "⚪"
                        css_class = "nei"
                    
                    with st.container():
                        confidence = claim.get("confidence", 0) or 0
                        
                        # Main claim card
                        claim_html = f"""
                        <div class="claim-card {css_class}">
                            <strong>{emoji} {verdict}</strong> ({confidence:.0%} confidence)<br>
                            <em>"{claim['quote']}"</em>
                        """
                        
                        # Add reason tag if available (CoT mode)
                        reason = claim.get("reason")
                        if reason:
                            claim_html += f'<div class="reason-tag">💡 {reason}</div>'
                        
                        claim_html += "</div>"
                        st.markdown(claim_html, unsafe_allow_html=True)
                        
                        with st.expander("View details"):
                            st.markdown(f"**Atomic Claim:** {claim['atomic_claim']}")
                            st.markdown(f"**Type:** {claim['claim_type']}")
                            
                            # Show decomposer reasoning if available
                            if reason:
                                st.markdown(f"**Classification Reason:** {reason}")
                            
                            if claim.get('evidence'):
                                st.markdown(f"**Evidence:** {claim['evidence']}")
                            if claim.get('source'):
                                st.markdown(f"**Source:** [{claim['source']}]({claim['source']})")
                            
                            # Show verifier reasoning if available (CoT mode)
                            verifier_reasoning = claim.get("reasoning")
                            if verifier_reasoning:
                                st.markdown("**Verifier Reasoning:**")
                                st.markdown(f"""
                                <div class="reasoning-box">
                                    🧠 {verifier_reasoning}
                                </div>
                                """, unsafe_allow_html=True)
                
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to API. Make sure the backend is running.")
        except requests.exceptions.Timeout:
            st.error("⚠️ Request timed out. The reasoning mode may take longer - please try again.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif check_clicked:
    st.warning("Please enter some text to fact-check.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Vera • A Lightweight Fact-Checking Pipeline</p>",
    unsafe_allow_html=True
)

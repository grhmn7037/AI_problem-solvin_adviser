# frontend/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from collections import Counter
import random

# --- إضافة مسار src إلى sys.path ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, '..'))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

# --- استيراد الكلاسات الضرورية ---
analyzer_instance = None
recommender_instance = None
analyzer_error_msg = None
recommender_error_msg = None

try:
    from src.analysis.problem_analyzer import ProblemAnalyzer
    from src.analysis.recommendation_engine import RecommendationEngine
except ImportError as e_import:
    print(f"خطأ فادح في استيراد الوحدات: {e_import}")
    analyzer_error_msg = f"خطأ في استيراد الوحدات: {e_import}. تحقق من الطرفية لمزيد من التفاصيل."
except Exception as e_general_import:
    print(f"خطأ غير متوقع أثناء الاستيراد: {e_general_import}")
    analyzer_error_msg = f"خطأ غير متوقع أثناء الاستيراد: {e_general_import}"

# --- إعدادات الصفحة ---
st.set_page_config(page_title="مستشار المشاكل الذكي", layout="wide", page_icon="🤖")

# --- قاموس الترجمات (مع إضافات للشات بوت) ---
TRANSLATIONS = {
    "ar": {
        "page_title": "مستشار المشاكل الذكي", "app_header": "🤖 مستشار المشاكل الذكي",
        "app_subheader": "أدخل تفاصيل المشكلة الجديدة للحصول على تحليل وتوصيات مبدئية.",
        "form_header": "📝 تفاصيل المشكلة الجديدة", "problem_title_label": "عنوان المشكلة*",
        "problem_title_help": "مثال: الشبكة بطيئة في قسم المبيعات",
        "description_label": "الوصف الأولي للمشكلة*", "description_help": "صف المشكلة بالتفصيل قدر الإمكان.",
        "domain_label": "مجال المشكلة", "domain_help": "اختر المجال الأقرب للمشكلة.",
        "domain_options": ["", "تقني", "إداري", "مالي", "شخصي", "تعليم عالي - تكنولوجيا", "شخصي - صيانة سيارة", "أخرى"],
        "complexity_label": "مستوى التعقيد المتوقع", "complexity_help": "كيف تقيم تعقيد هذه المشكلة؟",
        "complexity_options": ["", "بسيط", "متوسط", "عالي"],
        "status_label": "الحالة الحالية للمشكلة",
        "status_options": ["جديدة", "مفتوحة", "قيد التحليل", "مغلقة", "معلقة"],
        "problem_source_label": "مصدر المشكلة", "problem_source_help": "مثال: شكاوى المستخدمين، ملاحظة شخصية",
        "optional_details_header": "إضافة تفاصيل اختيارية (نصوص وصفية)",
        "estimated_cost_label": "التكلفة المقدرة (نص وصفي)",
        "estimated_cost_help": "مثال: منخفض، 1000-1500 دولار، حوالي 500 ريال",
        "estimated_time_label": "الوقت المقدر للتنفيذ (نص وصفي)",
        "estimated_time_help": "مثال: فوري، 3 أيام، 2-4 أسابيع",
        "overall_budget_label": "الميزانية الإجمالية (نص وصفي)", "overall_budget_help": "مثال: 2000 دولار، متوسط",
        "stakeholders_label": "الأطراف المعنية", "refined_statement_label": "البيان النهائي للمشكلة (إذا توفر)",
        "submit_button_label": "🔍 تحليل المشكلة",
        "error_title_description_missing": "يرجى إدخال عنوان المشكلة ووصفها على الأقل.",
        "analysis_results_header": "📊 نتائج التحليل", "input_title_label": "العنوان المدخل:",
        "kmeans_cluster_label": "تصنيف عنقود K-Means:", "cluster_summary_label": "ملخص العنقود:",
        "bertopic_topic_label": "تصنيف موضوع BERTopic:", "topic_summary_label": "ملخص الموضوع:",
        "recommendations_header": "💡 التوصيات المقترحة",
        "kmeans_recommendations_label": "التوصيات بناءً على عنقود K-Means المشابه:",
        "bertopic_recommendations_label": "التوصيات بناءً على موضوع BERTopic المشابه:",
        "no_specific_recommendations": "لم يتم العثور على توصيات محددة بناءً على التصنيفات الحالية.",
        "recommender_unavailable_warning": "محرك التوصيات غير متاح أو لا توجد بيانات تاريخية كافية للتوصيات.",
        "sidebar_about_header": "عن المشروع",
        "sidebar_about_info": "هذا التطبيق يستخدم نماذج التعلم الآلي لتحليل المشاكل وتقديم رؤى وتوصيات مبدئية للمساعدة في إدارتها بشكل أفضل.",
        "sidebar_how_it_works_header": "كيف يعمل؟",
        "sidebar_how_it_works_steps": """1.  أدخل تفاصيل المشكلة الجديدة.\n2.  يقوم النظام بمعالجة المدخلات.\n3.  يستخدم نموذج K-Means لتحديد عنقود المشاكل المشابهة.\n4.  يستخدم نموذج BERTopic لتحديد الموضوع الرئيسي للمشكلة.\n5.  يعرض ملخصًا لخصائص هذا العنقود والموضوع.\n6.  (اختياري) يقترح حلولاً أو دروسًا مستفادة من مشاكل تاريخية مشابهة.""",
        "loading_analyzer_once": "يتم الآن تحميل ProblemAnalyzer (يحدث مرة واحدة أو عند تغيير الكود)...",
        "loading_recommender_once": "يتم الآن تحميل RecommendationEngine (يحدث مرة واحدة أو عند تغيير الكود)...",
        "analyzer_load_fail_warning": "فشل تحميل واحد أو أكثر من نماذج التعلم الآلي الأساسية في ProblemAnalyzer: {details}",
        "profile_data_load_warning": "تحذير Streamlit: لم يتم تحميل بيانات الملفات التعريفية في ProblemAnalyzer. قد تكون الملخصات محدودة.",
        "analyzer_init_success": "ProblemAnalyzer تم تحميله بنجاح.",
        "analyzer_init_fatal_error": "حدث خطأ فادح واستثناء عام أثناء تهيئة ProblemAnalyzer: {e}",
        "recommender_load_fail_warning": "فشل تحميل البيانات التاريخية في RecommendationEngine. قد لا تتوفر توصيات.",
        "recommender_init_success": "RecommendationEngine تم تحميله بنجاح.",
        "recommender_init_fatal_error": "حدث خطأ فادح أثناء تهيئة RecommendationEngine: {e}",
        "app_components_load_fail": "فشل تحميل المكونات الأساسية للتطبيق. يرجى مراجعة الطرفية للمزيد من التفاصيل.",
        "processing_problem_spinner": "جاري تحليل المشكلة...",
        "searching_recommendations_spinner": "جاري البحث عن توصيات...",
        "not_specified_placeholder": "غير محدد", "no_summary_available": "لا يوجد ملخص متاح.",
        "error_during_analysis": "حدث خطأ أثناء تحليل المشكلة",  # مفتاح ترجمة جديد
        "bertopic_noise_warning": "المشكلة صُنفت كموضوع ضوضاء/غير محدد ({topic_id}) بواسطة BERTopic، لا توجد توصيات موضوعية محددة.",
        "no_kmeans_cluster_recs_warning": "لم يتم العثور على مشاكل أخرى في عنقود K-Means رقم {cluster_id} لتقديم توصيات محددة.",
        "no_bertopic_topic_recs_warning": "لم يتم العثور على مشاكل أخرى في موضوع BERTopic رقم {topic_id} لتقديم توصيات محددة.",
        "chatbot_title": "المساعد الذكي", "chatbot_toggle_open": "💬 فتح المساعد",
        "chatbot_toggle_close": "✕ إغلاق المساعد",
        "chatbot_input_placeholder": "اكتب سؤالك هنا...", "chatbot_send_button": "إرسال",
        "chatbot_greeting": "مرحباً! كيف يمكنني مساعدتك اليوم؟ (اكتب 'مساعدة' لعرض الخيارات)",
        "faq_help_options": "يمكنك سؤالي عن:\n- كيفية إضافة مشكلة\n- تعديل مشكلة\n- حالة المشكلة\n- الكلمات المفتاحية",
        "faq_how_to_add_problem": "لإضافة مشكلة، املأ الحقول في النموذج الرئيسي واضغط 'تحليل المشكلة'.",
        "faq_what_is_problem_status": "حالة المشكلة توضح مرحلة معالجتها.",
        "faq_edit_problem": "حالياً، لا يمكن تعديل المشاكل من هذه الواجهة.",
        "faq_keywords": "الكلمات المفتاحية تساعد في فهم محتوى المشكلة.",
        "faq_thank_you": "على الرحب والسعة!", "faq_default_reply": "عذرًا، لم أفهم سؤالك. جرب 'مساعدة'."
    },
    "en": {
        "page_title": "Intelligent Problem Advisor", "app_header": "🤖 Intelligent Problem Advisor",
        "app_subheader": "Enter new problem details for analysis and recommendations.",
        "form_header": "📝 New Problem Details", "problem_title_label": "Problem Title*",
        "problem_title_help": "e.g., Network is slow in sales department",
        "description_label": "Initial Problem Description*", "description_help": "Describe in detail.",
        "domain_label": "Problem Domain", "domain_help": "Select the closest domain.",
        "domain_options": ["", "Technical", "Administrative", "Financial", "Personal", "Higher Ed - Tech",
                           "Personal - Car Maintenance", "Other"],
        "complexity_label": "Expected Complexity", "complexity_help": "How complex is this problem?",
        "complexity_options": ["", "Simple", "Medium", "High"],
        "status_label": "Current Problem Status",
        "status_options": ["New", "Open", "Under Analysis", "Closed", "Pending"],
        "problem_source_label": "Problem Source", "problem_source_help": "e.g., User complaints, Personal observation",
        "optional_details_header": "Add Optional Details (Descriptive Texts)",
        "estimated_cost_label": "Estimated Cost (Descriptive)", "estimated_cost_help": "e.g., Low, 1000-1500 USD",
        "estimated_time_label": "Estimated Time to Implement (Descriptive)",
        "estimated_time_help": "e.g., Immediate, 3 days",
        "overall_budget_label": "Overall Budget (Descriptive)", "overall_budget_help": "e.g., 2000 USD, Medium",
        "stakeholders_label": "Stakeholders Involved", "refined_statement_label": "Refined Problem Statement",
        "submit_button_label": "🔍 Analyze Problem",
        "error_title_description_missing": "Please enter title and description.",
        "analysis_results_header": "📊 Analysis Results", "input_title_label": "Input Title:",
        "kmeans_cluster_label": "K-Means Cluster:", "cluster_summary_label": "Cluster Summary:",
        "bertopic_topic_label": "BERTopic Topic:", "topic_summary_label": "Topic Summary:",
        "recommendations_header": "💡 Suggested Recommendations",
        "kmeans_recommendations_label": "Based on similar K-Means Cluster:",
        "bertopic_recommendations_label": "Based on similar BERTopic Topic:",
        "no_specific_recommendations": "No specific recommendations found.",
        "recommender_unavailable_warning": "Recommender unavailable or insufficient historical data.",
        "sidebar_about_header": "About",
        "sidebar_about_info": "This app uses ML to analyze problems and provide insights.",
        "sidebar_how_it_works_header": "How It Works",
        "sidebar_how_it_works_steps": "1. Enter problem details.\n2. System processes input.\n3. K-Means identifies similar problem clusters.\n4. BERTopic identifies the main topic.\n5. Summaries are displayed.\n6. (Optional) Suggestions from historical problems.",
        "loading_analyzer_once": "Loading ProblemAnalyzer...",
        "loading_recommender_once": "Loading RecommendationEngine...",
        "analyzer_load_fail_warning": "Failed to load ML models in ProblemAnalyzer: {details}",
        "profile_data_load_warning": "Warning: Profile data not loaded in ProblemAnalyzer.",
        "analyzer_init_success": "ProblemAnalyzer loaded.",
        "analyzer_init_fatal_error": "Fatal error initializing ProblemAnalyzer: {e}",
        "recommender_load_fail_warning": "Failed to load historical data in RecommendationEngine.",
        "recommender_init_success": "RecommendationEngine loaded.",
        "recommender_init_fatal_error": "Fatal error initializing RecommendationEngine: {e}",
        "app_components_load_fail": "Failed to load app components. Check terminal.",
        "processing_problem_spinner": "Analyzing problem...",
        "searching_recommendations_spinner": "Searching for recommendations...",
        "not_specified_placeholder": "Not Specified", "no_summary_available": "No summary available.",
        "error_during_analysis": "Error during analysis",  # مفتاح ترجمة جديد
        "bertopic_noise_warning": "Problem classified as noise ({topic_id}) by BERTopic, no specific topical recommendations.",
        "no_kmeans_cluster_recs_warning": "No other problems found in K-Means cluster {cluster_id} for recommendations.",
        "no_bertopic_topic_recs_warning": "No other problems found in BERTopic topic {topic_id} for recommendations.",
        "chatbot_title": "Smart Assistant", "chatbot_toggle_open": "💬 Open", "chatbot_toggle_close": "✕",
        "chatbot_input_placeholder": "Type here...", "chatbot_send_button": "Send",
        "chatbot_greeting": "Hello! How can I help? (Type 'help')",
        "faq_help_options": "You can ask about:\n- Adding a problem\n- Editing a problem\n- Problem status\n- Keywords",
        "faq_how_to_add_problem": "To add a problem, fill the main form and click 'Analyze Problem'.",
        "faq_what_is_problem_status": "Problem status shows its current stage.",
        "faq_edit_problem": "Currently, editing problems is not supported via this interface.",
        "faq_keywords": "Keywords help in understanding problem content.",
        "faq_thank_you": "You're welcome!", "faq_default_reply": "Sorry, I didn't understand. Try 'help'."
    }
}


def get_translation(lang_code, key, **kwargs):
    default_lang_translations = TRANSLATIONS.get("en", {})
    lang_translations = TRANSLATIONS.get(lang_code, default_lang_translations)
    message_template_or_list = lang_translations.get(key, default_lang_translations.get(key, None))
    if message_template_or_list is None: return f"<{key}_NOT_FOUND>"
    if isinstance(message_template_or_list, str):
        try:
            return message_template_or_list.format(**kwargs)
        except KeyError:
            return message_template_or_list
    elif isinstance(message_template_or_list, list):
        return message_template_or_list
    else:
        return str(message_template_or_list)


language_options = {"العربية": "ar", "English": "en"}
selected_language_name = st.sidebar.selectbox(label="اختر اللغة / Select Language",
                                              options=list(language_options.keys()))
LANG_CODE = language_options[selected_language_name]

direction_css = "rtl" if LANG_CODE == "ar" else "ltr"
text_align_css = "right" if LANG_CODE == "ar" else "left"
chat_window_position_css = "left: 20px; right: auto;" if LANG_CODE == "ar" else "right: 20px; left: auto;"
user_message_margin_css = "margin-left: auto !important; margin-right: 0 !important; border-bottom-right-radius: 0 !important; border-bottom-left-radius: 15px !important;" if LANG_CODE == "ar" else "margin-right: auto !important; margin-left: 0 !important; border-bottom-left-radius: 0 !important; border-bottom-right-radius: 15px !important;"
bot_message_margin_css = "margin-right: auto !important; margin-left: 0 !important; border-bottom-left-radius: 0 !important; border-bottom-right-radius: 15px !important;" if LANG_CODE == "ar" else "margin-left: auto !important; margin-right: 0 !important; border-bottom-right-radius: 0 !important; border-bottom-left-radius: 15px !important;"
input_button_margin_css = "margin-right: 8px !important; margin-left: 0 !important;" if LANG_CODE == "ar" else "margin-left: 8px !important; margin-right: 0 !important;"

# ... (بعد تعريف LANG_CODE والمتغيرات المعتمدة عليه مثل direction_css, text_align_css, إلخ) ...

st.markdown(f"""
    <style>
    /* =========================================== */
    /* === General RTL/LTR and Base Styles === */
    /* =========================================== */
    body, html, [class*="st-"], div[data-testid="stAppViewContainer"], div[data-testid="stHeader"] {{
        direction: {direction_css} !important;
    }}

    /* General text alignment for most elements */
    div, p, span, h1, h2, h3, h4, h5, h6, label, li, th, td {{
        text-align: {text_align_css} !important;
        direction: {direction_css} !important;
    }}

    /* Input fields alignment */
    .stTextInput input, 
    .stTextArea textarea, 
    .stSelectbox div[data-baseweb="select"] > div,
    .stDateInput input,
    .stNumberInput input {{
        text-align: {text_align_css} !important; 
        direction: {direction_css} !important;
    }}

    /* Selectbox dropdown text alignment */
    div[data-baseweb="select"] > div > div > div {{ /* Inner text of selected option */
        direction: {direction_css} !important;
        text-align: {text_align_css} !important;
    }}
    div[data-baseweb="popover"] {{ /* The dropdown list itself */
        direction: {direction_css} !important;
        text-align: {text_align_css} !important;
    }}

    /* Labels and Buttons text direction */
    label, 
    .stButton > button, 
    .stFormSubmitButton > button {{
        /* direction is inherited, text-align might be needed if not aligning correctly */
         text-align: {text_align_css} !important; /* To ensure label text aligns */
    }}
    /* Ensure Streamlit button content (text/icon) itself respects direction */
    .stButton > button > div, .stFormSubmitButton > button > div {{
        direction: {direction_css} !important;
    }}


    /* Markdown specific alignment */
    .stMarkdown, 
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] ul,
    div[data-testid="stMarkdownContainer"] ol {{
        text-align: {text_align_css} !important;
        direction: {direction_css} !important;
    }}

    /* Center main title */
    h1 {{
        text-align: center !important;
    }}

    /* Alerts (info, warning, error) */
    .stAlert {{
        text-align: {text_align_css} !important;
        direction: {direction_css} !important;
    }}
    .stAlert > div > div > div > p {{ /* Paragraph inside alert */
         text-align: {text_align_css} !important;
         direction: {direction_css} !important;
    }}


    /* =========================================== */
    /* === Chatbot Styles (Floating) === */
    /* =========================================== */

    /* Chatbot Window Container (set by st.markdown with id) */
    #chatbot-window {{
        position: fixed !important;
        bottom: 20px !important; /* Adjust if it overlaps with a toggle button at the bottom */
        {chat_window_position_css} 'left: 20px;' or 'right: 20px;'
        width: 370px !important;   /* Slightly wider */
        max-width: 95% !important;
        height: 500px !important;  /* Slightly taller */
        max-height: calc(100vh - 40px) !important; /* Ensure it doesn't go off-screen */
        background-color: white !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 12px !important; /* Slightly more rounded */
        box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
        z-index: 1000 !important; 
        display: flex !important; 
        flex-direction: column !important;
        transition: transform 0.3s ease-out, opacity 0.3s ease-out; /* For smooth open/close if we use JS */
    }}

    /* Chatbot Header */
    .chatbot-header {{
        background-color: #007bff; /* Streamlit primary color */
        color: white;
        padding: 0.6rem 1rem; /* Adjusted padding */
        border-top-left-radius: 11px; /* Match window radius */
        border-top-right-radius: 11px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        cursor: grab; /* Optional: if you want to make it draggable later */
    }}
    .chatbot-header h5 {{
        margin: 0;
        font-size: 1.05rem; /* Adjusted font size */
        font-weight: 600;
    }}
    /* Style for a close button if implemented with HTML/CSS within the header */
    .chatbot-header .close-btn {{
        background: none;
        border: none;
        color: white;
        font-size: 1.6rem;
        font-weight: bold;
        cursor: pointer;
        padding: 0 0.5rem;
        line-height: 1;
    }}

    /* Messages Area */
    .chatbot-messages-container {{
        flex-grow: 1;
        padding: 15px;
        overflow-y: auto; /* Enable scrolling */
        background-color: #f8f9fa;
        display: flex;
        flex-direction: column-reverse; /* Newest messages at the bottom, scroll from bottom */
    }}
    /* Inner div for messages to allow correct scrolling with column-reverse */
    #message-list-inner {{
        display: flex;
        flex-direction: column; /* Messages stack vertically */
    }}

    /* Message Bubbles */
    .message-bubble {{
        margin-bottom: 12px; /* Increased margin */
        padding: 10px 15px;  /* Increased padding */
        border-radius: 20px; /* More rounded */
        max-width: 85%;
        word-wrap: break-word;
        box-shadow: 0 1px 1px rgba(0,0,0,0.05);
        font-size: 0.95rem;
    }}
    .user-message {{
        background-color: #007bff;
        color: white;
        {user_message_margin_css} /* Handles LTR/RTL alignment */
    }}
    .bot-message {{
        background-color: #e9ecef;
        color: #212529; /* Darker text for better contrast */
        {bot_message_margin_css} /* Handles LTR/RTL alignment */
    }}

    /* Chatbot Input Area */
    .chatbot-input-area {{
        padding: 0.75rem; /* Increased padding */
        border-top: 1px solid #dee2e6;
        background-color: #ffffff;
        display: flex;
        align-items: center;
    }}
    /* Ensure the stTextInput's div takes up a
    .chatbot-input-area div[data-testid="stTextInput"] {{
        flex-grow: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    /* Style the actual input field */
    .chatbot-input-area input[type="text"] {{
        width: 100% !important;
        direction: {direction_css} !important;
        text-align: {text_align_css} !important;
        border-radius: 20px !important; /* Rounded input field */
        border: 1px solid #ced4da !important;
        padding: 0.5rem 1rem !important; /* Padding inside input */
        height: auto !important; /* Let padding define height */
    }}
    /* Style the Streamlit button used for sending */
    .chatbot-input-area div[data-testid="stButton"] > button {{
        {input_button_margin_css}
        flex-shrink: 0 !important;
        border-radius: 50% !important; /* Circular send button */
        width: 40px !important;  /* Fixed size */
        height: 40px !important; /* Fixed size */
        padding: 0 !important; /* Remove padding to center icon/text */
        font-size: 1.2rem !important; /* Adjust icon/text size */
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_analyzer_cached_i18n():
    global analyzer_error_msg;
    print(get_translation(LANG_CODE, "loading_analyzer_once"))
    analyzer_obj = None
    try:
        analyzer_obj = ProblemAnalyzer()
        if not analyzer_obj or not analyzer_obj.clustering_model or not analyzer_obj.topic_model or \
                not analyzer_obj.clustering_model.kmeans_model or \
                not analyzer_obj.clustering_model.column_transformer or \
                not analyzer_obj.clustering_model.sentence_model or \
                not analyzer_obj.topic_model.model:
            analyzer_error_msg = get_translation(LANG_CODE, "analyzer_load_fail_warning", details="some components")
            print(analyzer_error_msg);
            return None
        if analyzer_obj.df_profile_data is None: print(get_translation(LANG_CODE, "profile_data_load_warning"))
        print(get_translation(LANG_CODE, "analyzer_init_success"));
        return analyzer_obj
    except Exception as e_load_analyzer:
        analyzer_error_msg = get_translation(LANG_CODE, "analyzer_init_fatal_error", e=str(e_load_analyzer))
        print(analyzer_error_msg);
        import traceback;
        traceback.print_exc();
        return None


@st.cache_resource
def load_recommender_cached_i18n():
    global recommender_error_msg;
    print(get_translation(LANG_CODE, "loading_recommender_once"))
    recommender_obj = None
    try:
        from src.analysis.recommendation_engine import HISTORICAL_DATA_WITH_ALL_RESULTS_PATH
        recommender_obj = RecommendationEngine(historical_data_path=HISTORICAL_DATA_WITH_ALL_RESULTS_PATH)
        if recommender_obj.historical_data is None:
            recommender_error_msg = get_translation(LANG_CODE, "recommender_load_fail_warning")
            print(recommender_error_msg);
            return None
        print(get_translation(LANG_CODE, "recommender_init_success"));
        return recommender_obj
    except Exception as e_load_recommender:
        recommender_error_msg = get_translation(LANG_CODE, "recommender_init_fatal_error", e=str(e_load_recommender))
        print(recommender_error_msg);
        import traceback;
        traceback.print_exc();
        return None


if analyzer_instance is None and analyzer_error_msg is None: analyzer_instance = load_analyzer_cached_i18n()
if recommender_instance is None and recommender_error_msg is None and analyzer_error_msg is None: recommender_instance = load_recommender_cached_i18n()

# --- Chatbot State and Logic ---
if 'chat_open' not in st.session_state: st.session_state.chat_open = False
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [{"sender": "bot", "text": get_translation(LANG_CODE, "chatbot_greeting")}]
if 'chat_input_key_counter' not in st.session_state: st.session_state.chat_input_key_counter = 0

FAQ_KNOWLEDGE_BASE = {
    "ar": {
        "مساعدة": get_translation("ar", "faq_help_options"),
        "مرحبا": get_translation("ar", "كيف حالك"),

        "كيف اضيف مشكلة": get_translation("ar", "faq_how_to_add_problem"),
        "ما هو المجال": get_translation("ar", "faq_what_is_domain"), # سؤال جديد
        "ما هو التعقيد": get_translation("ar", "faq_what_is_complexity"), # سؤال جديد
        "شكرا": get_translation("ar", "faq_thank_you"),

    },
    "en": {
        "help": get_translation("en", "faq_help_options"),
        "how to add problem": get_translation("en", "faq_how_to_add_problem"),
        "what is domain": get_translation("en", "faq_what_is_domain"), # سؤال جديد
        "what is complexity": get_translation("en", "faq_what_is_complexity"), # سؤال جديد
        "thank you": get_translation("en", "faq_thank_you"),
    }
}


def get_chatbot_response_i18n(user_message, lang_code):
    user_message_lower = user_message.lower().strip()
    current_lang_faq = FAQ_KNOWLEDGE_BASE.get(lang_code, FAQ_KNOWLEDGE_BASE.get("en", {}))
    for keyword, response in current_lang_faq.items():
        if keyword.lower() in user_message_lower: return response
    return get_translation(lang_code, "faq_default_reply")


# --- Chatbot UI ---
# زر التبديل في الشريط الجانبي (أكثر موثوقية في Streamlit)
st.sidebar.markdown("---")  # فاصل
toggle_button_label_sidebar = get_translation(LANG_CODE,
                                              "chatbot_toggle_close" if st.session_state.chat_open else "chatbot_toggle_open")
if st.sidebar.button(toggle_button_label_sidebar, key="sidebar_toggle_chat_button"):
    st.session_state.chat_open = not st.session_state.chat_open
    # لا حاجة لـ st.rerun() هنا إذا كان الشرط الذي يعرض النافذة سيُعاد تقييمه تلقائيًا

# نافذة الشات بوت
if st.session_state.chat_open:
    # استخدام عمود وهمي لمحاولة تثبيت النافذة - هذا قد لا يكون مثاليًا
    # الطريقة الأفضل هي استخدام div مع position:fixed كما في CSS
    # وسيقوم Streamlit بوضع عناصر st.markdown, st.container, st.text_input, st.button داخل هذا الـ div
    st.markdown('<div id="chatbot-window">', unsafe_allow_html=True)  # تطبيق ID للـ CSS

    # Header مع زر إغلاق (ملاحظة: زر الإغلاق هذا لن يعمل عبر Python مباشرة بدون حيل)
    header_cols_chat = st.columns([0.9, 0.1])  # أعمدة لترتيب العنوان وزر الإغلاق
    with header_cols_chat[0]:
        st.markdown(
            f"<div class='chatbot-header' style='border-radius: 7px 7px 0 0;'><h5>{get_translation(LANG_CODE, 'chatbot_title')}</h5></div>",
            unsafe_allow_html=True)
    with header_cols_chat[1]:
        if st.button(get_translation(LANG_CODE, "chatbot_toggle_close"), key="header_close_chat_btn",
                     help="إغلاق نافذة المساعد"):
            st.session_state.chat_open = False
            st.rerun()  # لإعادة تحميل وإخفاء النافذة فورًا

    # Messages Area
    # يجب أن تكون هذه الحاوية قادرة على التمرير وتحديث نفسها عند إضافة رسائل
    message_area = st.container()
    with message_area:
        st.markdown('<div class="chatbot-messages-container" style="height: 300px;">', unsafe_allow_html=True)
        st.markdown('<div id="message-list">', unsafe_allow_html=True)
        for msg in st.session_state.chat_messages:
            bubble_class = "user-message" if msg["sender"] == "user" else "bot-message"
            escaped_text = msg["text"].replace("<", "<").replace(">", ">").replace("\n", "<br>")
            st.markdown(f'<div class="message-bubble {bubble_class}">{escaped_text}</div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    # Input Area
    # استخدام st.form لمنطقة الإدخال لمنع إعادة التحميل عند الكتابة
    with st.form(key=f"chatbot_input_form_{LANG_CODE}_{st.session_state.chat_input_key_counter}", clear_on_submit=True):
        # استخدام أعمدة لوضع حقل الإدخال وزر الإرسال في نفس السطر
        input_cols_chat = st.columns([0.8, 0.2])
        with input_cols_chat[0]:
            user_chat_input = st.text_input(
                get_translation(LANG_CODE, "chatbot_input_placeholder"),
                label_visibility="collapsed"
                # لا نستخدم key هنا لأن clear_on_submit في st.form ستقوم بتفريغه
            )
        with input_cols_chat[1]:
            submit_chat_button = st.form_submit_button(label=get_translation(LANG_CODE, "chatbot_send_button"))

    if submit_chat_button and user_chat_input:
        st.session_state.chat_messages.append({"sender": "user", "text": user_chat_input})
        bot_response = get_chatbot_response_i18n(user_chat_input, LANG_CODE)
        st.session_state.chat_messages.append({"sender": "bot", "text": bot_response})
        # لا حاجة لتغيير المفتاح هنا لأن clear_on_submit=True في st.form
        st.rerun()  # لإعادة تحميل وعرض الرسائل الجديدة

    st.markdown('</div>', unsafe_allow_html=True)  # إغلاق chatbot-window

# --- Main App UI (Problem Input Form & Results) ---
st.title(get_translation(LANG_CODE, "app_header"))
if analyzer_error_msg: st.error(analyzer_error_msg)
if recommender_error_msg: st.error(recommender_error_msg)

if analyzer_instance:
    st.markdown(get_translation(LANG_CODE, "app_subheader"))
    with st.form(key="problem_form"):
        st.subheader(get_translation(LANG_CODE, "form_header"))
        title = st.text_input(get_translation(LANG_CODE, "problem_title_label"),
                              help=get_translation(LANG_CODE, "problem_title_help"))
        description_initial_val = st.text_area(get_translation(LANG_CODE, "description_label"), height=150,
                                               help=get_translation(LANG_CODE, "description_help"))
        col1, col2 = st.columns(2)
        with col1:
            domain_val = st.selectbox(get_translation(LANG_CODE, "domain_label"),
                                      options=get_translation(LANG_CODE, "domain_options"),
                                      help=get_translation(LANG_CODE, "domain_help"))
            complexity_level_val = st.selectbox(get_translation(LANG_CODE, "complexity_label"),
                                                options=get_translation(LANG_CODE, "complexity_options"),
                                                help=get_translation(LANG_CODE, "complexity_help"))
        with col2:
            status_val = st.selectbox(get_translation(LANG_CODE, "status_label"),
                                      options=get_translation(LANG_CODE, "status_options"), index=0)
            problem_source_val = st.text_input(get_translation(LANG_CODE, "problem_source_label"),
                                               help=get_translation(LANG_CODE, "problem_source_help"))
        with st.expander(get_translation(LANG_CODE, "optional_details_header")):
            estimated_cost_str_val = st.text_input(get_translation(LANG_CODE, "estimated_cost_label"),
                                                   help=get_translation(LANG_CODE, "estimated_cost_help"))
            estimated_time_str_val = st.text_input(get_translation(LANG_CODE, "estimated_time_label"),
                                                   help=get_translation(LANG_CODE, "estimated_time_help"))
            overall_budget_str_val = st.text_input(get_translation(LANG_CODE, "overall_budget_label"),
                                                   help=get_translation(LANG_CODE, "overall_budget_help"))
            stakeholders_involved_str_val = st.text_input(get_translation(LANG_CODE, "stakeholders_label"))
            refined_problem_statement_final_str_val = st.text_area(
                get_translation(LANG_CODE, "refined_statement_label"), height=100)

        submit_button_main_form = st.form_submit_button(label=get_translation(LANG_CODE, "submit_button_label"))

    if submit_button_main_form:
        if not title or not description_initial_val:
            st.error(get_translation(LANG_CODE, "error_title_description_missing"))
        else:
            st.markdown("---");
            st.subheader(get_translation(LANG_CODE, "analysis_results_header"))
            problem_data_input = {
                'title': title, 'description_initial': description_initial_val,
                'domain': domain_val if domain_val else None,
                'complexity_level': complexity_level_val if complexity_level_val else None,
                'status': status_val if status_val else None,
                'problem_source': problem_source_val if problem_source_val else None,
                'sentiment_label': None,
                'estimated_cost': estimated_cost_str_val if estimated_cost_str_val else None,
                'overall_budget': overall_budget_str_val if overall_budget_str_val else None,
                'estimated_time_to_implement': estimated_time_str_val if estimated_time_str_val else None,
                'stakeholders_involved': stakeholders_involved_str_val if stakeholders_involved_str_val else None,
                'refined_problem_statement_final': refined_problem_statement_final_str_val if refined_problem_statement_final_str_val else None,
                'active_listening_notes': None, 'key_questions_asked': None, 'initial_hypotheses': None,
                'key_findings_from_analysis': None, 'potential_root_causes_list': None,
                'solution_description': None, 'justification_for_choice': None, 'what_went_well': None,
                'what_could_be_improved': None, 'recommendations_for_future': None, 'key_takeaways': None
            }
            with st.spinner(get_translation(LANG_CODE, "processing_problem_spinner")):
                if analyzer_instance:
                    analysis_output = analyzer_instance.analyze_new_problem(problem_data_input)
                else:
                    st.error(get_translation(LANG_CODE, "app_components_load_fail")); analysis_output = {}

            if analysis_output and not analysis_output.get("error"):
                st.markdown(
                    f"**{get_translation(LANG_CODE, 'input_title_label')}** {analysis_output.get('input_problem_data', {}).get('title')}")
                kmeans_cluster_val = analysis_output.get('kmeans_cluster',
                                                         get_translation(LANG_CODE, "not_specified_placeholder"))
                st.info(f"**{get_translation(LANG_CODE, 'kmeans_cluster_label')}** `{kmeans_cluster_val}`")
                st.markdown(
                    f"**{get_translation(LANG_CODE, 'cluster_summary_label')}**\n{analysis_output.get('cluster_profile_summary', get_translation(LANG_CODE, 'no_summary_available'))}")
                bertopic_topic_val = analysis_output.get('bertopic_topic',
                                                         get_translation(LANG_CODE, "not_specified_placeholder"))
                st.info(f"**{get_translation(LANG_CODE, 'bertopic_topic_label')}** `{bertopic_topic_val}`")
                st.markdown(
                    f"**{get_translation(LANG_CODE, 'topic_summary_label')}**\n{analysis_output.get('topic_profile_summary', get_translation(LANG_CODE, 'no_summary_available'))}")

                if recommender_instance and recommender_instance.historical_data is not None:
                    st.markdown("---");
                    st.subheader(get_translation(LANG_CODE, "recommendations_header"))
                    with st.spinner(get_translation(LANG_CODE, "searching_recommendations_spinner")):
                        recommendations_output = recommender_instance.get_recommendations(analysis_output)

                    if recommendations_output.get("based_on_kmeans_cluster"):
                        st.markdown(f"**{get_translation(LANG_CODE, 'kmeans_recommendations_label')}**")
                        for rec in recommendations_output["based_on_kmeans_cluster"]: st.write(f"- {rec}")

                    if recommendations_output.get("based_on_bertopic_topic"):
                        st.markdown(f"**{get_translation(LANG_CODE, 'bertopic_recommendations_label')}**")
                        for rec in recommendations_output["based_on_bertopic_topic"]: st.write(f"- {rec}")

                    if recommendations_output.get("general_warnings"):
                        for warning_tuple in recommendations_output["general_warnings"]:
                            if isinstance(warning_tuple, tuple) and len(warning_tuple) == 2:
                                warning_key, warning_params = warning_tuple
                                st.warning(get_translation(LANG_CODE, warning_key, **warning_params))
                            else:
                                st.warning(str(warning_tuple))  # Fallback for simple string warnings

                    no_cluster_recs = not recommendations_output.get("based_on_kmeans_cluster")
                    no_topic_recs = not recommendations_output.get("based_on_bertopic_topic")
                    # Show "no specific recommendations" only if no specific recs AND no specific warnings were shown
                    # (general_warnings might include "noise topic" which explains lack of topic recs)
                    has_specific_warnings = any(
                        wt[0] in ["no_kmeans_cluster_recs_warning", "no_bertopic_topic_recs_warning"]
                        for wt in recommendations_output.get("general_warnings", [])
                        if isinstance(wt, tuple)
                    )
                    if no_cluster_recs and no_topic_recs and not has_specific_warnings:
                        st.write(get_translation(LANG_CODE, "no_specific_recommendations"))
                else:
                    st.warning(get_translation(LANG_CODE, "recommender_unavailable_warning"))
            elif analysis_output and analysis_output.get("error"):
                st.error(
                    f"{get_translation(LANG_CODE, 'error_during_analysis', default_text='Error during analysis')}: {analysis_output.get('error')}")
else:
    if not analyzer_error_msg:  # Only show this if there wasn't an initial import/config error
        st.error(get_translation(LANG_CODE, "app_components_load_fail"))

# --- الشريط الجانبي (كما هو) ---
st.sidebar.header(get_translation(LANG_CODE, "sidebar_about_header"))
st.sidebar.info(get_translation(LANG_CODE, "sidebar_about_info"))
st.sidebar.markdown("---")
st.sidebar.subheader(get_translation(LANG_CODE, "sidebar_how_it_works_header"))
st.sidebar.markdown(get_translation(LANG_CODE, "sidebar_how_it_works_steps"))
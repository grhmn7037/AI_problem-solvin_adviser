# src/utils/text_processing.py
import re
import nltk
import string
import pandas as pd
from langdetect import detect, LangDetectException  # *** استيراد جديد ***
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer  # مجذر عربي
from nltk.stem.porter import PorterStemmer  # *** مجذر إنجليزي جديد ***

_nltk_resources_downloaded = False


def download_nltk_resources():
    global _nltk_resources_downloaded
    if _nltk_resources_downloaded:
        return
    resources = ["stopwords", "punkt"]
    for resource in resources:
        try:
            nltk.data.find(f"{'corpora' if resource == 'stopwords' else 'tokenizers'}/{resource}")
            # print(f"NLTK resource '{resource}' already available.")
        except LookupError:
            print(f"Downloading NLTK resource '{resource}'...")
            nltk.download(resource, quiet=True)  # quiet=True لتقليل المخرجات
    _nltk_resources_downloaded = True


download_nltk_resources()

# --- إعداد الموارد اللغوية ---
ARABIC_STOPWORDS = set(stopwords.words('arabic'))
ENGLISH_STOPWORDS = set(stopwords.words('english'))
# يمكنك إضافة كلمات شائعة مخصصة لكل لغة إذا أردت
CUSTOM_AR_STOPWORDS = {"مثل", "ايضا", "كان", "يكون", "أو", "و", "في", "من", "الى", "علي", "حتي", "الخ", "التي", "الذي"}
CUSTOM_EN_STOPWORDS = {"also", "get", "make", "would", "could"}  # أمثلة
ARABIC_STOPWORDS.update(CUSTOM_AR_STOPWORDS)
ENGLISH_STOPWORDS.update(CUSTOM_EN_STOPWORDS)

ARABIC_STEMMER = ISRIStemmer()
ENGLISH_STEMMER = PorterStemmer()


# --- دوال التنظيف الخاصة بكل لغة ---
def normalize_arabic_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub(r'[\u064B-\u0652]', '', text)  # إزالة التشكيل
    text = re.sub(r'ـ+', '', text)  # إزالة التطويل
    return text


def stem_arabic_words(words: list[str]) -> list[str]:
    return [ARABIC_STEMMER.stem(word) for word in words]


def stem_english_words(words: list[str]) -> list[str]:
    return [ENGLISH_STEMMER.stem(word) for word in words]


# --- دوال التنظيف العامة ---
def remove_punctuation_and_digits_generic(text: str) -> str:
    if not isinstance(text, str): return ""
    arabic_punctuation = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ"""
    all_punctuation = string.punctuation + arabic_punctuation
    # الأرقام العربية الشرقية والغربية
    all_digits = string.digits + "٠١٢٣٤٥٦٧٨٩"
    translator = str.maketrans('', '', all_punctuation + all_digits)
    return text.translate(translator)


def remove_urls_emails_hashtags_mentions(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # URLs
    text = re.sub(r'\S*@\S*\s?', '', text)  # Emails
    text = re.sub(r'\@\w+|\#\w+', '', text)  # Hashtags & Mentions
    return text


# --- خط أنابيب المعالجة الرئيسي ---
def preprocess_text_pipeline(text: str,
                             language_code: str = None,
                             use_arabic_stemming: bool = False,
                             use_english_stemming: bool = True) -> str:  # افترض أننا نريد تجذير الإنجليزية افتراضيًا
    if not isinstance(text, str) or pd.isna(text) or text.strip() == '':
        return ""

    # 1. تنظيفات عامة أولية
    text = text.lower()  # مهم للإنجليزية واللغات اللاتينية الأخرى
    text = remove_urls_emails_hashtags_mentions(text)

    # 2. اكتشاف اللغة إذا لم يتم توفيرها
    detected_lang = language_code
    if not detected_lang:
        try:
            # اكتشاف اللغة من جزء من النص (أول 200 حرف مثلاً لزيادة الدقة)
            # langdetect قد يخطئ مع النصوص القصيرة جداً
            sample_text_for_lang_detect = text[:200] if len(text) > 20 else text
            if sample_text_for_lang_detect.strip():
                detected_lang = detect(sample_text_for_lang_detect)
            else:  # إذا كان النص فارغًا بعد التنظيفات الأولية
                return ""
        except LangDetectException:
            print(f"تحذير: لم يتمكن langdetect من تحديد لغة النص: '{text[:50]}...'. سيتم تطبيق التنظيف العام فقط.")
            detected_lang = "unknown"  # أو أي رمز افتراضي

    # 3. معالجة خاصة بالعربية
    if detected_lang == 'ar':
        text = normalize_arabic_text(text)
        text = remove_punctuation_and_digits_generic(text)  # إزالة الترقيم بعد التطبيع
        words = nltk.word_tokenize(text)
        words = [word for word in words if word.lower() not in ARABIC_STOPWORDS and len(word) > 1]
        if use_arabic_stemming:
            words = stem_arabic_words(words)
        text = " ".join(words)
    # 4. معالجة خاصة بالإنجليزية
    elif detected_lang == 'en':
        text = remove_punctuation_and_digits_generic(text)
        words = nltk.word_tokenize(text)  # يتطلب أن يكون النص قد تم تحويله لـ lower() بالفعل
        words = [word for word in words if
                 word not in ENGLISH_STOPWORDS and len(word) > 1]  # الكلمات الشائعة الإنجليزية عادة ما تكون lower
        if use_english_stemming:
            words = stem_english_words(words)
        text = " ".join(words)
    # 5. معالجة للغات الأخرى (فرنسي، كردي) - حاليًا تنظيف عام
    # يمكنك إضافة elif detected_lang == 'fr': أو elif detected_lang == 'ku': هنا
    # مع منطق خاص لكل لغة إذا توفرت الموارد (كلمات شائعة، مجذرات)
    elif detected_lang in ['fr', 'ku']:  # مثال للفرنسية والكردية
        text = remove_punctuation_and_digits_generic(text)
        # لا يوجد تجذير أو إزالة كلمات شائعة مخصصة لهما حاليًا في هذا الكود
        # NLTK يوفر كلمات شائعة فرنسية: stopwords.words('french')
        # للكردية، قد تحتاج للبحث عن قائمة كلمات شائعة.
        # مثال للفرنسية:
        if detected_lang == 'fr':
            french_stopwords = set(stopwords.words('french'))
            words = nltk.word_tokenize(text)
            words = [word for word in words if word.lower() not in french_stopwords and len(word) > 1]
            text = " ".join(words)
        # للكردية، حاليًا لا يوجد معالجة خاصة سوى إزالة الترقيم

    # 6. معالجة عامة للغات غير المعروفة أو غير المدعومة بشكل خاص
    else:  # unknown or other languages
        text = remove_punctuation_and_digits_generic(text)
        # يمكنك تركها كما هي أو تطبيق تنظيفات عامة جدًا
        words = text.split()  # تقسيم بسيط بالمسافات
        text = " ".join(word for word in words if len(word) > 1)

    # 7. تنظيفات نهائية
    text = " ".join(text.split())  # إزالة المسافات البيضاء الزائدة المتعددة
    if text.strip().lower() == "none":  # إزالة كلمة "none" إذا كانت هي كل المتبقي
        text = ""
    else:  # إزالة كلمة "none" المضمنة ككلمة كاملة
        text = re.sub(r'\bnone\b', '', text, flags=re.IGNORECASE)
        text = " ".join(text.split())  # إزالة مسافات زائدة مرة أخرى

    return text.strip()


if __name__ == '__main__':
    # اختبارات
    arabic_text = "السلام عليكم ورحمة الله. هذه مشكلة بالعربية، ويجب إزالة الكلمات الشائعة مثل من و في. الرقم 123."
    english_text = "Hello world from the a new English problem statement, number 456! This is also important."
    french_text = "Bonjour le monde. Ceci est un problème en français avec des mots comme de et la. Numéro 789."
    kurdish_text = "سڵاو جیهان. ئەمە کێشەیەکە بە زمانی کوردی. ژمارە ١٢٣."  # مثال نص كردي (سوراني)
    mixed_text = "مشكلة عربية English problem"
    none_text = "None"

    print(f"AR: '{arabic_text}'\n   -> '{preprocess_text_pipeline(arabic_text, use_arabic_stemming=True)}'\n")
    print(f"EN: '{english_text}'\n   -> '{preprocess_text_pipeline(english_text, use_english_stemming=True)}'\n")
    print(f"FR: '{french_text}'\n   -> '{preprocess_text_pipeline(french_text)}'\n")  # لا يوجد تجذير فرنسي مضاف حاليًا
    print(f"KU: '{kurdish_text}'\n   -> '{preprocess_text_pipeline(kurdish_text)}'\n")  # معالجة عامة للكردية
    print(f"MIXED: '{mixed_text}'\n   -> '{preprocess_text_pipeline(mixed_text)}'\n")  # سيكتشف لغة واحدة على الأرجح
    print(f"NONE_STR: '{none_text}'\n   -> '{preprocess_text_pipeline(none_text)}'\n")
    print(f"EMPTY: ''\n   -> '{preprocess_text_pipeline('')}'\n")
# # src/utils/text_processing.py
# import re
# import nltk
# import string
# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.stem.isri import ISRIStemmer  # مثال على مجذر عربي
#
# _nltk_resources_downloaded = False
#
#
# def download_nltk_resources():
#     global _nltk_resources_downloaded
#     if _nltk_resources_downloaded:
#         return
#
#     try:
#         stopwords.words('arabic')
#         stopwords.words('english')
#         # print("NLTK stopwords for Arabic and English already available.") # يمكن إزالة هذه المخرجات إذا أصبحت مزعجة
#     except LookupError:
#         print("Downloading NLTK stopwords...")
#         nltk.download('stopwords')
#     try:
#         nltk.data.find('tokenizers/punkt')
#         # print("NLTK Punkt tokenizer already available.") # يمكن إزالة هذه المخرجات
#     except LookupError:
#         print("Downloading NLTK Punkt tokenizer...")
#         nltk.download('punkt')
#     _nltk_resources_downloaded = True
#
#
# download_nltk_resources()
#
# ARABIC_STOPWORDS = set(stopwords.words('arabic'))
# ENGLISH_STOPWORDS = set(stopwords.words('english'))
# ALL_STOPWORDS = ARABIC_STOPWORDS.union(ENGLISH_STOPWORDS)
# CUSTOM_STOPWORDS = {"مثل", "ايضا", "كان", "يكون", "أو", "و", "في", "من", "الى", "علي", "حتي", "الخ", "التي", "الذي"}
# ALL_STOPWORDS.update(CUSTOM_STOPWORDS)
#
#
# def normalize_arabic_text(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     text = re.sub("[إأآا]", "ا", text)
#     text = re.sub("ى", "ي", text)
#     text = re.sub("ؤ", "و", text)
#     text = re.sub("ئ", "ي", text)
#     text = re.sub("ة", "ه", text)
#     text = re.sub("گ", "ك", text)
#     text = re.sub(r'[\u064B-\u0652]', '', text)
#     text = re.sub(r'ـ+', '', text)
#     return text
#
#
# def remove_punctuation_and_digits(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     arabic_punctuation = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ"""
#     all_punctuation = string.punctuation + arabic_punctuation
#     translator = str.maketrans('', '', all_punctuation + string.digits)
#     return text.translate(translator)
#
#
# def remove_stopwords(text: str, custom_stopwords_list: set = None) -> str:
#     if not isinstance(text, str):
#         return ""
#     stopwords_to_use = ALL_STOPWORDS
#     if custom_stopwords_list:
#         stopwords_to_use = stopwords_to_use.union(custom_stopwords_list)
#     words = nltk.word_tokenize(text)
#     filtered_words = [word for word in words if word.lower() not in stopwords_to_use and len(word) > 1]
#     return " ".join(filtered_words)
#
#
# def stem_text_arabic(text: str) -> str:
#     if not isinstance(text, str):
#         return ""
#     stemmer = ISRIStemmer()
#     words = nltk.word_tokenize(text)
#     stemmed_words = [stemmer.stem(word) for word in words]
#     return " ".join(stemmed_words)
#
#
# def remove_specific_word_from_text(text: str, word_to_remove: str = "none") -> str:
#     """
#     يزيل كلمة معينة (ككلمة كاملة، غير حساسة لحالة الأحرف) من النص.
#     """
#     if not isinstance(text, str) or text.strip() == "":
#         return text
#
#     # استخدام تعبير عادي (\b لحدود الكلمة) لضمان إزالة الكلمة المستقلة فقط
#     # re.IGNORECASE لجعل البحث غير حساس لحالة الأحرف
#     pattern = r'\b' + re.escape(word_to_remove) + r'\b'
#     cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
#
#     # إزالة المسافات البيضاء المزدوجة أو المسافات في بداية/نهاية النص التي قد تنتج عن الإزالة
#     cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
#     return cleaned_text
#
#
# def preprocess_text_pipeline(text: str, use_stemming: bool = False) -> str:
#     if not isinstance(text, str) or pd.isna(text):
#         return ""
#
#     text = text.lower()
#     text = normalize_arabic_text(text)
#     text = remove_punctuation_and_digits(text)
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#     text = re.sub(r'\@\w+|\#\w+', '', text)
#     text = " ".join(text.split())  # إزالة المسافات البيضاء الزائدة أولاً
#     text = remove_stopwords(text)
#
#     # معالجة إذا كان النص بأكمله "none"
#     if text.strip() == "none":
#         text = ""
#     else:
#         # إذا لم يكن النص بأكمله "none"، قم بإزالة تكرارات "none" المضمنة
#         text = remove_specific_word_from_text(text, "none")
#
#     if use_stemming:
#         text = stem_text_arabic(text)
#
#     return text.strip()
#
#
# if __name__ == '__main__':
#     sample_arabic_text_with_none = "None"
#     sample_arabic_text_complex = "السلامُ عليكمْ. هذا مثالٌ none آخر."
#     problematic_text_from_data1 = "حب جهه واحده none none none none none"
#     problematic_text_from_data2 = "اختناق مروري none none none none none"
#     problematic_text_from_data3 = "مرحبا none بالعالم"
#     empty_text = ""
#     text_with_only_spaces = "   "
#
#     print(
#         f"Original: '{sample_arabic_text_with_none}' -> Processed: '{preprocess_text_pipeline(sample_arabic_text_with_none)}'")
#     print(
#         f"Original: '{sample_arabic_text_complex}' -> Processed: '{preprocess_text_pipeline(sample_arabic_text_complex)}'")
#     print(
#         f"Original from data1: '{problematic_text_from_data1}' -> Processed: '{preprocess_text_pipeline(problematic_text_from_data1)}'")
#     print(
#         f"Original from data2: '{problematic_text_from_data2}' -> Processed: '{preprocess_text_pipeline(problematic_text_from_data2)}'")
#     print(
#         f"Original from data3: '{problematic_text_from_data3}' -> Processed: '{preprocess_text_pipeline(problematic_text_from_data3)}'")
#     print(f"Original empty: '{empty_text}' -> Processed: '{preprocess_text_pipeline(empty_text)}'")
#     print(
#         f"Original spaces: '{text_with_only_spaces}' -> Processed: '{preprocess_text_pipeline(text_with_only_spaces)}'")
#     print(f"Original None (Python): {None} -> Processed: '{preprocess_text_pipeline(None)}'")
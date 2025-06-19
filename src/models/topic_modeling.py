# src/models/topic_modeling.py
import pandas as pd
import numpy as np
import os

# لا نحتاج joblib هنا لأن BERTopic لديه طريقته الخاصة للحفظ والتحميل (save/load)
# ولكننا سنحتاج bertopic و sentence_transformers إذا أردنا إعادة التدريب داخل الكلاس لاحقًا.

# من المفترض أن تكون هذه المسارات نسبية لجذر المشروع عند التشغيل أو يتم تمريرها
DEFAULT_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_MODELS_DIR = os.path.join(DEFAULT_PROJECT_ROOT, 'data', 'models')
DEFAULT_BERTOPIC_MODEL_PATH = os.path.join(DEFAULT_MODELS_DIR, 'bertopic_model.pkl')

# نحتاج إلى استيراد BERTopic للتحميل
# تأكد من أن bertopic و sentence_transformers مثبتتان في بيئتك
try:
    from bertopic import BERTopic
    # from sentence_transformers import SentenceTransformer # قد لا نحتاجه مباشرة هنا إذا كان النموذج محملًا بالكامل
except ImportError:
    print("تحذير: مكتبة BERTopic أو SentenceTransformer غير مثبتة.")
    print("يرجى تثبيتها: pip install bertopic sentence-transformers")
    BERTopic = None  # لتعريف المتغير وتجنب أخطاء لاحقة إذا فشل الاستيراد


class ProblemTopicModel:
    def __init__(self, model_path: str = DEFAULT_BERTOPIC_MODEL_PATH):
        """
        تهيئة نموذج تحليل الموضوعات. يقوم بتحميل نموذج BERTopic من المسار المحدد.

        Args:
            model_path (str): مسار ملف نموذج BERTopic المحفوظ (.pkl).
        """
        self.model = None
        self.model_path = model_path
        if BERTopic is not None:  # فقط حاول التحميل إذا تم استيراد BERTopic بنجاح
            self.load_model(self.model_path)
        else:
            print("لا يمكن تهيئة ProblemTopicModel لأن مكتبة BERTopic غير متاحة.")

    def load_model(self, model_path: str) -> bool:
        """
        تحميل نموذج BERTopic من ملف.

        Args:
            model_path (str): مسار ملف النموذج.

        Returns:
            bool: True إذا تم التحميل بنجاح، False خلاف ذلك.
        """
        if BERTopic is None:
            print("خطأ: مكتبة BERTopic غير متاحة، لا يمكن تحميل النموذج.")
            return False
        try:
            print(f"محاولة تحميل نموذج BERTopic من: {model_path}")
            if not os.path.exists(model_path):
                print(f"خطأ: ملف النموذج غير موجود في المسار: {model_path}")
                self.model = None
                return False
            # BERTopic.load يتطلب أحيانًا تحديد embedding_model إذا لم يتم حفظه بالكامل
            # ولكن إذا استخدمنا serialization="pickle" عند الحفظ، يجب أن يكون كل شيء مضمنًا.
            self.model = BERTopic.load(model_path)
            print("تم تحميل نموذج BERTopic بنجاح.")
            return True
        except FileNotFoundError:
            print(f"خطأ في تحميل نموذج BERTopic: ملف غير موجود - {model_path}")
            self.model = None
        except Exception as e:
            print(f"خطأ عام أثناء تحميل نموذج BERTopic: {e}")
            self.model = None
        return False

    def get_topics_for_texts(self, texts: list[str]) -> tuple[list[int], np.ndarray]:
        """
        يحدد الموضوعات والاحتمالات لقائمة من النصوص الجديدة.

        Args:
            texts (list[str]): قائمة بالنصوص (يجب أن تكون نصوصًا نظيفة،
                                 كما هو الحال في عمود 'processed_text').

        Returns:
            tuple[list[int], np.ndarray]:
                - قائمة بأرقام الموضوعات المعينة لكل نص.
                - مصفوفة NumPy باحتمالات انتماء كل نص للموضوعات المختلفة (إذا كان النموذج يدعمها).
                  BERTopic.transform يعيد (topics, probabilities).
        """
        if self.model is None:
            print("خطأ: نموذج BERTopic غير محمل. لا يمكن تحديد الموضوعات.")
            return [], np.array([])

        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            print("خطأ: الإدخال يجب أن يكون قائمة من السلاسل النصية.")
            return [], np.array([])

        if not texts:
            print("قائمة النصوص المدخلة فارغة.")
            return [], np.array([])

        print(f"\nتحديد الموضوعات لـ {len(texts)} نص(نصوص) جديدة...")
        try:
            # BERTopic.transform يتوقع قائمة من النصوص
            topics, probabilities = self.model.transform(texts)
            print(f"تم تحديد الموضوعات بنجاح.")
            return topics, probabilities
        except Exception as e:
            print(f"حدث خطأ أثناء استدعاء model.transform(): {e}")
            return [], np.array([])

    def get_topic_info_df(self) -> pd.DataFrame:
        """
        يعيد DataFrame بمعلومات عن جميع الموضوعات المكتشفة بواسطة النموذج المحمل.
        """
        if self.model is None:
            print("خطأ: نموذج BERTopic غير محمل.")
            return pd.DataFrame()
        try:
            return self.model.get_topic_info()
        except Exception as e:
            print(f"حدث خطأ أثناء الحصول على معلومات الموضوعات: {e}")
            return pd.DataFrame()

    def get_keywords_for_topic(self, topic_id: int) -> list[tuple[str, float]]:
        """
        يعيد قائمة بالكلمات الرئيسية وأوزانها لموضوع معين.
        """
        if self.model is None:
            print("خطأ: نموذج BERTopic غير محمل.")
            return []
        try:
            if topic_id == -1 and "-1_ ভট্টাচার্য್ಯ" in self.model.get_topic_info()[
                "Name"].values:  # معالجة خاصة لاسم غريب للضوضاء أحيانًا
                return [("Noise/Outlier Topic", 1.0)]
            return self.model.get_topic(topic_id)
        except Exception as e:
            print(f"حدث خطأ أثناء الحصول على الكلمات الرئيسية للموضوع {topic_id}: {e}")
            return []


# --- مثال للاستخدام (للاختبار) ---
if __name__ == '__main__':
    print("--- بدء اختبار ProblemTopicModel ---")

    # إنشاء كائن من الكلاس (سيحاول تحميل نموذج BERTopic المحفوظ)
    topic_model_instance = ProblemTopicModel()

    if topic_model_instance.model:
        print("\n--- اختبار تحديد الموضوعات لنصوص جديدة ---")
        sample_texts = [
            "هناك مشكلة في تفاعل الطلاب مع النظام التعليمي الجديد",  # يجب أن يكون قريبًا من Topic 1
            "جهازي المحمول لا يعمل بشكل جيد بعد سقوطه",  # يجب أن يكون قريبًا من Topic 2
            "سيارتي لا تبدأ اليوم صباحا والمفتاح لا يدور"  # قد يكون Topic 0 أو -1
        ]

        # افترض أن هذه النصوص قد تم تنظيفها مسبقًا بنفس طريقة 'processed_text'
        # للحصول على أفضل النتائج. هنا نستخدمها كما هي للاختبار السريع.

        predicted_topics, predicted_probs = topic_model_instance.get_topics_for_texts(sample_texts)

        if predicted_topics:  # إذا لم تكن القائمة فارغة
            for i, text in enumerate(sample_texts):
                print(f"\nالنص: \"{text}\"")
                print(f"  الموضوع المتوقع: {predicted_topics[i]}")
                # يمكنك عرض الاحتمالات إذا أردت (predicted_probs[i])

            print("\n--- معلومات عن الموضوعات من النموذج المحمل ---")
            all_topics_info = topic_model_instance.get_topic_info_df()
            if not all_topics_info.empty:
                # display(all_topics_info)  # يتطلب أن تكون في بيئة تدعم display مثل Jupyter أو IPython
                print(all_topics_info.to_string())  # .to_string() لعرض الـ DataFrame كاملاً بشكل أفضل في الطرفية
                # أو ببساطة:
                # print(all_topics_info)
                print("\nالكلمات الرئيسية للموضوع 0 (إذا كان موجودًا):")
                print(topic_model_instance.get_keywords_for_topic(0))
            else:
                print("لم يتم العثور على معلومات الموضوعات.")
    else:
        print("\nلم يتم تحميل نموذج BERTopic. لا يمكن إجراء اختبار التنبؤ.")

    print("\n--- انتهاء اختبار ProblemTopicModel ---")